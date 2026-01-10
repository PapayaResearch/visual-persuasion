import os
import io
import re
import time
import random
import csv
import json
import logging
import threading
import dataclasses
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from itertools import combinations
from utils.wrappers import ImageModel, LanguageModel


@dataclasses.dataclass
class VisualNudgeCompetition:
    """
    Orchestrates paired contests between images, iteratively improving losers until equilibrium.
    Each pair of images competes, and the loser is improved based on judge feedback.
    Process continues until neither image can be improved further (equilibrium).
    """
    name: str
    base_prior: str
    image_editing_model: ImageModel
    judge_prompts: list[str]
    evaluator_model: LanguageModel
    equilibrium_threshold: float
    min_rounds_before_equilibrium: int
    max_rounds_per_pair: int
    tie_breaking_strategy: str
    use_last_winner_as_base: bool
    # For optimizer flow
    optimizer_model: LanguageModel
    # For proposer/selector flow
    num_improvement_proposals: int
    proposer_model: LanguageModel
    selector_model: LanguageModel
    # Whether to use proposer/selector or optimizer
    use_candidates: bool

    def __post_init__(self):
        """Initialize tracking variables."""
        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039
        self._counter_lock = threading.Lock()
        self.contest_history = {}

    def _compose_prompt(self, instruction: Optional[str] = None) -> str:
        """
        Combine the base template with the current instruction.
        Falls back to base_prior when instruction is empty.
        """
        if instruction and (len(instruction.strip()) > 0):
            return "%s\n\n%s" % (self.base_prior.strip(), instruction)
        elif self.base_prior:
            return self.base_prior.strip()

    def _conduct_contest(
        self,
        image_a_bytes: bytes,
        image_b_bytes: bytes,
        image_a_name: str,
        image_b_name: str
    ) -> tuple[str, float, str]:
        """
        Conduct a multi-judge contest between two images.
        Returns: (winner_name, winner_score, aggregated_feedback)
        winner_score is the proportion of consistent judges preferring the winner (0-1)
        """
        contest_start = time.time()
        logging.info(f"\n🥊 CONTEST: {image_a_name} vs {image_b_name}\n")

        def evaluate_single(judge_id: int, is_a_first: bool):
            """Single judge evaluation"""
            judge_start = time.time()

            images = [image_a_bytes, image_b_bytes] if is_a_first else [image_b_bytes, image_a_bytes]
            choice_map = {
                "first": image_a_name if is_a_first else image_b_name,
                "second": image_b_name if is_a_first else image_a_name,
            }

            logging.info(f"Judge {judge_id}: Evaluating with {image_a_name} as {'first' if is_a_first else 'second'} image.\n")

            evaluation = self.evaluator_model.get_response(
                images=images,
                judge_prompt=self.judge_prompts[judge_id],
                metadata="Context: the user is looking for a(n) %s." % image_a_name.split("_")[0].lower()
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            judge_duration = time.time() - judge_start
            logging.debug(f"  ⏱️  Judge {judge_id} ({image_a_name} {'1st' if is_a_first else '2nd'}): {judge_duration:.2f}s")
            logging.info(f"Judge {judge_id}: Chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason)

        def break_tie(image_a_name: str, image_b_name: str):
            """Apply tie-breaking strategy"""
            if self.tie_breaking_strategy == "first":
                winner = image_a_name
            elif self.tie_breaking_strategy == "second":
                winner = image_b_name
            elif self.tie_breaking_strategy == "random":
                winner = random.choice([image_a_name, image_b_name])
            else:
                raise ValueError(f"Unknown tie-breaking strategy: {self.tie_breaking_strategy}")
            return winner

        # Run all evaluations in parallel
        judge_results = {}  # judge_id -> {True: result, False: result}

        num_judges = len(self.judge_prompts)
        with ThreadPoolExecutor(max_workers=min(num_judges * 2, 8)) as eval_executor:
            future_to_judge = {}
            for judge_id in range(num_judges):
                for is_a_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, judge_id, is_a_first)
                    future_to_judge[future] = (judge_id, is_a_first)

            for future in as_completed(future_to_judge):
                judge_id, is_a_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_a_first] = result

        # Aggregate consistent judges
        votes = {image_a_name: 0, image_b_name: 0}
        feedback_by_choice = {image_a_name: [], image_b_name: []}
        total_consistent_judges = 0

        for judge_id, results in judge_results.items():
            result_a_first = results.get(True)
            result_b_first = results.get(False)

            choice_a_first, reason_a_first = result_a_first
            choice_b_first, reason_b_first = result_b_first

            # Only count consistent judges
            if choice_a_first == choice_b_first:
                logging.info(f"Judge {judge_id}: Consistent - chose '{choice_a_first}'\n")
                total_consistent_judges += 1
                votes[choice_a_first] += 1
                feedback_by_choice[choice_a_first].append(reason_a_first)
            else:
                logging.warning(f"Judge {judge_id}: Inconsistent - skipping.\n")

        # Determine winner
        if total_consistent_judges == 0:
            logging.warning("No consistent judges. Applying tie-breaking strategy.\n")
            winner_score = 0.5
            feedback = "No consistent preference detected."
            winner = break_tie(image_a_name, image_b_name)
        else:
            winner = max(votes, key=votes.get)
            winner_score = votes[winner] / total_consistent_judges
            feedback = "\n".join(feedback_by_choice[winner])

            # Check for 50-50 tie and apply tie-breaking
            if winner_score == 0.5:
                logging.warning("Judges split 50-50. Applying tie-breaking strategy.\n")
                winner = break_tie(image_a_name, image_b_name)

            logging.info(f"🏆 WINNER: {winner} ({votes[winner]}/{total_consistent_judges} = {winner_score:.2%})\n")

        contest_duration = time.time() - contest_start
        logging.debug(f"⏱️  Contest completed: {contest_duration:.2f}s")

        return winner, winner_score, feedback

    def _visualize_competition(
        self,
        pair_name: str,
        base_a: str,
        base_b: str,
        round_history: list[dict],
        results_dir: str,
        viz_path: str
    ):
        """
        Create grid display of competition progress.
        Columns: Round number, Contestant A, Contestant B
        Rows: Each round of competition
        """
        num_rounds = len(round_history)
        if num_rounds == 0:
            return

        # Create figure with 3 columns (round label, image A, image B)
        fig, axes = plt.subplots(num_rounds, 3, figsize=(15, 5 * num_rounds))

        # Handle single round case
        if num_rounds == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f"Competition Progress: {base_a} vs {base_b}", fontsize=16, fontweight="bold", y=1.02)

        for round_idx, round_data in enumerate(round_history):
            round_num = round_data["round"]
            winner = round_data["winner"]
            score = round_data["score"]

            # Round label column
            ax_label = axes[round_idx, 0]
            ax_label.text(0.5, 0.5, f"Round {round_num}\n{winner} wins\n{score:.1%}",
                        ha="center", va="center", fontsize=14, fontweight="bold")
            ax_label.axis("off")

            # Image A column
            ax_a = axes[round_idx, 1]
            img_a = round_data["image_a"]
            ax_a.imshow(img_a)
            ax_a.set_title(f"{base_a}", fontsize=12, fontweight="bold" if winner == base_a else "normal")
            ax_a.axis("off")

            # Image B column
            ax_b = axes[round_idx, 2]
            img_b = round_data["image_b"]
            ax_b.imshow(img_b)
            ax_b.set_title(f"{base_b}", fontsize=12, fontweight="bold" if winner == base_b else "normal")
            ax_b.axis("off")

        plt.tight_layout()

        # Save visualization
        fig.savefig(viz_path, dpi=300, bbox_inches="tight")
        logging.info(f"📊 Visualization saved to {viz_path}\n")
        plt.close(fig)

    def _select_best_proposal(
        self,
        candidate_images: list[dict],
        feedback: str = ""
    ) -> dict:
        """
        Use selector model to choose the best proposal from candidates.
        Returns: The best candidate dict
        """
        logging.info(f"\n🎯 SELECTING BEST PROPOSAL from {len(candidate_images)} candidates\n")

        # Prepare images and descriptions for selector
        image_bytes_list = [c["image_bytes"] for c in candidate_images]
        descriptions = [f"Candidate {i+1}: {c['prompt']}" for i, c in enumerate(candidate_images)]

        selector_response = self.selector_model.get_response(
            images=image_bytes_list,
            candidate_descriptions="\n".join(descriptions),
            num_candidates=len(candidate_images),
            judge_feedback=feedback
        )

        selected_idx = int(selector_response.choice) - 1  # Assuming 1-indexed

        best_candidate = candidate_images[selected_idx]
        logging.info(f"✅ Selected candidate {selected_idx+1}: {best_candidate['prompt']}\n")

        return best_candidate

    def _improve_loser_with_candidates(
        self,
        loser_name: str,
        loser_image_bytes: bytes,
        loser_prompt: str,
        feedback: str,
        round_num: int,
        results_dir: str,
        pair_name: str,
        history_of_prompts: list[str],
        original_image_bytes: bytes
    ) -> tuple[bytes, str, Image.Image]:
        """
        Generate improved versions of the losing image based on judge feedback.
        Uses proposer/selector flow after the initial zero-shot edit.
        Returns: (best_improved_bytes, best_improved_prompt, best_improved_image)
        """
        improve_start = time.time()
        logging.info(f"\n🔧 IMPROVING LOSER (Round {round_num})\n")
        logging.info(f"Previous prompt: {loser_prompt}\n")
        logging.info(f"Judge feedback:\n{feedback}\n")

        history_entries = list(history_of_prompts)
        has_prior_edits = len(history_of_prompts) > 0
        history_text = "\n".join([f"  - {p}" for p in history_entries]) if history_entries else "None"
        logging.info(f"Edit history:\n{history_text}\n")

        is_original_state = (loser_image_bytes == original_image_bytes) and not has_prior_edits
        if is_original_state:
            logging.info("Applying base prior zero-shot edit before invoking proposer/selector.\n")
            editing_prompt = self._compose_prompt(None)
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                editing_prompt,
                loser_image_bytes,
                original_image_bytes
            )

            with self._counter_lock:
                self._total_num_images_generated += 1

            optimized_path = os.path.join(
                results_dir,
                f"{pair_name}_{loser_name}_round-{round_num}_optimized.jpg"
            )
            edited_image.save(optimized_path)
            logging.info(f"Saved zero-shot optimized candidate to {optimized_path}\n")

            zero_shot_path = os.path.join(results_dir, f"{pair_name}_{loser_name}_zero-shot.jpg")
            if not os.path.isfile(zero_shot_path):
                edited_image.save(zero_shot_path)

            prompt_txt_path = os.path.join(results_dir, f"{pair_name}_{loser_name}_zero-shot_prompt.txt")
            with open(prompt_txt_path, "w") as prompt_file:
                prompt_file.write(editing_prompt)

            total_cost = self._total_num_images_generated * self._cost_per_image_generated
            logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")
            logging.debug(f"Total images: {self._total_num_images_generated}, Cost: ${total_cost:.2f}")

            improve_duration = time.time() - improve_start
            logging.debug(f"⏱️  Improvement phase: {improve_duration:.2f}s")

            return (
                edited_image_bytes,
                editing_prompt,
                edited_image
            )

        proposer_response = self.proposer_model.get_response(
            current_prompt=loser_prompt,
            history_of_prompts=history_text,
            current_iteration=round_num,
            # judge_feedback=feedback,
            total_iterations=self.min_rounds_before_equilibrium,
            num_proposals=self.num_improvement_proposals,
            metadata="The product here is a(n) %s; make sure your edits still center this product." % pair_name.split("_")[1].lower()
        )
        logging.debug(f"⏱️  Proposal category is: {pair_name.split('_')[1]}")

        candidate_prompts = proposer_response.candidate_prompts
        logging.info(f"Generated {len(candidate_prompts)} improvement candidates:\n")
        for i, prompt in enumerate(candidate_prompts):
            logging.info(f"  Candidate {i+1}: {prompt}\n")

        def generate_single_image(i: int, prompt: str):
            """Generate a single improved image"""
            logging.info(f"Generating improved image {i+1}/{len(candidate_prompts)}\n")
            editing_prompt = self._compose_prompt(prompt)
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                editing_prompt,
                loser_image_bytes,
                original_image_bytes
            )

            if edited_image is None or edited_image_bytes is None:
                logging.warning(f"Candidate {i+1} generation failed; skipping.\n")
                return None

            with self._counter_lock:
                self._total_num_images_generated += 1

            candidate_path = os.path.join(results_dir, f"{pair_name}_{loser_name}_round-{round_num}_candidate-{i+1}.jpg")
            edited_image.save(candidate_path)

            return {
                "prompt": prompt,
                "image": edited_image,
                "image_bytes": edited_image_bytes,
                "save_path": candidate_path,
                "candidate_idx": i,
                "full_prompt": editing_prompt
            }

        candidate_images = []
        with ThreadPoolExecutor(max_workers=min(len(candidate_prompts), 4)) as img_executor:
            futures = [img_executor.submit(generate_single_image, i, prompt) for i, prompt in enumerate(candidate_prompts)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    candidate_images.append(result)

        if not candidate_images:
            logging.warning("All candidate generations failed; keeping previous image.\n")
            return loser_image_bytes, loser_prompt, Image.open(io.BytesIO(loser_image_bytes))

        candidate_images.sort(key=lambda c: c["candidate_idx"])
        best_candidate = self._select_best_proposal(candidate_images, feedback=feedback)

        total_cost = self._total_num_images_generated * self._cost_per_image_generated
        logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")
        logging.debug(f"Total images: {self._total_num_images_generated}, Cost: ${total_cost:.2f}")

        prompt_save_path = os.path.join(
            results_dir,
            f"{pair_name}_{loser_name}_round-{round_num}_prompt.txt"
        )
        with open(prompt_save_path, "w") as prompt_file:
            prompt_file.write(best_candidate.get("full_prompt", self._compose_prompt(best_candidate["prompt"])))

        optimized_path = os.path.join(
            results_dir,
            f"{pair_name}_{loser_name}_round-{round_num}_optimized.jpg"
        )
        best_candidate["image"].save(optimized_path)
        logging.info(f"Saved optimized candidate to {optimized_path}\n")

        improve_duration = time.time() - improve_start
        logging.debug(f"⏱️  Improvement phase: {improve_duration:.2f}s")

        return (
            best_candidate["image_bytes"],
            best_candidate["prompt"],
            best_candidate["image"]
        )

    def _improve_loser_without_candidates(
        self,
        loser_name: str,
        loser_image_bytes: bytes,
        loser_prompt: str,
        feedback: str,
        round_num: int,
        results_dir: str,
        pair_name: str,
        history_of_prompts: list[str],
        original_image_bytes: bytes
    ) -> tuple[bytes, str, Image.Image]:
        """
        Improve the losing image by asking the optimizer for a better prompt and regenerating once.
        Returns: (improved_bytes, improved_prompt, improved_image)
        """
        improve_start = time.time()
        logging.info(f"\n🔧 IMPROVING LOSER (Round {round_num})\n")
        logging.info(f"Previous prompt: {loser_prompt}\n")
        logging.info(f"Judge feedback:\n{feedback}\n")

        # Format history for optimizer visibility
        history_text = "\n".join([f"  - {p}" for p in history_of_prompts]) if history_of_prompts else "None"
        logging.info(f"Edit history:\n{history_text}\n")

        is_original_state = loser_image_bytes == original_image_bytes
        if is_original_state:
            improved_prompt = None
            logging.info("Applying base prior before switching to optimizer-driven edits.\n")
        else:
            optimizer_input = {
                "current_prompt": self._compose_prompt(loser_prompt),
                "current_image": loser_image_bytes,
                "history_of_prompts": history_text,
                "judge_feedback": feedback or "No feedback provided.",
                "metadata": "The product here is a(n) %s." % pair_name.split("_")[1]
            }
            optimizer_response = self.optimizer_model.get_response(**optimizer_input)
            improved_prompt = optimizer_response.new_prompt.strip()

            logging.info(f"Optimizer prompt:\n{improved_prompt}\n")

        editing_prompt = self._compose_prompt(improved_prompt)
        edited_image, edited_image_bytes = self.image_editing_model.edit(
            editing_prompt,
            loser_image_bytes,
            original_image_bytes
        )

        with self._counter_lock:
            self._total_num_images_generated += 1

        improved_path = os.path.join(
            results_dir,
            f"{pair_name}_{loser_name}_round-{round_num}_optimized.jpg"
        )
        edited_image.save(improved_path)
        editing_prompt_path = os.path.join(
            results_dir,
            f"{pair_name}_{loser_name}_round-{round_num}_optimized_prompt.txt"
        )
        with open(editing_prompt_path, "w") as outfile:
            outfile.write(f"Prompt used for optimization:\n\n{editing_prompt}")

        first_improved_path = os.path.join(
            results_dir,
            f"{pair_name}_{loser_name}_zero-shot.jpg"
        )
        if not os.path.isfile(first_improved_path):
            edited_image.save(first_improved_path)

        logging.info(f"Saved optimized candidate to {improved_path}\n")

        total_cost = self._total_num_images_generated * self._cost_per_image_generated
        logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")
        logging.debug(f"Total images: {self._total_num_images_generated}, Cost: ${total_cost:.2f}")

        improve_duration = time.time() - improve_start
        logging.debug(f"⏱️  Improvement phase: {improve_duration:.2f}s")

        return (
            edited_image_bytes,
            improved_prompt,
            edited_image
        )

    def _improve_loser(self, **kwargs) -> tuple[bytes, str, Image.Image]:
        """
        Chooses the improvement strategy based on configuration.
        """
        if self.use_candidates:
            return self._improve_loser_with_candidates(**kwargs)
        else:
            return self._improve_loser_without_candidates(**kwargs)

    def _run_paired_contest(
        self,
        image_a_path: str,
        image_b_path: str,
        results_dir: str,
        pair_idx: int,
        total_pairs: int
    ) -> dict:
        """
        Run a paired contest between two images until equilibrium is reached.
        Only the loser is improved each round.
        Returns final state of both images.
        """
        pair_start = time.time()

        base_a = os.path.splitext(os.path.basename(image_a_path))[0]
        base_b = os.path.splitext(os.path.basename(image_b_path))[0]
        pair_name = f"pair-{pair_idx+1}_{base_a}_vs_{base_b}"

        logging.info(f"\n{'='*80}\n")
        logging.info(f"PAIRED CONTEST {pair_idx+1}/{total_pairs}: {base_a} vs {base_b}\n")
        logging.info(f"{'='*80}\n")

        # Load initial images
        with open(image_a_path, "rb") as f:
            image_a_bytes = f.read()
        with open(image_b_path, "rb") as f:
            image_b_bytes = f.read()

        image_a_obj = Image.open(io.BytesIO(image_a_bytes))
        image_b_obj = Image.open(io.BytesIO(image_b_bytes))

        # Initialize state with edit history and champion tracking
        state_a = {
            "bytes": image_a_bytes,
            "prompt": self.base_prior,
            "image": image_a_obj,
            "name": base_a,
            "edit_history": [],  # Track all prompts used
            "champion_bytes": image_a_bytes,
            "champion_prompt": self.base_prior,
            "champion_image": image_a_obj.copy()
        }
        state_b = {
            "bytes": image_b_bytes,
            "prompt": self.base_prior,
            "image": image_b_obj,
            "name": base_b,
            "edit_history": [],
            "champion_bytes": image_b_bytes,
            "champion_prompt": self.base_prior,
            "champion_image": image_b_obj.copy()
        }

        # Save originals
        state_a["image"].save(os.path.join(results_dir, f"{pair_name}_{base_a}_original.jpg"))
        state_b["image"].save(os.path.join(results_dir, f"{pair_name}_{base_b}_original.jpg"))

        image_a_bytes_original = state_a["bytes"]
        image_b_bytes_original = state_b["bytes"]

        # Contest loop
        round_num = 0
        equilibrium_reached = False
        contest_history = []
        visualization_history = []  # For matplotlib visualization

        viz_path = os.path.join(results_dir, f"{pair_name}_visualization.png")

        if os.path.isfile(viz_path):
            logging.info(f"The final visualization for {pair_name} already exists. Skipping contest.\n")
            return {
                "pair_name": pair_name,
                "image_a": base_a,
                "image_b": base_b,
                "rounds": 0,
                "equilibrium": False,
                "history": [],
                "final_state_a": state_a,
                "final_state_b": state_b
            }

        while round_num < self.max_rounds_per_pair and not equilibrium_reached:
            round_num += 1
            logging.info(f"\n{'='*60}\n")
            logging.info(f"ROUND {round_num}/{self.max_rounds_per_pair}\n")
            logging.info(f"{'='*60}\n")

            # Conduct contest
            winner_name, winner_score, feedback = self._conduct_contest(
                image_a_bytes=state_a["bytes"],
                image_b_bytes=state_b["bytes"],
                image_a_name=base_a,
                image_b_name=base_b
            )

            # Record history
            round_info = {
                "round": round_num,
                "pair_name": pair_name,
                "winner": winner_name,
                "score": winner_score,
                "feedback": feedback.split("\n")
            }
            with open(os.path.join(results_dir, f"{pair_name}_round-{round_num}_info.json"), "w") as json_file:
                json.dump(round_info, json_file, indent=4)

            contest_history.append(round_info)

            # Store images for visualization
            visualization_history.append({
                "round": round_num,
                "winner": winner_name,
                "score": winner_score,
                "image_a": state_a["image"].copy(),
                "image_b": state_b["image"].copy()
            })

            # Determine winner and loser
            if winner_name == base_a:
                winner_state = state_a
                loser_state = state_b
                loser_name = base_b
            else:
                winner_state = state_b
                loser_state = state_a
                loser_name = base_a

            # Save round results with winner marked
            winner_state["image"].save(os.path.join(results_dir, f"{pair_name}_{winner_name}_round-{round_num}.jpg"))
            winner_state["image"].save(os.path.join(results_dir, f"{pair_name}_{winner_name}_round-{round_num}_WINNER.jpg"))
            loser_state["image"].save(os.path.join(results_dir, f"{pair_name}_{loser_name}_round-{round_num}.jpg"))

            # Update champion tracking with actual contest winner (after saving images)
            winner_state["champion_bytes"] = winner_state["bytes"]
            winner_state["champion_prompt"] = winner_state["prompt"]
            winner_state["champion_image"] = winner_state["image"].copy()

            if self.use_last_winner_as_base:
                loser_state["bytes"] = loser_state["champion_bytes"]
                loser_state["prompt"] = loser_state["champion_prompt"]
                loser_state["image"] = loser_state["champion_image"].copy()

            # Check for equilibrium (score close to 0.5 = no clear winner)
            if round_num >= self.min_rounds_before_equilibrium:
                if winner_score < self.equilibrium_threshold:
                    logging.info(f"\n🎯 EQUILIBRIUM REACHED! (score: {winner_score:.2%} < {self.equilibrium_threshold:.2%})\n")
                    equilibrium_reached = True
                    break

            # Improve ONLY the loser
            logging.info(f"\n🔄 Improving {loser_name} (loser of round {round_num})\n")

            improved_bytes, improved_prompt, improved_image = self._improve_loser(
                loser_name=loser_name,
                loser_image_bytes=loser_state["bytes"],
                loser_prompt=loser_state["prompt"],
                feedback=feedback,
                round_num=round_num,
                results_dir=results_dir,
                pair_name=pair_name,
                history_of_prompts=loser_state["edit_history"],
                original_image_bytes=image_a_bytes_original if loser_name == base_a else image_b_bytes_original
            )

            # Update ONLY loser's state
            loser_state["bytes"] = improved_bytes
            loser_state["prompt"] = improved_prompt
            loser_state["image"] = improved_image
            loser_state["edit_history"].append(improved_prompt)

        # Generate final visualization
        self._visualize_competition(
            pair_name=pair_name,
            base_a=base_a,
            base_b=base_b,
            round_history=visualization_history,
            results_dir=results_dir,
            viz_path=viz_path
        )

        # Save final results
        logging.info(f"\n{'='*60}\n")
        if equilibrium_reached:
            logging.info(f"✅ Equilibrium reached after {round_num} rounds\n")
        else:
            logging.info(f"⚠️  Max rounds ({self.max_rounds_per_pair}) reached\n")
        logging.info(f"{'='*60}\n")

        # Save finals
        state_a["image"].save(os.path.join(results_dir, f"{pair_name}_{base_a}_final.jpg"))
        state_b["image"].save(os.path.join(results_dir, f"{pair_name}_{base_b}_final.jpg"))

        # Save summary
        summary_path = os.path.join(results_dir, f"{pair_name}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Paired Contest Summary: {base_a} vs {base_b}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total rounds: {round_num}\n")
            f.write(f"Equilibrium reached: {equilibrium_reached}\n\n")
            f.write(f"Contest History:\n")
            for entry in contest_history:
                f.write(f"  Round {entry['round']}: {entry['winner']} won ({entry['score']:.2%})\n")
            f.write(f"\nFinal State:\n")
            f.write(f"\n{base_a}:\n")
            f.write(f"  Prompt: {state_a['prompt']}\n")
            f.write(f"  Edit History: {len(state_a['edit_history'])} edits\n")
            for i, edit in enumerate(state_a["edit_history"], 1):
                f.write(f"    {i}. {edit}\n")
            f.write(f"\n{base_b}:\n")
            f.write(f"  Prompt: {state_b['prompt']}\n")
            f.write(f"  Edit History: {len(state_b['edit_history'])} edits\n")
            for i, edit in enumerate(state_b["edit_history"], 1):
                f.write(f"    {i}. {edit}\n")

        pair_duration = time.time() - pair_start
        logging.debug(f"\n⏱️  Pair {pair_idx+1} total: {pair_duration:.2f}s ({pair_duration/60:.2f}m)\n")

        return {
            "pair_name": pair_name,
            "image_a": base_a,
            "image_b": base_b,
            "rounds": round_num,
            "equilibrium": equilibrium_reached,
            "history": contest_history,
            "final_state_a": state_a,
            "final_state_b": state_b
        }

    def run(self, pairs: list[tuple[str, str]], results_dir: str, max_workers: int = 1):
        """
        Run paired contests for all combinations of images.
        Each pair competes until equilibrium is reached.
        """
        run_start = time.time()

        logging.debug(f"\n{'='*80}")
        logging.debug(f"🥊 Starting Paired Contest Competition")
        logging.debug(f"   Total pairs: {len(pairs)}")
        logging.debug(f"   Max rounds per pair: {self.max_rounds_per_pair}")
        logging.debug(f"   Equilibrium threshold: {self.equilibrium_threshold}")
        logging.debug(f"   Max workers: {max_workers}")
        logging.debug(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        # Process pairs
        results = []

        with tqdm(total=len(pairs), desc="Pairs completed", unit="pair") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_paired_contest,
                        pair[0], pair[1],
                        results_dir,
                        idx, len(pairs)
                    ): pair
                    for idx, pair in enumerate(pairs)
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Generate global summary
        summary_path = os.path.join(results_dir, "global_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Paired Contest Competition - Global Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total pairs: {len(pairs)}\n")
            f.write(f"Total images generated: {self._total_num_images_generated}\n")
            f.write(f"Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}\n\n")

            equilibrium_count = sum(1 for r in results if r["equilibrium"])
            f.write(f"Pairs reaching equilibrium: {equilibrium_count}/{len(pairs)}\n")
            f.write(f"Average rounds per pair: {sum(r['rounds'] for r in results)/len(results):.1f}\n\n")

            f.write(f"Individual Pair Results:\n")
            f.write(f"{'-'*80}\n")
            for r in results:
                f.write(f"\n{r['pair_name']}:\n")
                f.write(f"  Rounds: {r['rounds']}\n")
                f.write(f"  Equilibrium: {r['equilibrium']}\n")
                f.write(f"  History: ")
                for h in r["history"]:
                    f.write(f"R{h['round']}:{h['winner']}({h['score']:.0%}) ")
                f.write(f"\n")

        run_duration = time.time() - run_start
        logging.debug(f"\n{'='*80}")
        logging.debug(f"✅ Paired Contest Competition Complete!")
        logging.debug(f"⏱️  TOTAL RUNTIME: {run_duration:.2f}s ({run_duration/60:.2f}m)")
        logging.debug(f"   Total pairs: {len(pairs)}")
        logging.debug(f"   Pairs with equilibrium: {equilibrium_count}/{len(pairs)}")
        logging.debug(f"   Average rounds: {sum(r['rounds'] for r in results)/len(results):.1f}")
        logging.debug(f"   Total images generated: {self._total_num_images_generated}")
        logging.debug(f"   Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}")
        logging.debug(f"{'='*80}\n")

        return results


def evaluate_pair_comparability(
    image_id_1: str,
    img_bytes_1: bytes,
    image_id_2: str,
    img_bytes_2: bytes,
    judge_prompts: list[str],
    evaluator_model: LanguageModel,
    comparability_threshold: float
) -> tuple[bool, float, str]:
    """
    Evaluate whether a pair of images is "comparable" on the target parameter.
    A pair is considered comparable if the votes are evenly split.
    """
    def evaluate_single(judge_id: int, is_1_first: bool):
        """Single judge evaluation with position tracking."""
        images = [img_bytes_1, img_bytes_2] if is_1_first else [img_bytes_2, img_bytes_1]
        choice_map = {
            "first": image_id_1 if is_1_first else image_id_2,
            "second": image_id_2 if is_1_first else image_id_1,
        }

        evaluation = evaluator_model.get_response(
            images=images,
            judge_prompt=judge_prompts[judge_id],
            metadata="Context: the user is looking for a(n) %s." % image_id_1.split("_")[0].lower()
        )

        if evaluation is None:
            return None

        real_choice = choice_map.get(evaluation.choice.lower())
        return (real_choice, evaluation.reason)

    # Run all evaluations in parallel (each judge evaluates twice with swapped positions)
    judge_results = {}

    num_judges = len(judge_prompts)
    with ThreadPoolExecutor(max_workers=min(num_judges * 2, 8)) as executor:
        future_to_judge = {}
        for judge_id in range(num_judges):
            for is_1_first in [True, False]:
                future = executor.submit(evaluate_single, judge_id, is_1_first)
                future_to_judge[future] = (judge_id, is_1_first)

        for future in as_completed(future_to_judge):
            judge_id, is_1_first = future_to_judge[future]
            result = future.result()

            if judge_id not in judge_results:
                judge_results[judge_id] = {}
            judge_results[judge_id][is_1_first] = result

    # Aggregate consistent judges
    votes = {image_id_1: 0, image_id_2: 0}
    total_consistent_judges = 0

    for judge_id, results in judge_results.items():
        result_1_first = results.get(True)
        result_2_first = results.get(False)

        if result_1_first is None or result_2_first is None:
            continue

        choice_1_first, _ = result_1_first
        choice_2_first, _ = result_2_first

        # Only count consistent judges
        if choice_1_first == choice_2_first:
            total_consistent_judges += 1
            votes[choice_1_first] += 1

    # Determine comparability
    if total_consistent_judges == 0:
        # No consistent judges = treat as comparable (high uncertainty)
        return True, 0.5, image_id_1

    winner = max(votes, key=votes.get)
    winner_score = votes[winner] / total_consistent_judges

    # Pair is comparable if the decision is close to even
    is_comparable = winner_score <= comparability_threshold

    logging.info(
        f"Pair {image_id_1} vs {image_id_2}: "
        f"winner={winner}, score={winner_score:.2%}, "
        f"comparable={is_comparable}"
    )

    return is_comparable, winner_score, winner


def filter_comparable_pairs(
    image_paths: list[str],
    results_dir: str,
    judge_prompts: list[str],
    evaluator_model: LanguageModel,
    comparability_threshold: float,
    category_pattern: str,
    max_workers: int = 1
) -> list[tuple[str, str]]:
    """
    Phase 1: Filter all pairs to find comparable images.
    Returns list of (image_path_1, image_path_2) tuples for comparable pairs.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"PHASE 1: Filtering Comparable Pairs")
    logging.info(f"{'='*80}\n")

    # Group images by category
    image_categories = {}
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        match = re.match(category_pattern, base_name)
        if match:
            category = match.group(1)
        else:
            category = "default"
        image_categories.setdefault(category, []).append(image_path)

    # Generate all pairs within each category
    all_pairs = []
    for category, paths in image_categories.items():
        category_pairs = list(combinations(paths, 2))
        all_pairs.extend(category_pairs)

    logging.info(f"Total pairs to evaluate: {len(all_pairs)}")

    # Load image bytes
    image_bytes_cache = {}
    for path in image_paths:
        with open(path, "rb") as f:
            image_bytes_cache[path] = f.read()

    # Evaluate all pairs for comparability
    comparable_pairs = []
    comparability_results = []

    def evaluate_pair(pair: tuple[str, str]):
        path_1, path_2 = pair
        id_1 = os.path.splitext(os.path.basename(path_1))[0]
        id_2 = os.path.splitext(os.path.basename(path_2))[0]

        is_comparable, score, winner = evaluate_pair_comparability(
            id_1, image_bytes_cache[path_1],
            id_2, image_bytes_cache[path_2],
            judge_prompts,
            evaluator_model,
            comparability_threshold
        )

        return {
            "pair": pair,
            "id_1": id_1,
            "id_2": id_2,
            "is_comparable": is_comparable,
            "score": score,
            "winner": winner
        }

    with tqdm(total=len(all_pairs), desc="Evaluating pairs", unit="pair") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_pair, pair): pair for pair in all_pairs}

            for future in as_completed(futures):
                result = future.result()
                comparability_results.append(result)

                if result["is_comparable"]:
                    comparable_pairs.append(result["pair"])

                pbar.update(1)

    # Save comparability results
    comparability_csv = os.path.join(results_dir, "comparability_results.csv")
    with open(comparability_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id_1", "id_2", "is_comparable", "score", "winner"])
        writer.writeheader()
        for r in comparability_results:
            writer.writerow({
                "id_1": r["id_1"],
                "id_2": r["id_2"],
                "is_comparable": r["is_comparable"],
                "score": r["score"],
                "winner": r["winner"]
            })

    logging.info(f"\n{'='*60}")
    logging.info(f"Phase 1 Complete: Found {len(comparable_pairs)}/{len(all_pairs)} comparable pairs")
    logging.info(f"Comparability results saved to: {comparability_csv}")
    logging.info(f"{'='*60}\n")

    return comparable_pairs


@dataclasses.dataclass
class OptimizationPipeline:
    """
    Orchestrates parametric optimization of product images.
    Phase 1: Filter pairs to identify "comparable" images
    Phase 2: Run competition between comparable pairs
    """
    name: str
    base_prior: str
    image_editing_model: ImageModel
    judge_prompts: list[str]
    evaluator_model: LanguageModel
    equilibrium_threshold: float
    min_rounds_before_equilibrium: int
    max_rounds_per_pair: int
    tie_breaking_strategy: str
    use_last_winner_as_base: bool
    optimizer_model: LanguageModel
    num_improvement_proposals: int
    proposer_model: LanguageModel
    selector_model: LanguageModel
    use_candidates: bool
    comparability_threshold: float
    category_pattern: str

    def __post_init__(self):
        """Initialize the pipeline."""
        # Track comparable pairs
        self.comparable_pairs: list[tuple[str, str]] = []

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Run the full optimization pipeline.
        Phase 1: Filter pairs to find comparable images
        Phase 2: Run competition on comparable pairs
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"COMPETITION PIPELINE")
        logging.info(f"Number of Images: {len(image_paths)}")
        logging.info(f"{'='*80}\n")

        # Phase 1: Filter comparable pairs
        phase1_dir = os.path.join(results_dir, "phase1_comparability")
        os.makedirs(phase1_dir, exist_ok=True)

        comparable_pairs = filter_comparable_pairs(
            image_paths=image_paths,
            results_dir=phase1_dir,
            judge_prompts=self.judge_prompts,
            evaluator_model=self.evaluator_model,
            comparability_threshold=self.comparability_threshold,
            category_pattern=self.category_pattern,
            max_workers=max_workers
        )

        if not comparable_pairs:
            logging.warning("No comparable pairs found. Exiting competition.")
            return

        # Phase 2: Run competition on comparable pairs
        logging.info(f"\n{'='*80}")
        logging.info(f"PHASE 2: Running Competition on {len(comparable_pairs)} Comparable Pairs")
        logging.info(f"{'='*80}\n")

        phase2_dir = os.path.join(results_dir, "phase2_competition")
        os.makedirs(phase2_dir, exist_ok=True)

        # Run competition on the comparable pairs
        competition = VisualNudgeCompetition(
            name=self.name,
            base_prior=self.base_prior,
            image_editing_model=self.image_editing_model,
            judge_prompts=self.judge_prompts,
            evaluator_model=self.evaluator_model,
            equilibrium_threshold=self.equilibrium_threshold,
            min_rounds_before_equilibrium=self.min_rounds_before_equilibrium,
            max_rounds_per_pair=self.max_rounds_per_pair,
            tie_breaking_strategy=self.tie_breaking_strategy,
            use_last_winner_as_base=self.use_last_winner_as_base,
            optimizer_model=self.optimizer_model,
            num_improvement_proposals=self.num_improvement_proposals,
            proposer_model=self.proposer_model,
            selector_model=self.selector_model,
            use_candidates=self.use_candidates
        )

        competition_results = competition.run(
            pairs=comparable_pairs,
            results_dir=phase2_dir,
            max_workers=max_workers
        )

        logging.info(f"\nCompetition pipeline complete. Results saved to: {results_dir}\n")

        return {
            "comparable_pairs": comparable_pairs,
            "competition_results": competition_results
        }
