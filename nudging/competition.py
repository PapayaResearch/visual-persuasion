import os
import io
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
from utils.wrappers import ImageModel, LanguageModel


@dataclasses.dataclass
class VisualNudgeCompetition:
    """
    Orchestrates paired contests between two zero-shot edits of the same image.
    Instead of loading comparable pairs, generates two zero-shot versions using the same prompt.
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
    # For optimizer flow (used when num_candidates == 0)
    optimizer_model: LanguageModel
    # For proposer flow (used when num_candidates > 0)
    num_candidates: int
    proposer_model: LanguageModel

    def __post_init__(self):
        """Initialize tracking variables."""
        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039
        self._counter_lock = threading.Lock()

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
        logging.info(f"\n🥊 CONTEST: {image_a_name} vs {image_b_name}\n")

        def evaluate_single(judge_id: int, is_a_first: bool):
            """Single judge evaluation"""
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

            # Check for 50-50 tie and apply tie-breaking
            if winner_score == 0.5:
                logging.warning("Judges split 50-50. Applying tie-breaking strategy.\n")
                winner = break_tie(image_a_name, image_b_name)

            feedback = "\n".join(feedback_by_choice[winner])
            logging.info(f"🏆 WINNER: {winner} ({votes[winner]}/{total_consistent_judges} = {winner_score:.2%})\n")

        return winner, winner_score, feedback

    def _visualize_competition(
        self,
        base_a: str,
        base_b: str,
        round_history: list[dict],
        viz_path: str,
        results_dir: str
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

            # Image A column - load from saved file
            ax_a = axes[round_idx, 1]
            img_a_path = os.path.join(results_dir, f"{base_a}_round-{round_num}.jpg")
            img_a = Image.open(img_a_path)
            ax_a.imshow(img_a)
            ax_a.set_title(f"{base_a}", fontsize=12, fontweight="bold" if winner == base_a else "normal")
            ax_a.axis("off")
            img_a.close()  # Free memory immediately

            # Image B column - load from saved file
            ax_b = axes[round_idx, 2]
            img_b_path = os.path.join(results_dir, f"{base_b}_round-{round_num}.jpg")
            img_b = Image.open(img_b_path)
            ax_b.imshow(img_b)
            ax_b.set_title(f"{base_b}", fontsize=12, fontweight="bold" if winner == base_b else "normal")
            ax_b.axis("off")
            img_b.close()  # Free memory immediately

        plt.tight_layout()

        # Save visualization
        fig.savefig(viz_path, dpi=300, bbox_inches="tight")
        logging.info(f"📊 Visualization saved to {viz_path}\n")
        plt.close(fig)

    def _run_tournament(
        self,
        candidate_images: list[dict],
        rival_name: str,
        rival_image_bytes: bytes
    ) -> dict:
        """
        Run a tournament between all candidates and the rival.
        Each candidate competes against the rival, and the best performer wins.

        Args:
            candidate_images: List of candidate dicts with 'prompt', 'image', 'image_bytes'
            rival_name: Name of the rival image
            rival_image_bytes: Bytes of the rival image

        Returns: The winning candidate dict (or a dict representing the rival if it wins)
        """
        logging.info(f"\n🏆 TOURNAMENT: {len(candidate_images)} candidates vs {rival_name}\n")

        # Track scores for each candidate
        scores = {}

        # Run each candidate against the rival
        for i, candidate in enumerate(candidate_images):
            candidate_name = f"candidate_{i+1}"

            logging.info(f"\n--- Match {i+1}/{len(candidate_images)}: {candidate_name} vs {rival_name} ---\n")

            winner_name, winner_score, feedback = self._conduct_contest(
                image_a_bytes=candidate["image_bytes"],
                image_b_bytes=rival_image_bytes,
                image_a_name=candidate_name,
                image_b_name=rival_name
            )

            # Record scores: higher is better for the candidate
            if winner_name == candidate_name:
                scores[i] = winner_score
                logging.info(f"✅ Candidate {i+1} wins with score {winner_score:.2%}\n")
            else:
                # Rival won - give candidate the inverse score
                scores[i] = 1 - winner_score
                logging.info(f"❌ Candidate {i+1} loses (rival wins with score {winner_score:.2%})\n")

        # Find the best performing candidate (always pick one, even if all lost to rival)
        best_idx = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_idx]
        winner = candidate_images[best_idx]

        if best_score > 0.5:
            logging.info(f"\n🏆 TOURNAMENT WINNER: Candidate {best_idx+1} with score {best_score:.2%} (beat rival)\n")
        else:
            logging.info(f"\n🏆 TOURNAMENT WINNER: Candidate {best_idx+1} with score {best_score:.2%} (best among candidates, but lost to rival)\n")

        return winner

    def _improve_loser_with_candidates(
        self,
        loser_name: str,
        loser_image_bytes: bytes,
        loser_prompt: str,
        feedback: str,
        round_num: int,
        results_dir: str,
        pair_name: str,
        history_of_prompts: list[dict],
        original_image_bytes: bytes,
        winner_name: str = None,
        winner_image_bytes: bytes = None
    ) -> tuple[bytes, str, Image.Image, dict]:
        """
        Generate improved versions of the losing image based on judge feedback.
        Uses proposer flow after the initial zero-shot edit.
        """
        logging.info(f"\n🔧 IMPROVING LOSER (Round {round_num})\n")
        logging.info(f"Previous prompt: {loser_prompt}\n")
        logging.info(f"Judge feedback:\n{feedback}\n")

        has_prior_edits = len(history_of_prompts) > 0
        if has_prior_edits:
            history_text = ""
            for entry in history_of_prompts:
                if entry.get("won_next_round") is None:
                    continue
                round = entry["round"]
                prompt = entry["prompt"]
                won_next_round = "Won" if entry.get("won_next_round") else "Lost"
                history_text += f"  - Round {round} ({won_next_round}): {prompt}\n"
        else:
            history_text = "None"
        logging.info(f"Edit history:\n{history_text}\n")

        # For first round with no history, use empty history
        proposer_input = {
            "current_prompt": self._compose_prompt(loser_prompt),
            "history_of_prompts": history_text,
            "current_iteration": round_num,
            "judge_feedback": feedback or "No feedback provided.",
            "total_iterations": self.min_rounds_before_equilibrium,
            "num_candidates": self.num_candidates,
            "metadata": "The image here is of a(n) %s." % loser_name.split("_")[0].lower()
        }
        proposer_response = self.proposer_model.get_response(**proposer_input)

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

            candidate_path = os.path.join(results_dir, f"{loser_name}_round-{round_num}_candidate-{i+1}.jpg")
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
            metadata = "All candidate generations failed"
            return (
                loser_image_bytes,
                loser_prompt,
                Image.open(io.BytesIO(loser_image_bytes)),
                metadata
            )

        candidate_images.sort(key=lambda c: c["candidate_idx"])

        # Run tournament between all candidates and the winner (rival)
        tournament_winner = self._run_tournament(
            candidate_images=candidate_images,
            rival_name=winner_name,
            rival_image_bytes=winner_image_bytes
        )

        # Free memory: close losing candidate images
        for candidate in candidate_images:
            if candidate is not tournament_winner:
                candidate["image"].close()

        total_cost = self._total_num_images_generated * self._cost_per_image_generated
        logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")

        # Save the tournament winner (always a candidate now)
        optimized_path = os.path.join(
            results_dir,
            f"{loser_name}_round-{round_num}_optimized.jpg"
        )
        tournament_winner["image"].save(optimized_path)
        logging.info(f"Saved optimized candidate to {optimized_path}\n")

        metadata = {
            "strategy": "proposer_tournament",
            "proposer_input": proposer_input,
            "candidate_prompts": candidate_prompts,
            "tournament_winner": candidate_images.index(tournament_winner) if tournament_winner in candidate_images else "unknown"
        }

        return (
            tournament_winner["image_bytes"],
            tournament_winner["prompt"],
            tournament_winner["image"],
            metadata
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
        history_of_prompts: list[dict],
        original_image_bytes: bytes
    ) -> tuple[bytes, str, Image.Image, dict]:
        """
        Improve the losing image by asking the optimizer for a better prompt and regenerating once.
        """
        logging.info(f"\n🔧 IMPROVING LOSER (Round {round_num})\n")
        logging.info(f"Previous prompt: {loser_prompt}\n")
        logging.info(f"Judge feedback:\n{feedback}\n")

        # Format history for optimizer visibility
        has_prior_edits = len(history_of_prompts) > 0
        if has_prior_edits:
            history_text = ""
            for entry in history_of_prompts:
                if entry.get("won_next_round") is None:
                    continue
                round = entry["round"]
                prompt = entry["prompt"]
                won_next_round = "Won" if entry.get("won_next_round") else "Lost"
                history_text += f"  - Round {round} ({won_next_round}): {prompt}\n"
        else:
            history_text = "None"
        logging.info(f"Edit history:\n{history_text}\n")

        if not has_prior_edits:
            improved_prompt = None
            logging.info("Applying base prior before switching to optimizer-driven edits.\n")
        else:
            optimizer_input = {
                "current_prompt": self._compose_prompt(loser_prompt),
                "current_image": loser_image_bytes,
                "history_of_prompts": history_text,
                "current_iteration": round_num,
                "judge_feedback": feedback or "No feedback provided.",
                "metadata": "The image here is of a(n) %s." % loser_name.split("_")[0].lower()
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
            f"{loser_name}_round-{round_num}_optimized.jpg"
        )
        edited_image.save(improved_path)

        logging.info(f"Saved optimized candidate to {improved_path}\n")

        total_cost = self._total_num_images_generated * self._cost_per_image_generated
        logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")

        if has_prior_edits:
            optimizer_input_to_log = optimizer_input.copy()
            optimizer_input_to_log.pop("current_image")
            metadata = {
                "strategy": "optimizer",
                "optimizer_input": optimizer_input_to_log,
                "optimizer_output": improved_prompt
            }
        else:
            metadata = {"strategy": "zero_shot"}

        return (
            edited_image_bytes,
            improved_prompt,
            edited_image,
            metadata
        )

    def _improve_loser(self, **kwargs) -> tuple[bytes, str, Image.Image, dict]:
        """
        Chooses the improvement strategy based on configuration.
        If num_candidates > 0, uses proposer flow.
        If num_candidates == 0, uses optimizer flow.
        """
        if self.num_candidates > 0:
            return self._improve_loser_with_candidates(**kwargs)
        elif self.num_candidates == 0:
            return self._improve_loser_without_candidates(**kwargs)

    def _run_paired_contest(
        self,
        image_path: str,
        results_dir: str,
        pair_idx: int,
        total_pairs: int
    ) -> dict:
        """
        Run a paired contest between two zero-shot edits of the same image until equilibrium is reached.
        Only the loser is improved each round.
        Returns final state of both images.
        """
        base_img = os.path.splitext(os.path.basename(image_path))[0]
        # Create variants with proper format for evaluation parser
        # Parser expects: PREFIX_CATEGORY_ID_STATUS.jpg
        # So we make base_a/base_b = "CATEGORY_ID_X" to get "PREFIX_CATEGORY_ID_X_STATUS.jpg"
        category, image_id = base_img.split('_', 1)
        base_a = f"{category}_{image_id}_A"
        base_b = f"{category}_{image_id}_B"
        # Use empty prefix since base names already contain full ID
        pair_name = ""

        # Initialize structured log for this pair
        pair_logs = []

        logging.info(f"\n{'='*80}\n")
        logging.info(f"PAIRED CONTEST {pair_idx+1}/{total_pairs}: {base_a} vs {base_b}\n")
        logging.info(f"{'='*80}\n")

        # Load original image
        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

        original_image_obj = Image.open(io.BytesIO(original_image_bytes))

        # Generate first zero-shot edit
        prompt_a = self._compose_prompt(None)

        edited_image_a, edited_image_a_bytes = self.image_editing_model.edit(
            prompt_a,
            original_image_bytes,
            original_image_bytes
        )

        with self._counter_lock:
            self._total_num_images_generated += 1

        # Generate second zero-shot edit with instruction to be different
        prompt_b = self._compose_prompt(
            "IMPORTANT: Generate a very different image from the other image shown."
        )

        edited_image_b, edited_image_b_bytes = self.image_editing_model.edit(
            prompt_b,
            original_image_bytes,
            edited_image_a_bytes  # Pass image A as context to differentiate from
        )

        with self._counter_lock:
            self._total_num_images_generated += 1

        # Initialize state with edit history and champion tracking
        state_a = {
            "bytes": edited_image_a_bytes,
            "prompt": None,
            "image": edited_image_a,
            "name": base_a,
            "edit_history": [],  # Track all prompts used
            "champion_bytes": edited_image_a_bytes,
            "champion_prompt": None
        }
        state_b = {
            "bytes": edited_image_b_bytes,
            "prompt": None,
            "image": edited_image_b,
            "name": base_b,
            "edit_history": [],
            "champion_bytes": edited_image_b_bytes,
            "champion_prompt": None
        }

        # Save original (only one since both variants come from same source)
        original_image_obj.save(os.path.join(results_dir, f"{base_img}_original.jpg"))

        # Contest loop
        round_num = 0
        equilibrium_reached = False
        visualization_history = []  # For matplotlib visualization
        final_winner_name = None
        final_winner_state = None

        viz_path = os.path.join(results_dir, f"{base_img}_visualization.png")

        if os.path.isfile(viz_path):
            logging.info(f"The final visualization for {base_img} already exists. Skipping contest.\n")
            return {
                "pair_name": base_img,
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

            # Initialize round log entry
            round_log = {
                "round_number": round_num,
                "contest": {},
                "improvement": {}
            }

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

            # Add contest results to log
            round_log["contest"] = {
                "winner": winner_name,
                "loser": base_b if winner_name == base_a else base_a,
                "winner_score": winner_score,
                "feedback": feedback.split("\n"),
                "winner_prompt": (self._compose_prompt(state_a["prompt"]) if winner_name == base_a
                                    else self._compose_prompt(state_b["prompt"])),
                "loser_prompt": (self._compose_prompt(state_b["prompt"]) if winner_name == base_a
                                    else self._compose_prompt(state_a["prompt"]))
            }

            # Track metadata for visualization (images will be reloaded from disk)
            visualization_history.append({
                "round": round_num,
                "winner": winner_name,
                "score": winner_score
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
            winner_state["image"].save(os.path.join(results_dir, f"{winner_name}_round-{round_num}.jpg"))
            winner_state["image"].save(os.path.join(results_dir, f"{winner_name}_round-{round_num}_WINNER.jpg"))
            loser_state["image"].save(os.path.join(results_dir, f"{loser_name}_round-{round_num}.jpg"))

            # Update champion tracking with actual contest winner (after saving images)
            winner_state["champion_bytes"] = winner_state["bytes"]
            winner_state["champion_prompt"] = winner_state["prompt"]
            # Don't copy image - will reconstruct from bytes if needed

            # Mark winner's most recent edit as winning
            if len(winner_state["edit_history"]) > 0:
                winner_state["edit_history"][-1]["won_next_round"] = True

            # Track final winner
            final_winner_name = winner_name
            final_winner_state = winner_state

            # Save zero-shot winner after first round
            if round_num == 1:
                winner_state["image"].save(os.path.join(results_dir, f"{base_img}_zero-shot.jpg"))
                logging.info(f"💾 Saved zero-shot winner: {winner_name}\n")

            if self.use_last_winner_as_base:
                loser_state["bytes"] = loser_state["champion_bytes"]
                loser_state["prompt"] = loser_state["champion_prompt"]
                loser_state["image"] = Image.open(io.BytesIO(loser_state["champion_bytes"]))

            # Check for equilibrium (score close to 0.5 = no clear winner)
            if round_num >= self.min_rounds_before_equilibrium:
                if winner_score < self.equilibrium_threshold:
                    logging.info(f"\n🎯 EQUILIBRIUM REACHED! (score: {winner_score:.2%} < {self.equilibrium_threshold:.2%})\n")
                    equilibrium_reached = True
                    pair_logs.append(round_log)
                    break

            # Improve ONLY the loser
            logging.info(f"\n🔄 Improving {loser_name} (loser of round {round_num})\n")

            # Update previous round's history entry BEFORE calling improve_loser (so history is correct)
            if len(loser_state["edit_history"]) > 0:
                if loser_state["edit_history"][-1]["won_next_round"] is None:
                    loser_state["edit_history"][-1]["won_next_round"] = False

            improved_bytes, improved_prompt, improved_image, improvement_metadata = self._improve_loser(
                loser_name=loser_name,
                loser_image_bytes=loser_state["bytes"],
                loser_prompt=loser_state["prompt"],
                feedback=feedback,
                round_num=round_num,
                results_dir=results_dir,
                pair_name=pair_name,
                history_of_prompts=loser_state["edit_history"],
                original_image_bytes=original_image_bytes,
                winner_name=winner_name,
                winner_image_bytes=winner_state["bytes"]
            )

            # Log improvement to structured log
            round_log["improvement"] = {
                "improved_image": loser_name,
                "previous_prompt": self._compose_prompt(loser_state["prompt"]),
                "improved_prompt": self._compose_prompt(improved_prompt)
            }
            round_log["improvement"].update(improvement_metadata)
            pair_logs.append(round_log)

            # Update ONLY loser's state
            loser_state["bytes"] = improved_bytes
            loser_state["prompt"] = improved_prompt
            loser_state["image"] = improved_image
            history_entry = {
                "round": round_num,
                "prompt": improved_prompt,
                "won_next_round": None  # Will be updated after next contest
            }
            loser_state["edit_history"].append(history_entry)

        # Generate final visualization
        self._visualize_competition(
            base_a=base_a,
            base_b=base_b,
            round_history=visualization_history,
            viz_path=viz_path,
            results_dir=results_dir
        )

        # Save final results
        logging.info(f"\n{'='*60}\n")
        if equilibrium_reached:
            logging.info(f"✅ Equilibrium reached after {round_num} rounds\n")
        else:
            logging.info(f"⚠️  Max rounds ({self.max_rounds_per_pair}) reached\n")
        logging.info(f"{'='*60}\n")

        # Save only the final winner
        logging.info(f"Final winner: {final_winner_name}\n")
        final_winner_state["image"].save(os.path.join(results_dir, f"{base_img}_final.jpg"))

        # Save structured log to JSON
        log_path = os.path.join(results_dir, f"{base_img}_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(pair_logs, f, indent=2, ensure_ascii=False)
        logging.info(f"📝 Detailed log for pair saved to {log_path}\n")

        # Save summary
        summary_path = os.path.join(results_dir, f"{base_img}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Paired Contest Summary: {base_a} vs {base_b}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total rounds: {round_num}\n")
            f.write(f"Equilibrium reached: {equilibrium_reached}\n\n")
            f.write(f"Contest History:\n")
            for round_log in pair_logs:
                if "contest" in round_log and "winner" in round_log["contest"]:
                    f.write(f"  Round {round_log['round_number']}: {round_log['contest']['winner']} won ({round_log['contest']['winner_score']:.2%})\n")
            f.write(f"\nFinal State:\n")
            f.write(f"\n{base_a}:\n")
            f.write(f"  Prompt: {state_a['prompt']}\n")
            f.write(f"  Edit History: {len(state_a['edit_history'])} edits\n")
            for i, edit in enumerate(state_a["edit_history"], 1):
                round = edit["round"]
                prompt = edit["prompt"]
                f.write(f"    {i}. Round {round}: {prompt}\n")
            f.write(f"\n{base_b}:\n")
            f.write(f"  Prompt: {state_b['prompt']}\n")
            f.write(f"  Edit History: {len(state_b['edit_history'])} edits\n")
            for i, edit in enumerate(state_b["edit_history"], 1):
                round = edit["round"]
                prompt = edit["prompt"]
                f.write(f"    {i}. Round {round}: {prompt}\n")

        return {
            "pair_name": base_img,
            "image_a": base_a,
            "image_b": base_b,
            "rounds": round_num,
            "equilibrium": equilibrium_reached,
            "final_state_a": state_a,
            "final_state_b": state_b
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Run paired contests between two zero-shot edits of each image.
        Each pair competes until equilibrium is reached.
        """
        run_start = time.time()

        logging.info(f"\n{'='*80}")
        logging.info(f"🥊 Starting Paired Contest Competition (No Bias)")
        logging.info(f"   Total images: {len(image_paths)}")
        logging.info(f"   Base prior: {self.base_prior}")
        logging.info(f"   Max rounds per pair: {self.max_rounds_per_pair}")
        logging.info(f"   Equilibrium threshold: {self.equilibrium_threshold}")
        logging.info(f"   Max workers: {max_workers}")
        logging.info(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        # Process images
        results = []

        with tqdm(total=len(image_paths), desc="Images completed", unit="image") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_paired_contest,
                        image_path,
                        results_dir,
                        idx, len(image_paths)
                    ): image_path
                    for idx, image_path in enumerate(image_paths)
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Generate global summary
        summary_path = os.path.join(results_dir, "global_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Paired Contest Competition (No Bias) - Global Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total images: {len(image_paths)}\n")
            f.write(f"Total images generated: {self._total_num_images_generated}\n")
            f.write(f"Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}\n\n")

            equilibrium_count = sum(1 for r in results if r["equilibrium"])
            f.write(f"Pairs reaching equilibrium: {equilibrium_count}/{len(image_paths)}\n")
            f.write(f"Average rounds per pair: {sum(r['rounds'] for r in results)/len(results):.1f}\n\n")

        run_duration = time.time() - run_start
        logging.info(f"\n{'='*80}")
        logging.info(f"✅ Paired Contest Competition (No Bias) Complete!")
        logging.info(f"⏱️  TOTAL RUNTIME: {run_duration:.2f}s ({run_duration/60:.2f}m)")
        logging.info(f"   Total images: {len(image_paths)}")
        logging.info(f"   Pairs with equilibrium: {equilibrium_count}/{len(image_paths)}")
        logging.info(f"   Average rounds: {sum(r['rounds'] for r in results)/len(results):.1f}")
        logging.info(f"   Total images generated: {self._total_num_images_generated}")
        logging.info(f"   Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}")
        logging.info(f"{'='*80}\n")

        return results
