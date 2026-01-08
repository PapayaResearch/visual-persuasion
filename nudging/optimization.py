import os
import io
import time
import threading
import logging
import dataclasses
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from string import Template
from itertools import combinations
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.wrappers import LanguageModel, ImageModel


@dataclasses.dataclass
class ScoreCompetition:
    """
    Orchestrates paired contests between images, iteratively improving losers until equilibrium.
    Each pair of images competes, and the loser is improved based on judge feedback.
    Process continues until neither image can be improved further (equilibrium).
    """
    name: str
    editing_context_prompt: str
    initial_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    judge_prompt: str
    evaluator_model: LanguageModel

    # Optimizer for improving losers
    num_improvement_proposals: int
    proposer_model: LanguageModel
    selector_model: LanguageModel

    # Equilibrium detection
    equilibrium_threshold: float = 0.51  # Win rate below this (close to 0.5) indicates equilibrium
    min_rounds_before_equilibrium: int = 10  # Minimum rounds before checking for equilibrium
    max_rounds_per_pair: int = 10  # Maximum improvement rounds per pair before declaring equilibrium

    def __post_init__(self):
        """Initialize tracking variables."""
        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039
        self._counter_lock = threading.Lock()

        self.contest_history = {}

    def _conduct_contest(
        self,
        image_a_bytes: bytes,
        image_b_bytes: bytes,
        image_a_name: str,
        image_b_name: str,
        prev_score_a: int = None,
        prev_score_b: int = None,
        prev_winner: str = None
    ) -> tuple[str, float, str, int, int]:
        """
        Conduct a score-based contest between two images.
        Returns: (winner_name, winner_score_ratio, feedback, score_a, score_b)
        winner_score_ratio is winner_score / (winner_score + loser_score)
        score_a and score_b are the raw scores (0-100) for each image
        """
        contest_start = time.time()
        logging.info(f"\n🥊 CONTEST: {image_a_name} vs {image_b_name}\n")

        def evaluate_single(image_bytes: bytes, image_name: str):
            """Evaluate a single image and return its score"""
            eval_start = time.time()
            logging.info(f"Evaluating {image_name}...\n")

            evaluation = self.evaluator_model.get_response(
                image=image_bytes,
                judge_prompt=self.judge_prompt
            )

            eval_duration = time.time() - eval_start
            logging.debug(f"  ⏱️  {image_name} evaluation: {eval_duration:.2f}s")
            logging.info(f"📊 {image_name}: score={evaluation.score} - {evaluation.reason}\n")
            return evaluation.score, evaluation.reason

        # Check if we can reuse previous scores (only re-evaluate the loser's improved image)
        if prev_winner is not None and prev_score_a is not None and prev_score_b is not None:
            # Only evaluate the image that was improved (the previous loser)
            if prev_winner == image_a_name:
                # A was winner, B was loser and improved - only evaluate B
                score_a = prev_score_a
                reason_a = "(previous score reused)"
                logging.info(f"📊 {image_a_name}: score={score_a} (reused from previous round)\n")
                score_b, reason_b = evaluate_single(image_b_bytes, image_b_name)
            else:
                # B was winner, A was loser and improved - only evaluate A
                score_b = prev_score_b
                reason_b = "(previous score reused)"
                logging.info(f"📊 {image_b_name}: score={score_b} (reused from previous round)\n")
                score_a, reason_a = evaluate_single(image_a_bytes, image_a_name)
        else:
            # First round - evaluate both images in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(evaluate_single, image_a_bytes, image_a_name)
                future_b = executor.submit(evaluate_single, image_b_bytes, image_b_name)

                score_a, reason_a = future_a.result()
                score_b, reason_b = future_b.result()

        # Determine winner based on scores (higher score wins)
        if score_a >= score_b:  # Tie goes to first image
            winner = image_a_name
            winner_score = score_a
            loser_score = score_b
            feedback = reason_a
        else:
            winner = image_b_name
            winner_score = score_b
            loser_score = score_a
            feedback = reason_b

        # Calculate ratio as winner / (winner + loser) for equilibrium detection
        total_score = winner_score + loser_score
        winner_score_ratio = winner_score / total_score

        logging.info(f"📊 SCORES: {image_a_name}={score_a}, {image_b_name}={score_b}\n")
        logging.info(f"🏆 WINNER: {winner} (ratio: {winner_score_ratio:.2%})\n")

        contest_duration = time.time() - contest_start
        logging.debug(f"⏱️  Contest completed: {contest_duration:.2f}s")

        return winner, winner_score_ratio, feedback, score_a, score_b

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

    def _improve_loser(
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
        Uses selector model to choose best proposal (not tested against winner).
        Returns: (best_improved_bytes, best_improved_prompt, best_improved_image)
        """
        improve_start = time.time()
        logging.info(f"\n🔧 IMPROVING LOSER (Round {round_num})\n")
        logging.info(f"Previous prompt: {loser_prompt}\n")
        logging.info(f"Judge feedback:\n{feedback}\n")

        # Format history for proposer
        history_text = "\n".join([f"  - {p}" for p in history_of_prompts]) if history_of_prompts else "None"
        logging.info(f"Edit history:\n{history_text}\n")

        # Generate improvement proposals
        proposer_response = self.proposer_model.get_response(
            current_prompt=loser_prompt,
            history_of_prompts=history_text,
            current_iteration=round_num,
            # judge_feedback=feedback,
            total_iterations=self.min_rounds_before_equilibrium,
            num_proposals=self.num_improvement_proposals,
            metadata="The product here is a(n) %s." % pair_name.split("_")[1]
        )
        logging.debug(f"⏱️  Proposal category is: {pair_name.split('_')[1]}")

        candidate_prompts = proposer_response.candidate_prompts
        logging.info(f"Generated {len(candidate_prompts)} improvement candidates:\n")
        for i, prompt in enumerate(candidate_prompts):
            logging.info(f"  Candidate {i+1}: {prompt}\n")

        # Generate images for all candidates in parallel
        def generate_single_image(i: int, prompt: str):
            """Generate a single improved image"""
            logging.info(f"Generating improved image {i+1}/{len(candidate_prompts)}\n")
            editing_prompt = f"{prompt}\n{self.editing_context_prompt}"
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                f"{editing_prompt}\n{self.background_state_prompt}",
                loser_image_bytes,
                original_image_bytes
            )

            with self._counter_lock:
                self._total_num_images_generated += 1

            # Save candidate
            candidate_path = os.path.join(results_dir, f"{pair_name}_{loser_name}_round-{round_num}_candidate-{i+1}.jpg")
            edited_image.save(candidate_path)

            return {
                "prompt": prompt,
                "image": edited_image,
                "image_bytes": edited_image_bytes,
                "save_path": candidate_path
            }

        candidate_images = []
        with ThreadPoolExecutor(max_workers=min(len(candidate_prompts), 4)) as img_executor:
            futures = [img_executor.submit(generate_single_image, i, prompt) for i, prompt in enumerate(candidate_prompts)]
            for future in as_completed(futures):
                candidate_images.append(future.result())

        # Select best candidate using selector model (NOT by testing against winner)
        best_candidate = self._select_best_proposal(candidate_images, feedback=feedback)

        # Log cost
        total_cost = self._total_num_images_generated * self._cost_per_image_generated
        logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")
        logging.debug(f"Total images: {self._total_num_images_generated}, Cost: ${total_cost:.2f}")

        improve_duration = time.time() - improve_start
        logging.debug(f"⏱️  Improvement phase: {improve_duration:.2f}s")

        return (
            best_candidate["image_bytes"],
            best_candidate["prompt"],
            best_candidate["image"]
        )

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
            score_a = round_data.get("score_a", 0)
            score_b = round_data.get("score_b", 0)

            # Round label column
            ax_label = axes[round_idx, 0]
            ax_label.text(0.5, 0.5, f"Round {round_num}\n{winner} wins\n{base_a}: {score_a}\n{base_b}: {score_b}",
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
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"📊 Visualization saved to {viz_path}\n")

    def _plot_score_history(
        self,
        pair_name: str,
        base_a: str,
        base_b: str,
        score_history: list[dict],
        results_dir: str,
        plot_path: str
    ):
        """
        Create a line plot showing score progression for both images over rounds.
        """
        num_rounds = len(score_history)
        if num_rounds == 0:
            return

        rounds = [entry["round"] for entry in score_history]
        scores_a = [entry["score_a"] for entry in score_history]
        scores_b = [entry["score_b"] for entry in score_history]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot both score lines
        ax.plot(rounds, scores_a, 'b-o', label=base_a, linewidth=2, markersize=8)
        ax.plot(rounds, scores_b, 'r-s', label=base_b, linewidth=2, markersize=8)

        # Add labels and title
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Score (0-100)", fontsize=12)
        ax.set_title(f"Score Progression: {base_a} vs {base_b}", fontsize=14, fontweight="bold")

        # Set axis limits
        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, num_rounds + 0.5)
        ax.set_xticks(rounds)

        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize=10)

        # Add score annotations
        for r, sa, sb in zip(rounds, scores_a, scores_b):
            ax.annotate(f'{sa}', (r, sa), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue')
            ax.annotate(f'{sb}', (r, sb), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='red')

        plt.tight_layout()

        # Save plot
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"📈 Score plot saved to {plot_path}\n")

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

        # Initialize state with edit history
        state_a = {
            "bytes": image_a_bytes,
            "prompt": self.initial_prompt,
            "image": Image.open(io.BytesIO(image_a_bytes)),
            "name": base_a,
            "edit_history": []  # Track all prompts used
        }
        state_b = {
            "bytes": image_b_bytes,
            "prompt": self.initial_prompt,
            "image": Image.open(io.BytesIO(image_b_bytes)),
            "name": base_b,
            "edit_history": []
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
        score_history = []  # Track scores for plotting
        visualization_history = []  # For matplotlib visualization

        viz_path = os.path.join(results_dir, f"{pair_name}_visualization.png")
        score_plot_path = os.path.join(results_dir, f"{pair_name}_scores.png")

        if os.path.isfile(viz_path):
            logging.info(f"The final visualization for {pair_name} already exists. Skipping contest.\n")
            return {
                "pair_name": pair_name,
                "image_a": base_a,
                "image_b": base_b,
                "rounds": 0,
                "equilibrium": False,
                "history": [],
                "score_history": [],
                "final_state_a": state_a,
                "final_state_b": state_b
            }

        # Track previous round scores to avoid re-evaluating unchanged winner
        prev_score_a = None
        prev_score_b = None
        prev_winner = None

        while round_num < self.max_rounds_per_pair and not equilibrium_reached:
            round_num += 1
            logging.info(f"\n{'='*60}\n")
            logging.info(f"ROUND {round_num}/{self.max_rounds_per_pair}\n")
            logging.info(f"{'='*60}\n")

            # Conduct contest (reuse previous winner's score if available)
            winner_name, winner_score, feedback, score_a, score_b = self._conduct_contest(
                image_a_bytes=state_a["bytes"],
                image_b_bytes=state_b["bytes"],
                image_a_name=base_a,
                image_b_name=base_b,
                prev_score_a=prev_score_a,
                prev_score_b=prev_score_b,
                prev_winner=prev_winner
            )

            # Store scores for next round
            prev_score_a = score_a
            prev_score_b = score_b
            prev_winner = winner_name

            # Record history
            contest_history.append({
                "round": round_num,
                "winner": winner_name,
                "score": winner_score,
                "score_a": score_a,
                "score_b": score_b
            })

            # Track scores for plotting
            score_history.append({
                "round": round_num,
                "score_a": score_a,
                "score_b": score_b
            })

            # Store images for visualization
            visualization_history.append({
                "round": round_num,
                "winner": winner_name,
                "score": winner_score,
                "score_a": score_a,
                "score_b": score_b,
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

        # Generate score progression plot
        self._plot_score_history(
            pair_name=pair_name,
            base_a=base_a,
            base_b=base_b,
            score_history=score_history,
            results_dir=results_dir,
            plot_path=score_plot_path
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
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Paired Contest Summary: {base_a} vs {base_b}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total rounds: {round_num}\n")
            f.write(f"Equilibrium reached: {equilibrium_reached}\n\n")
            f.write(f"Contest History:\n")
            for entry in contest_history:
                f.write(f"  Round {entry['round']}: {base_a}={entry['score_a']}, {base_b}={entry['score_b']} -> {entry['winner']} won\n")
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
            "score_history": score_history,
            "final_state_a": state_a,
            "final_state_b": state_b
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Run paired contests for all combinations of images.
        Each pair competes until equilibrium is reached.
        """
        run_start = time.time()

        # Generate all pairs
        image_categories = {}
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            category = base_name.split("_")[0]  # Assuming category is prefix before the first underscore
            image_categories.setdefault(category, [])
            image_categories[category].append(image_path)

        pairs = []
        for category, paths in image_categories.items():
            category_pairs = list(combinations(paths, 2))
            pairs.extend(category_pairs)

        logging.debug(f"\n{'='*80}")
        logging.debug(f"🥊 Starting Paired Contest Competition")
        logging.debug(f"   Images: {len(image_paths)}")
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
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Paired Contest Competition - Global Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total images: {len(image_paths)}\n")
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


@dataclasses.dataclass
class OptimizationPipeline:
    """
    Orchestrates parametric optimization of product images.
    Phase 1: Filter pairs to identify "comparable" images on the target parameter
    Phase 2: Run competition between comparable pairs using ScoreCompetition
    """
    name: str
    parameter: str
    comparability_prompts_template: list[str]
    comparability_threshold: float
    comparability_evaluator_model: LanguageModel

    # Competition settings (will be passed to ScoreCompetition)
    competition_prompt_template: str
    max_rounds_per_pair: int
    editing_context_prompt: str
    initial_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    evaluator_model: LanguageModel

    # Optimizer models for competition
    num_improvement_proposals: int
    proposer_model: LanguageModel
    selector_model: LanguageModel

    # Competition equilibrium settings
    equilibrium_threshold: float = 0.51
    min_rounds_before_equilibrium: int = 5

    # Regex pattern to extract category from filename
    category_pattern: str = r'^([^_]+)_'

    def __post_init__(self):
        """Initialize the pipeline and generate judge prompts for the target parameter."""
        # Generate judge prompts dynamically based on the parameter
        self.judge_prompts = self._generate_judge_prompts()
        # Track comparable pairs
        self.comparable_pairs: List[Tuple[str, str]] = []

        self.name = os.path.join("parametric-optimization", self.parameter.lower())

    def _generate_judge_prompts(self) -> List[str]:
        """Generate judge prompts based on the target parameter."""
        prompts = [
            Template(prompt).substitute(parameter=self.parameter) for prompt in self.comparability_prompts_template
        ]
        return prompts

    def _evaluate_pair_comparability(
        self,
        image_id_1: str,
        img_bytes_1: bytes,
        image_id_2: str,
        img_bytes_2: bytes
    ) -> Tuple[bool, float, str]:
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

            evaluation = self.comparability_evaluator_model.get_response(
                images=images,
                judge_prompt=self.judge_prompts[judge_id]
            )

            if evaluation is None:
                return None

            real_choice = choice_map.get(evaluation.choice.lower())
            return (real_choice, evaluation.reason)

        # Run all evaluations in parallel (each judge evaluates twice with swapped positions)
        judge_results = {}

        num_judges = len(self.judge_prompts)
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
        is_comparable = winner_score <= self.comparability_threshold

        logging.info(
            f"Pair {image_id_1} vs {image_id_2}: "
            f"winner={winner}, score={winner_score:.2%}, "
            f"comparable={is_comparable}"
        )

        return is_comparable, winner_score, winner

    def _filter_comparable_pairs(
        self,
        image_paths: List[str],
        results_dir: str,
        max_workers: int = 1
    ) -> List[Tuple[str, str]]:
        """
        Phase 1: Filter all pairs to find comparable images on the target parameter.
        Returns list of (image_path_1, image_path_2) tuples for comparable pairs.
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"PHASE 1: Filtering Comparable Pairs for Parameter '{self.parameter}'")
        logging.info(f"{'='*80}\n")

        # Group images by category
        import re
        image_categories = {}
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            match = re.match(self.category_pattern, base_name)
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

        def evaluate_pair(pair: Tuple[str, str]):
            path_1, path_2 = pair
            id_1 = os.path.splitext(os.path.basename(path_1))[0]
            id_2 = os.path.splitext(os.path.basename(path_2))[0]

            is_comparable, score, winner = self._evaluate_pair_comparability(
                id_1, image_bytes_cache[path_1],
                id_2, image_bytes_cache[path_2]
            )
            # is_comparable, score, winner = True, 0.5, id_1  # TEMP OVERRIDE FOR TESTING

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
        import csv
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

    def _create_competition(self) -> ScoreCompetition:
        """
        Create a ScoreCompetition instance with parameter-specific judge prompts.
        """
        # Create evaluator model for competition
        competition_judge_prompt = Template(self.competition_prompt_template).substitute(parameter=self.parameter)

        return ScoreCompetition(
            name=f"optimization_{self.parameter.lower()}",
            editing_context_prompt=self.editing_context_prompt,
            initial_prompt=self.initial_prompt,
            background_state_prompt=self.background_state_prompt,
            image_editing_model=self.image_editing_model,
            judge_prompt=competition_judge_prompt,
            evaluator_model=self.evaluator_model,
            num_improvement_proposals=self.num_improvement_proposals,
            proposer_model=self.proposer_model,
            selector_model=self.selector_model,
            equilibrium_threshold=self.equilibrium_threshold,
            min_rounds_before_equilibrium=self.min_rounds_before_equilibrium,
            max_rounds_per_pair=self.max_rounds_per_pair
        )

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Run the full parametric optimization pipeline.
        Phase 1: Filter pairs to find comparable images
        Phase 2: Run competition on comparable pairs
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"PARAMETRIC OPTIMIZATION PIPELINE")
        logging.info(f"Target Parameter: {self.parameter}")
        logging.info(f"Number of Images: {len(image_paths)}")
        logging.info(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        # Phase 1: Filter comparable pairs
        phase1_dir = os.path.join(results_dir, "phase1_comparability")
        os.makedirs(phase1_dir, exist_ok=True)

        comparable_pairs = self._filter_comparable_pairs(
            image_paths, phase1_dir, max_workers
        )

        if not comparable_pairs:
            logging.warning("No comparable pairs found. Exiting optimization.")
            return {"comparable_pairs": [], "competition_results": []}

        # Phase 2: Run competition on comparable pairs
        logging.info(f"\n{'='*80}")
        logging.info(f"PHASE 2: Running Competition on {len(comparable_pairs)} Comparable Pairs")
        logging.info(f"{'='*80}\n")

        phase2_dir = os.path.join(results_dir, "phase2_competition")
        os.makedirs(phase2_dir, exist_ok=True)

        # Create competition instance with parameter-specific prompts
        competition = self._create_competition()

        # Extract unique image paths from comparable pairs
        comparable_image_paths = list(set(
            path for pair in comparable_pairs for path in pair
        ))

        # Run competition
        competition_results = competition.run(
            comparable_image_paths, phase2_dir, max_workers
        )

        # Save final summary
        summary_path = os.path.join(results_dir, "optimization_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Parametric Optimization Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Target Parameter: {self.parameter}\n")
            f.write(f"Total images: {len(image_paths)}\n")
            f.write(f"Comparable pairs found: {len(comparable_pairs)}\n")
            f.write(f"Competition results: {len(competition_results)} pairs competed\n\n")
            
            f.write(f"Comparable Pairs:\n")
            for path_1, path_2 in comparable_pairs:
                id_1 = os.path.splitext(os.path.basename(path_1))[0]
                id_2 = os.path.splitext(os.path.basename(path_2))[0]
                f.write(f"  - {id_1} vs {id_2}\n")

        logging.info(f"\nOptimization pipeline complete. Summary saved to: {summary_path}")

        return {
            "comparable_pairs": comparable_pairs,
            "competition_results": competition_results
        }
