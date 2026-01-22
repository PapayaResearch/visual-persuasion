import os
import io
import time
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
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.wrappers import ImageModel, LanguageModel
import textwrap


@dataclasses.dataclass
class FeedbackDescentBaseline:
    """
    Feedback Descent-based single-image optimization baseline.
    Maintains a single best prompt and proposes improvements conditioned
    on accumulated textual feedback from pairwise comparisons.
    """
    name: str
    max_iterations: int
    convergence_patience: int
    image_editing_model: ImageModel
    base_prior: str
    judge_prompt: str
    # Evaluator for pairwise comparisons
    evaluator_model: LanguageModel
    # Proposer model for generating improved prompts
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

    def _propose_improvement(
        self,
        current_best_prompt: str,
        feedback_history: list[tuple[str, str]],
        category: str
    ) -> tuple[str, str]:
        """
        Propose an improved instruction conditioned on current best and feedback history.
        """
        if not feedback_history:
            feedback_text = "No previous feedback yet."
        else:
            feedback_text = "Previous attempts and feedback:\n\n"
            for i, (candidate, rationale) in enumerate(feedback_history, 1):
                feedback_text += f"{i}. CANDIDATE: {self._compose_prompt(candidate)}\n"
                feedback_text += f"   FEEDBACK: {rationale}\n\n"

        # Prepare proposer input
        proposer_input = {
            'current_prompt': self._compose_prompt(current_best_prompt),
            'feedback_history': feedback_text,
            "metadata": f"The image is of a(n) {category}."
        }

        logging.info(f"\n📝 Proposing improvement via M(x*, R)...\n")

        # Get proposed improvement from proposer model
        response = self.proposer_model.get_response(**proposer_input)
        proposed_prompt = response.new_prompt.strip()

        logging.info(f"Proposed prompt: {proposed_prompt}\n")

        return proposed_prompt, feedback_text

    def _compare_images(
        self,
        candidate_bytes: bytes,
        current_best_bytes: bytes,
        category: str,
    ) -> tuple[str, str]:
        """
        Compare candidate against current best using evaluator.

        To mitigate order bias, performs two judgments with swapped image orders (A-B and B-A)
        and declares a winner only if both judgments are consistent. Otherwise, retries up to
        three times total. If no consistent winner emerges, defaults to current_best.
        """
        max_attempts = 3

        def evaluate_single(is_best_first: bool):
            """Single evaluation with specified order"""
            images = [current_best_bytes, candidate_bytes] if is_best_first else [candidate_bytes, current_best_bytes]
            choice_map = {
                "first": 'current_best' if is_best_first else 'candidate',
                "second": 'candidate' if is_best_first else 'current_best',
            }

            evaluation = self.evaluator_model.get_response(
                images=images,
                judge_prompt=self.judge_prompt,
                metadata="The images are of a(n) %s." % category
            )

            winner = choice_map.get(evaluation.choice.lower())

            return (winner, evaluation.reason)

        for attempt in range(1, max_attempts + 1):
            logging.info(f"Comparison attempt {attempt}/{max_attempts}...\n")

            # Run both evaluations in parallel
            results = {}  # is_best_first -> (winner, reason)

            with ThreadPoolExecutor(max_workers=2) as eval_executor:
                future_to_order = {}
                for is_best_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, is_best_first)
                    future_to_order[future] = is_best_first

                for future in as_completed(future_to_order):
                    is_best_first = future_to_order[future]
                    result = future.result()
                    results[is_best_first] = result

            # Extract results
            winner_ab, reason_ab = results[True]   # A-B order (current_best, candidate)
            winner_ba, reason_ba = results[False]  # B-A order (candidate, current_best)

            # Check consistency
            if winner_ab == winner_ba and winner_ab is not None:
                # Consistent winner found
                winner = winner_ab
                # Use feedback from the A-B comparison
                feedback = reason_ab

                logging.info(f"✓ Consistent winner after attempt {attempt}: {winner}\n")
                logging.info(f"Feedback: {feedback}\n")

                return winner, feedback
            else:
                logging.info(f"✗ Inconsistent results in attempt {attempt}: A-B={winner_ab}, B-A={winner_ba}\n")
                logging.info(f"  A-B feedback: {reason_ab}\n")
                logging.info(f"  B-A feedback: {reason_ba}\n")

        # No consistent winner after max attempts - default to current_best
        logging.warning(f"⚠️  No consistent winner after {max_attempts} attempts. Defaulting to current_best.\n")
        winner = 'current_best'
        feedback = "Could not determine a consistent winner after multiple evaluations."

        return winner, feedback

    def _visualize_optimization(
        self,
        image_name: str,
        iteration_history: list[dict],
        viz_path: str
    ):
        """
        Create a visualization of the Feedback Descent optimization progress.
        Shows the best image at each iteration along with improvement status.
        """
        num_iterations = len(iteration_history)
        if num_iterations == 0:
            return

        # Create figure with two columns: text (left) and image (right)
        fig = plt.figure(figsize=(12, 3 * num_iterations))
        fig.suptitle(f"Feedback Descent: {image_name}", fontsize=16, fontweight="bold", y=0.995)

        for idx, entry in enumerate(iteration_history):
            iteration = entry["iteration"]
            image = entry["image"]
            improved = entry.get("improved", False)
            prompt = entry.get("prompt", "")

            # Text column (left)
            ax_text = plt.subplot(num_iterations, 2, 2 * idx + 1)
            ax_text.axis("off")

            if iteration == 0:
                full_text = f"Original Image"
            else:
                status = "✓ Improved" if improved else "= No change"
                title = f"Iteration {iteration}\n{status}\n"
                wrapped_prompt = textwrap.fill(prompt, width=40)
                full_text = f"{title}\nPrompt:\n{wrapped_prompt}"

            ax_text.text(0.5, 0.5, full_text, 
                        ha='center', va='center',
                        fontsize=9,
                        wrap=True,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            # Image column (right)
            ax_img = plt.subplot(num_iterations, 2, 2 * idx + 2)
            ax_img.imshow(image)
            ax_img.axis("off")

        plt.tight_layout()
        fig.savefig(viz_path, dpi=300, bbox_inches="tight")
        logging.info(f"📊 Visualization saved to {viz_path}\n")
        plt.close(fig)

    def _optimize_single_image(
        self,
        image_path: str,
        results_dir: str,
        image_idx: int,
        total_images: int
    ) -> dict:
        """
        Run Feedback Descent optimization loop on a single image.
        
        Algorithm:
        1. Initialize with base prior
        2. For each iteration:
            a. Propose improvement conditioned on best + feedback history
            b. Generate candidate image
            c. Compare candidate vs current best
            d. Get textual feedback
            e. If candidate wins, update best, otherwise accumulate feedback
        3. Stop after max_iterations or convergence
        """
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Check if already completed (for resume functionality)
        viz_path = os.path.join(results_dir, f"{image_name}_visualization.png")
        if os.path.isfile(viz_path):
            logging.info(f"Visualization for {image_name} already exists. Skipping.\n")
            return {
                "image_name": image_name,
                "iterations": 0,
                "improvements": 0,
                "final_prompt": None,
                "logs": []
            }

        logging.info(f"\n{'='*80}\n")
        logging.info(f"FEEDBACK DESCENT {image_idx+1}/{total_images}: {image_name}\n")
        logging.info(f"{'='*80}\n")

        # Load original image
        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

        original_image = Image.open(io.BytesIO(original_image_bytes))

        # Save original
        original_save_path = os.path.join(results_dir, f"{image_name}_original.jpg")
        original_image.save(original_save_path)

        # Initialize: current best starts as original (x*_0)
        current_best_prompt = None  # Will use base_prior only
        current_best_bytes = original_image_bytes
        current_best_image = original_image.copy()

        # Feedback history R = {(x1, r1), ..., (xt, rt)}
        feedback_history = []  # List of (prompt, rationale) tuples

        # Tracking for visualization
        iteration_history = [{"iteration": 0, "image": original_image.copy()}]
        optimization_logs = []

        # Convergence tracking
        iterations_without_improvement = 0

        for iteration in range(1, self.max_iterations + 1):
            logging.info(f"\n{'='*60}\n")
            logging.info(f"ITERATION {iteration}/{self.max_iterations}\n")
            logging.info(f"{'='*60}\n")

            iteration_log = {
                "iteration": iteration,
                "best_prompt_before": current_best_prompt,
            }

            # Step 1: Propose improvement - x_t = M(x*, R)
            proposed_prompt, feedback_text = self._propose_improvement(
                current_best_prompt,
                feedback_history,
                image_name.split('_')[0].lower()
            )

            iteration_log["proposed_prompt"] = proposed_prompt
            iteration_log["feedback_history_used"] = feedback_text

            # Step 2: Generate candidate image
            candidate_prompt = self._compose_prompt(proposed_prompt)
            logging.info(f"Generating candidate with prompt:\n{candidate_prompt}\n")

            candidate_image, candidate_bytes = self.image_editing_model.edit(
                candidate_prompt,
                current_best_bytes,
                original_image_bytes
            )

            if candidate_image is None:
                logging.warning(f"Image generation failed at iteration {iteration}, keeping current best.\n")
                iteration_log["error"] = "Image generation failed"
                optimization_logs.append(iteration_log)
                continue

            with self._counter_lock:
                self._total_num_images_generated += 1

            # Save candidate
            candidate_save_path = os.path.join(results_dir, f"{image_name}_iter-{iteration}_candidate.jpg")
            candidate_image.save(candidate_save_path)

            # Step 3: Compare candidate vs current best
            logging.info("Comparing candidate against current best...\n")

            winner, feedback = self._compare_images(
                candidate_bytes,
                current_best_bytes,
                image_name.split('_')[0].lower()
            )

            iteration_log["winner"] = winner
            iteration_log["feedback"] = feedback

            # Step 4: Add to feedback history - R = R ∪ {(x_t, r_t)}
            feedback_history.append((proposed_prompt, feedback))

            # Step 5: Update best if candidate wins (p_t = 1)
            improved = False
            if winner == 'candidate':
                logging.info(f"✅ Candidate wins (p_t=1)! Updating x* and resetting R.\n")
                current_best_prompt = proposed_prompt
                current_best_bytes = candidate_bytes
                current_best_image = candidate_image.copy()
                improved = True
                iterations_without_improvement = 0

                # Reset feedback history
                feedback_history = []
            else:
                logging.info(f"⏸️  Current best still better (p_t=0). Accumulating feedback in R.\n")
                iterations_without_improvement += 1

            iteration_log["improved"] = improved
            iteration_log["best_prompt_after"] = current_best_prompt
            optimization_logs.append(iteration_log)

            # Save current best at this iteration
            best_save_path = os.path.join(results_dir, f"{image_name}_iter-{iteration}_best.jpg")
            current_best_image.save(best_save_path)

            # Store for visualization
            iteration_history.append({
                "iteration": iteration,
                "image": current_best_image.copy(),
                "improved": improved,
                "prompt": current_best_prompt or "(no additional instruction yet)"
            })

            # Check convergence
            if iterations_without_improvement >= self.convergence_patience:
                logging.info(f"\n🎯 Converged! No improvement for {self.convergence_patience} iterations.\n")
                break

            # Log cost
            total_cost = self._total_num_images_generated * self._cost_per_image_generated
            logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")

        # Save final/best result
        final_save_path = os.path.join(results_dir, f"{image_name}_final.jpg")
        current_best_image.save(final_save_path)

        # Save zero-shot (first iteration's best)
        if len(iteration_history) > 1:
            zero_shot_path = os.path.join(results_dir, f"{image_name}_zero-shot.jpg")
            iteration_history[1]["image"].save(zero_shot_path)

        # Generate visualization
        viz_path = os.path.join(results_dir, f"{image_name}_visualization.png")
        self._visualize_optimization(image_name, iteration_history, viz_path)

        # Save detailed log
        log_path = os.path.join(results_dir, f"{image_name}_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(optimization_logs, f, indent=2, ensure_ascii=False)
        logging.info(f"📝 Detailed log saved to {log_path}\n")

        # Save summary
        summary_path = os.path.join(results_dir, f"{image_name}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Feedback Descent Summary: {image_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total iterations: {len(iteration_history)}\n")
            f.write(f"Images generated: {self._total_num_images_generated}\n")
            f.write(f"Improvements made: {sum(1 for h in iteration_history if h.get('improved', False))}\n\n")
            f.write(f"Optimization Progress:\n")
            for log in optimization_logs:
                f.write(f"\n  Iteration {log['iteration']}:\n")
                f.write(f"    Proposed: {log.get('proposed_prompt', 'N/A')}\n")
                f.write(f"    Result: {'✓ Improved' if log.get('improved') else '= No change'}\n")
                if 'feedback' in log:
                    f.write(f"    Feedback: {log['feedback']}\n")
            f.write(f"\nFinal Best Prompt:\n{current_best_prompt}\n")

        return {
            "image_name": image_name,
            "iterations": len(iteration_history),
            "improvements": sum(1 for h in iteration_history if h.get('improved', False)),
            "final_prompt": current_best_prompt,
            "logs": optimization_logs
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Run Feedback Descent optimization on all images.
        """
        run_start = time.time()

        logging.info(f"\n{'='*80}")
        logging.info(f"⚖️  Starting Feedback Descent Optimization")
        logging.info(f"   Total images: {len(image_paths)}")
        logging.info(f"   Max iterations per image: {self.max_iterations}")
        logging.info(f"   Convergence patience: {self.convergence_patience}")
        logging.info(f"   Max workers: {max_workers}")
        logging.info(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        # Process images (sequential recommended)
        results = []

        with tqdm(total=len(image_paths), desc="Images completed", unit="image") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._optimize_single_image,
                        image_path,
                        results_dir,
                        idx,
                        len(image_paths)
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
            f.write(f"Feedback Descent - Global Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total images processed: {len(image_paths)}\n")
            f.write(f"Total images generated: {self._total_num_images_generated}\n")
            f.write(f"Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Max iterations: {self.max_iterations}\n")
            f.write(f"  Convergence patience: {self.convergence_patience}\n\n")
            f.write(f"Per-image results:\n")
            for r in results:
                f.write(f"  - {r['image_name']}: {r['iterations']} iterations, {r['improvements']} improvements\n")

        run_duration = time.time() - run_start
        logging.info(f"\n{'='*80}")
        logging.info(f"✅ Feedback Descent Complete!")
        logging.info(f"⏱️  TOTAL RUNTIME: {run_duration:.2f}s ({run_duration/60:.2f}m)")
        logging.info(f"   Total images: {len(image_paths)}")
        logging.info(f"   Total images generated: {self._total_num_images_generated}")
        logging.info(f"   Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}")
        logging.info(f"{'='*80}\n")

        return results
