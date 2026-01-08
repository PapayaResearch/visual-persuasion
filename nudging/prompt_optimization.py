import os
import io
import time
import logging
import dataclasses
from PIL import Image
from typing import Optional
from utils.wrappers import ImageModel, LanguageModel


@dataclasses.dataclass
class VisualPromptOptimizer:
    """
    Iteratively optimizes a single image prompt based on judge feedback.
    The judge evaluates each iteration and provides a reward signal,
    which is used to guide the next round of improvements.
    """
    name: str
    base_prior: str
    editing_context_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    judge_prompt: str
    evaluator_model: LanguageModel

    # Optimizer settings
    num_improvement_proposals: int
    proposer_model: LanguageModel
    selector_model: LanguageModel

    # Optimization parameters
    max_iterations: int = 10
    reward_threshold: float = 0.9  # Stop if reward exceeds this
    min_improvement: float = 0.05  # Minimum reward increase to continue

    def __post_init__(self):
        """Initialize tracking variables."""
        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039
        self.optimization_history = []

    def _compose_prompt(self, instruction: Optional[str] = None) -> str:
        """
        Merge the base prior with a specific instruction (if provided).
        """
        base = self.base_prior
        extra = instruction

        if base and extra:
            return f"{base}\n\n{extra}"
        return extra or base

    def _build_full_editing_prompt(self, instruction: Optional[str]) -> str:
        """
        Append editing context and background prompts to the composed instruction.
        """
        segments = [self._compose_prompt(instruction)]
        if self.editing_context_prompt:
            segments.append(self.editing_context_prompt.strip())
        if self.background_state_prompt:
            segments.append(self.background_state_prompt.strip())
        return "\n".join(segment for segment in segments if segment).strip()

    def _evaluate_image(
        self,
        image_bytes: bytes,
        iteration: int
    ) -> tuple[float, str]:
        """
        Evaluate an image using the judge model.
        Returns: (reward_score, feedback_text)
        reward_score is between 0-1, where 1 is perfect
        """
        logging.info(f"\n📊 EVALUATING iteration {iteration}\n")

        evaluation = self.evaluator_model.get_response(
            images=[image_bytes],
            judge_prompt=self.judge_prompt
        )

        # Assuming evaluator returns a score (0-1) and reasoning
        reward = getattr(evaluation, 'score', 0.5)  # Default to 0.5 if no score
        feedback = evaluation.reason

        logging.info(f"Reward: {reward:.2%}\n")
        logging.info(f"Feedback: {feedback}\n")

        return reward, feedback

    def _generate_candidate_images(
        self,
        current_prompt: str,
        current_image_bytes: bytes,
        original_image_bytes: bytes,
        feedback: str,
        iteration: int,
        history: list[str],
        results_dir: str
    ) -> list[dict]:
        """
        Generate multiple candidate improved images using proposer model.
        Returns list of candidate dicts with image data and prompts.
        """
        logging.info(f"\n🔧 GENERATING CANDIDATES for iteration {iteration}\n")

        # Get improvement proposals
        history_text = "\n".join([f"  - {p}" for p in history]) if history else "None"

        proposer_response = self.proposer_model.get_response(
            current_prompt=current_prompt,
            history_of_prompts=history_text,
            current_iteration=iteration,
            judge_feedback=feedback,
            total_iterations=self.max_iterations,
            num_proposals=self.num_improvement_proposals
        )

        candidate_prompts = proposer_response.candidate_prompts
        logging.info(f"Generated {len(candidate_prompts)} proposals:\n")
        for i, prompt in enumerate(candidate_prompts, 1):
            logging.info(f"  Candidate {i}: {prompt}\n")

        # Generate images for each proposal
        candidates = []
        for i, prompt in enumerate(candidate_prompts, 1):
            logging.info(f"Generating image {i}/{len(candidate_prompts)}\n")

            editing_prompt = self._build_full_editing_prompt(prompt)
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                editing_prompt,
                current_image_bytes,
                original_image_bytes
            )

            if edited_image is None or edited_image_bytes is None:
                logging.warning(f"Candidate {i} generation failed; skipping.\n")
                continue

            self._total_num_images_generated += 1

            # Save candidate
            candidate_path = os.path.join(
                results_dir,
                f"iteration-{iteration:02d}_candidate-{i}.jpg"
            )
            edited_image.save(candidate_path)

            candidates.append({
                "prompt": prompt,
                "full_prompt": editing_prompt,
                "image": edited_image,
                "image_bytes": edited_image_bytes,
                "path": candidate_path,
                "index": i
            })

        return candidates

    def _select_best_candidate(
        self,
        candidates: list[dict],
        feedback: str
    ) -> dict:
        """
        Use selector model to choose the best candidate.
        Returns the selected candidate dict.
        """
        logging.info(f"\n🎯 SELECTING BEST from {len(candidates)} candidates\n")

        image_bytes_list = [c["image_bytes"] for c in candidates]
        descriptions = [f"Candidate {c['index']}: {c['prompt']}"
                       for c in candidates]

        selector_response = self.selector_model.get_response(
            images=image_bytes_list,
            candidate_descriptions="\n".join(descriptions),
            num_candidates=len(candidates),
            judge_feedback=feedback
        )

        selected_idx = int(selector_response.choice) - 1
        best_candidate = candidates[selected_idx]

        logging.info(f"✅ Selected candidate {best_candidate['index']}: "
                    f"{best_candidate['prompt']}\n")

        return best_candidate

    def optimize(
        self,
        initial_image_path: str,
        results_dir: str
    ) -> dict:
        """
        Run the optimization loop on a single image.

        Returns:
            dict with final state including:
            - final_image: PIL Image
            - final_prompt: str
            - final_reward: float
            - history: list of iteration results
        """
        start_time = time.time()

        logging.info(f"\n{'='*80}\n")
        logging.info(f"🎯 STARTING PROMPT OPTIMIZATION\n")
        logging.info(f"Image: {initial_image_path}\n")
        logging.info(f"Max iterations: {self.max_iterations}\n")
        logging.info(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        # Load initial image
        with open(initial_image_path, "rb") as f:
            original_image_bytes = f.read()

        original_image = Image.open(io.BytesIO(original_image_bytes))
        original_image.save(os.path.join(results_dir, "00_original.jpg"))

        # Initialize state
        current_image_bytes = original_image_bytes
        current_prompt = self.base_prior
        current_image = original_image
        edit_history = []

        # Apply initial base prior edit (zero-shot)
        logging.info("\n🚀 Applying initial base prior edit\n")
        editing_prompt = self._build_full_editing_prompt(None)
        edited_image, edited_image_bytes = self.image_editing_model.edit(
            editing_prompt,
            original_image_bytes,
            original_image_bytes
        )

        if edited_image is not None:
            self._total_num_images_generated += 1
            current_image = edited_image
            current_image_bytes = edited_image_bytes
            current_prompt = editing_prompt
            edit_history.append(editing_prompt)
            current_image.save(os.path.join(results_dir, "01_base_prior.jpg"))

        # Evaluate initial state
        best_reward, feedback = self._evaluate_image(current_image_bytes, 0)

        self.optimization_history.append({
            "iteration": 0,
            "prompt": current_prompt,
            "reward": best_reward,
            "feedback": feedback
        })

        # Optimization loop
        iteration = 1
        while iteration <= self.max_iterations:
            logging.info(f"\n{'='*60}\n")
            logging.info(f"ITERATION {iteration}/{self.max_iterations}\n")
            logging.info(f"Current reward: {best_reward:.2%}\n")
            logging.info(f"{'='*60}\n")

            # Check stopping criteria
            if best_reward >= self.reward_threshold:
                logging.info(f"\n🎉 Reward threshold reached: {best_reward:.2%}\n")
                break

            # Generate candidate improvements
            candidates = self._generate_candidate_images(
                current_prompt=current_prompt,
                current_image_bytes=current_image_bytes,
                original_image_bytes=original_image_bytes,
                feedback=feedback,
                iteration=iteration,
                history=edit_history,
                results_dir=results_dir
            )

            if not candidates:
                logging.warning("No valid candidates generated. Stopping.\n")
                break

            # Select best candidate
            best_candidate = self._select_best_candidate(candidates, feedback)

            # Evaluate selected candidate
            candidate_reward, candidate_feedback = self._evaluate_image(
                best_candidate["image_bytes"],
                iteration
            )

            # Check for improvement
            reward_improvement = candidate_reward - best_reward

            if reward_improvement < self.min_improvement:
                logging.info(f"\n📉 Insufficient improvement: {reward_improvement:.2%}\n")
                logging.info("Stopping optimization.\n")
                break

            # Update state
            current_image_bytes = best_candidate["image_bytes"]
            current_prompt = best_candidate["prompt"]
            current_image = best_candidate["image"]
            edit_history.append(best_candidate["prompt"])
            best_reward = candidate_reward
            feedback = candidate_feedback

            # Save iteration result
            current_image.save(os.path.join(
                results_dir,
                f"iteration-{iteration:02d}_selected.jpg"
            ))

            with open(os.path.join(results_dir, f"iteration-{iteration:02d}_prompt.txt"), "w") as f:
                f.write(best_candidate["full_prompt"])

            # Record history
            self.optimization_history.append({
                "iteration": iteration,
                "prompt": current_prompt,
                "reward": best_reward,
                "feedback": feedback,
                "improvement": reward_improvement
            })

            iteration += 1

        # Save final results
        current_image.save(os.path.join(results_dir, "final.jpg"))

        # Generate summary
        duration = time.time() - start_time
        total_cost = self._total_num_images_generated * self._cost_per_image_generated

        summary_path = os.path.join(results_dir, "optimization_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Prompt Optimization Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Initial reward: {self.optimization_history[0]['reward']:.2%}\n")
            f.write(f"Final reward: {best_reward:.2%}\n")
            f.write(f"Improvement: {best_reward - self.optimization_history[0]['reward']:.2%}\n")
            f.write(f"Total iterations: {iteration - 1}\n")
            f.write(f"Images generated: {self._total_num_images_generated}\n")
            f.write(f"Estimated cost: ${total_cost:.2f}\n")
            f.write(f"Duration: {duration:.2f}s ({duration/60:.2f}m)\n\n")

            f.write(f"Iteration History:\n")
            f.write(f"{'-'*60}\n")
            for entry in self.optimization_history:
                f.write(f"\nIteration {entry['iteration']}:\n")
                f.write(f"  Reward: {entry['reward']:.2%}\n")
                if 'improvement' in entry:
                    f.write(f"  Improvement: {entry['improvement']:.2%}\n")
                f.write(f"  Prompt: {entry['prompt']}\n")
                f.write(f"  Feedback: {entry['feedback']}\n")

            f.write(f"\n{'='*60}\n")
            f.write(f"Final Prompt:\n{current_prompt}\n")

        logging.info(f"\n{'='*80}\n")
        logging.info(f"✅ OPTIMIZATION COMPLETE\n")
        logging.info(f"Final reward: {best_reward:.2%}\n")
        logging.info(f"Total iterations: {iteration - 1}\n")
        logging.info(f"Images generated: {self._total_num_images_generated}\n")
        logging.info(f"Cost: ${total_cost:.2f}\n")
        logging.info(f"Duration: {duration:.2f}s\n")
        logging.info(f"{'='*80}\n")

        return {
            "final_image": current_image,
            "final_image_bytes": current_image_bytes,
            "final_prompt": current_prompt,
            "final_reward": best_reward,
            "history": self.optimization_history,
            "iterations": iteration - 1,
            "images_generated": self._total_num_images_generated,
            "cost": total_cost,
            "duration": duration
        }
