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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import textgrad as tg
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent

from utils.wrappers import ImageModel


@dataclasses.dataclass
class TextGradBaseline:
    """
    TextGrad-based single-image optimization baseline.
    Uses TextGrad's automatic differentiation through text to iteratively
    improve an image editing prompt based on evaluation feedback.
    """
    name: str
    base_prior: str
    image_editing_model: ImageModel
    # TextGrad engine name
    textgrad_engine: str
    # Evaluation prompt for the loss function
    evaluation_prompt: str
    # Instruction describing the optimization goal
    optimization_instruction: str
    # Number of optimization iterations
    max_iterations: int
    # Gradient memory for TGD optimizer (0 = no memory)
    gradient_memory: int
    # Constraints for the optimizer
    constraints: list

    def __post_init__(self):
        """Initialize tracking variables and TextGrad engine."""
        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039
        self._counter_lock = threading.Lock()

        # Initialize TextGrad engine
        engine_name = f"experimental:{self.textgrad_engine}"
        logging.info(f"Initializing TextGrad engine: {engine_name}")
        self._engine = tg.get_engine(engine_name)
        tg.set_backward_engine(self._engine, override=True)

    def _compose_prompt(self, instruction: Optional[str] = None) -> str:
        """
        Combine the base template with the current instruction.
        Falls back to base_prior when instruction is empty.
        """
        if instruction and (len(instruction.strip()) > 0):
            return "%s\n\n%s" % (self.base_prior.strip(), instruction)
        elif self.base_prior:
            return self.base_prior.strip()

    def _create_loss_function(self, category: str) -> tg.TextLoss:
        """
        Create a TextGrad loss function for evaluating image quality.
        """
        eval_prompt = self.evaluation_prompt.format(category=category)

        loss_fn = tg.TextLoss(
            eval_system_prompt=eval_prompt,
            engine=self._engine
        )
        return loss_fn

    def _evaluate_image(
        self,
        prompt_variable: Variable,
        category: str,
        full_editing_prompt: str
    ) -> Variable:
        """
        Evaluate the current image and return a loss Variable.
        """
        # Get the instruction part (what we're optimizing)
        instruction = prompt_variable.value if prompt_variable.value else "(no additional instruction yet)"

        # Create a description of what we're optimizing
        optimization_context = (
            f"FULL EDITING PROMPT USED: {full_editing_prompt}\n"
            f"ADDITIONAL INSTRUCTION BEING OPTIMIZED: {instruction}\n"
            f"CONTEXT: The user is looking for a(n) {category}\n"
            f"{self.optimization_instruction}"
        )

        context_var = Variable(
            value=optimization_context,
            requires_grad=False,
            role_description="Context about the current optimization state"
        )

        # Combine prompt and context for evaluation
        combined = prompt_variable + context_var

        # Create and apply loss function
        loss_fn = self._create_loss_function(category)
        loss = loss_fn(combined)

        return loss

    def _visualize_optimization(
        self,
        category: str,
        iteration_history: list[dict],
        viz_path: str
    ):
        """
        Create a visualization of the optimization progress.
        Shows the image at each iteration along with the prompt used.
        """
        num_iterations = len(iteration_history)
        if num_iterations == 0:
            return

        # Arrange images vertically (1 column, multiple rows)
        rows = num_iterations
        cols = 1

        fig, axes = plt.subplots(rows, cols, figsize=(8, 6 * rows))

        # Handle single iteration case
        if rows == 1:
            axes = [axes]

        fig.suptitle(f"TextGrad Optimization: {category}", fontsize=16, fontweight="bold", y=1.02)

        for idx, entry in enumerate(iteration_history):
            ax = axes[idx]

            iteration = entry["iteration"]
            image = entry["image"]

            ax.imshow(image)
            ax.set_title(f"Iteration {iteration}", fontsize=10)
            ax.axis("off")

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
        Run TextGrad optimization loop on a single image.

        The optimization loop:
        1. Start with base prior as initial prompt
        2. Edit image with current prompt
        3. Evaluate the result using TextGrad loss
        4. Backward pass to compute gradients on prompt
        5. Update prompt using TGD optimizer
        6. Repeat until max_iterations
        """
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        logging.info(f"\n{'='*80}\n")
        logging.info(f"TEXTGRAD OPTIMIZATION {image_idx+1}/{total_images}: {image_name}\n")
        logging.info(f"{'='*80}\n")

        # Load original image
        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

        original_image = Image.open(io.BytesIO(original_image_bytes))

        # Save original
        original_save_path = os.path.join(results_dir, f"{image_name}_original.jpg")
        original_image.save(original_save_path)

        # Initialize the prompt as a TextGrad Variable
        initial_instruction = ""
        prompt_variable = Variable(
            value=initial_instruction,
            requires_grad=True,
            role_description="image editing instruction to make the product more visually appealing"
        )

        # Create optimizer
        optimizer = TextualGradientDescent(
            parameters=[prompt_variable],
            engine=self._engine,
            constraints=self.constraints,
            gradient_memory=self.gradient_memory
        )

        # Tracking
        iteration_history = []
        optimization_logs = []

        current_image_bytes = original_image_bytes
        current_image = original_image.copy()
        best_image = original_image.copy()
        best_image_bytes = original_image_bytes
        best_prompt = None

        for iteration in range(self.max_iterations):
            logging.info(f"\n{'='*60}\n")
            logging.info(f"ITERATION {iteration + 1}/{self.max_iterations}\n")
            logging.info(f"{'='*60}\n")

            iteration_log = {
                "iteration": iteration + 1,
                "prompt_before": prompt_variable.value,
            }

            # Zero gradients
            optimizer.zero_grad()

            # Generate edited image with current prompt
            editing_prompt = self._compose_prompt(prompt_variable.value if prompt_variable.value else None)
            logging.info(f"Editing prompt:\n{editing_prompt}\n")

            edited_image, edited_image_bytes = self.image_editing_model.edit(
                editing_prompt,
                current_image_bytes,
                original_image_bytes
            )

            if edited_image is None:
                logging.warning(f"Image editing failed at iteration {iteration + 1}, keeping previous image.\n")
                continue

            with self._counter_lock:
                self._total_num_images_generated += 1

            current_image = edited_image
            current_image_bytes = edited_image_bytes

            # Save iteration result
            iter_save_path = os.path.join(results_dir, f"{image_name}_iter-{iteration + 1}.jpg")
            current_image.save(iter_save_path)

            # Store for visualization
            iteration_history.append({
                "iteration": iteration + 1,
                "image": current_image.copy()
            })

            # Compute loss using TextGrad
            logging.info("Computing TextGrad loss...\n")

            try:
                loss = self._evaluate_image(
                    prompt_variable,
                    image_name.split('_')[0].lower(),
                    editing_prompt  # Pass the full prompt that was actually used
                )

                logging.info(f"Loss evaluation:\n{loss.value}\n")
                iteration_log["loss_evaluation"] = loss.value

                # Backward pass - compute gradients
                logging.info("Computing gradients via backward pass...\n")
                loss.backward()

                # Get gradient text for logging
                gradient_text = prompt_variable.get_gradient_text()
                logging.info(f"Gradient (feedback):\n{gradient_text if gradient_text else 'No gradient'}\n")
                iteration_log["gradient"] = gradient_text

                # Optimizer step - update the prompt
                logging.info("Updating prompt via TGD optimizer...\n")
                optimizer.step()

                iteration_log["prompt_after"] = prompt_variable.value
                logging.info(f"Updated prompt:\n{prompt_variable.value}\n")

            except Exception as e:
                logging.error(f"TextGrad optimization error at iteration {iteration + 1}: {e}\n")
                iteration_log["error"] = str(e)

            optimization_logs.append(iteration_log)

            # Update best image
            best_image = current_image.copy()
            best_image_bytes = current_image_bytes
            best_prompt = prompt_variable.value

            # Log cost
            total_cost = self._total_num_images_generated * self._cost_per_image_generated
            logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")

        # Save final/best result
        final_save_path = os.path.join(results_dir, f"{image_name}_final.jpg")
        best_image.save(final_save_path)

        # Save zero-shot (first iteration)
        if len(iteration_history) > 0:
            zero_shot_path = os.path.join(results_dir, f"{image_name}_zero-shot.jpg")
            iteration_history[0]["image"].save(zero_shot_path)

        # Generate visualization
        viz_path = os.path.join(results_dir, f"{image_name}_visualization.png")
        self._visualize_optimization(image_name.split('_')[0].lower(), iteration_history, viz_path)

        # Save detailed log
        log_path = os.path.join(results_dir, f"{image_name}_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(optimization_logs, f, indent=2, ensure_ascii=False)
        logging.info(f"📝 Detailed log saved to {log_path}\n")

        # Save summary
        summary_path = os.path.join(results_dir, f"{image_name}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"TextGrad Optimization Summary: {image_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total iterations: {self.max_iterations}\n")
            f.write(f"Images generated: {len(iteration_history)}\n\n")
            f.write(f"Prompt Evolution:\n")
            for i, log in enumerate(optimization_logs):
                f.write(f"\n  Iteration {i + 1}:\n")
                f.write(f"    Before: {log.get('prompt_before', 'N/A')}\n")
                f.write(f"    After: {log.get('prompt_after', 'N/A')}\n")
            f.write(f"\nFinal Prompt:\n{best_prompt}\n")

        return {
            "image_name": image_name,
            "iterations": len(iteration_history),
            "final_prompt": best_prompt,
            "logs": optimization_logs
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Run TextGrad optimization on all images.
        """
        run_start = time.time()

        # Check for comparability results to filter images
        image_dir = os.path.dirname(image_paths[0]) if image_paths else ""
        comparability_results_csv = os.path.join(image_dir, "comparability_results.csv")

        images_to_process = []

        if os.path.isfile(comparability_results_csv):
            logging.info(f"Reading comparable images from: {comparability_results_csv}")
            seen_images = set()
            with open(comparability_results_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['is_comparable'].lower() == "true":
                        for img_id in [row['id_1'], row['id_2']]:
                            if img_id not in seen_images:
                                img_path = os.path.join(image_dir, img_id + '.jpg')
                                if os.path.isfile(img_path):
                                    images_to_process.append(img_path)
                                    seen_images.add(img_id)
        else:
            # Process all provided images
            images_to_process = image_paths

        if not images_to_process:
            logging.error("No images found to process.")
            raise ValueError("No images found to process.")

        logging.info(f"\n{'='*80}")
        logging.info(f"🎯 Starting TextGrad Baseline Optimization")
        logging.info(f"   Total images: {len(images_to_process)}")
        logging.info(f"   Max iterations per image: {self.max_iterations}")
        logging.info(f"   TextGrad engine: {self.textgrad_engine}")
        logging.info(f"   Gradient memory: {self.gradient_memory}")
        logging.info(f"   Max workers: {max_workers}")
        logging.info(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        # Process images
        results = []
        with tqdm(total=len(images_to_process), desc="Images optimized", unit="image") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._optimize_single_image,
                        image_path,
                        results_dir,
                        idx,
                        len(images_to_process)
                    ): image_path
                    for idx, image_path in enumerate(images_to_process)
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Generate global summary
        summary_path = os.path.join(results_dir, "global_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"TextGrad Baseline - Global Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total images processed: {len(images_to_process)}\n")
            f.write(f"Total images generated: {self._total_num_images_generated}\n")
            f.write(f"Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  TextGrad engine: {self.textgrad_engine}\n")
            f.write(f"  Max iterations: {self.max_iterations}\n")
            f.write(f"  Gradient memory: {self.gradient_memory}\n")
            f.write(f"  Constraints: {self.constraints}\n\n")
            f.write(f"Per-image results:\n")
            for r in results:
                f.write(f"  - {r['image_name']}: {r['iterations']} iterations\n")

        run_duration = time.time() - run_start
        logging.info(f"\n{'='*80}")
        logging.info(f"✅ TextGrad Baseline Complete!")
        logging.info(f"⏱️  TOTAL RUNTIME: {run_duration:.2f}s ({run_duration/60:.2f}m)")
        logging.info(f"   Total images: {len(images_to_process)}")
        logging.info(f"   Total images generated: {self._total_num_images_generated}")
        logging.info(f"   Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}")
        logging.info(f"{'='*80}\n")

        return results
