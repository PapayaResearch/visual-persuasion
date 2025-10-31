import os
import io
import logging
import dataclasses
from PIL import Image
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel, LanguageModel

@dataclasses.dataclass
class VisualNudge:
    """
    Orchestrates the visual nudging pipeline.
    Hydra instantiates this class and its model dependencies automatically.
    """
    iterations: int
    editing_context_prompt: str
    initial_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    num_judges: int
    evaluator_prompt: str
    evaluator_model: LanguageModel
    optimizer_model: LanguageModel

    def _process_single_image(
        self,
        image_path: str,
        img_idx: int,
        total_images: int,
        results_dir: str,
        pbar: tqdm
    ) -> None:
        """
        Processes a single image through the optimization loop.
        """
        base_filename, _ = os.path.splitext(os.path.basename(image_path))
        logging.info(f"\n===== Processing Image {img_idx + 1}/{total_images}: {base_filename} =====\n")

        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

            original_image = Image.open(io.BytesIO(original_image_bytes))
            original_save_path = os.path.join(results_dir, f"{base_filename}_iter-0-original.jpg")
            original_image.save(original_save_path)
            logging.info(f"Saved original image to: {original_save_path}\n")

            current_prompt = self.initial_prompt
            best_prompt = current_prompt
            best_image = original_image
            context_image_bytes = original_image_bytes

            for iter in range(self.iterations + 1):
                logging.info("\n>> ITERATION " + ("BEST" if iter == self.iterations else f"{iter + 1}/{self.iterations}") + " <<\n")
                logging.info(f"PROMPT:\n{current_prompt}\n")

                # Edit image (using the original image for the first iteration)
                editing_prompt = f"{current_prompt}\n{self.editing_context_prompt}"
                edited_image, edited_image_bytes = self.image_editing_model.edit(
                    f"{editing_prompt}\n{self.background_state_prompt}",
                    original_image_bytes,
                    context_image_bytes
                )

                edited_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter + 1}-a_edited.jpg")
                edited_image.save(edited_image_save_path)
                logging.info(f"Saved edited image to: {edited_image_save_path}\n")

                # Evaluate edit
                selected_choice, aggregated_reason = self._evaluate(
                    edited_image_bytes=edited_image_bytes,
                    original_image_bytes=original_image_bytes,
                    context_image_bytes=context_image_bytes
                )

                if selected_choice.lower() == "edited":
                    logging.info("VLM preferred the new image. Updating context for next iteration.\n")
                    context_image_bytes = edited_image_bytes
                    best_prompt = current_prompt
                    best_image = edited_image
                elif selected_choice.lower() == "original":
                    best_image = Image.open(io.BytesIO(context_image_bytes))
                    logging.info("VLM preferred the old image. Retaining previous context for next iteration.\n")
                
                best_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter + 1}-b_best.jpg")
                best_image.save(best_image_save_path)
                logging.info(f"Saved best image to: {best_image_save_path}\n")

                if iter == self.iterations:
                    best_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-n-edit.jpg")
                    best_image.save(best_image_save_path)
                    best_prompt_save_path = os.path.join(results_dir, f"{base_filename}_iter-n-prompt.txt")
                    with open(best_prompt_save_path, "w") as f:
                        f.write(best_prompt)
                    logging.info(f"Saved best image to: {best_image_save_path}\n")
                    logging.info(f"Best prompt:\n{best_prompt}\n")
                    break
                
                # Get new prompt from the optimizer
                response = self.optimizer_model.get_response(
                    current_prompt=best_prompt,
                    reason=aggregated_reason
                )

                logging.info(f"{response}\n")

                # Update the prompt for the next iteration
                current_prompt = response.new_prompt


                # Update progress at end of iteration
                pbar.update(1)

    def _evaluate(self, edited_image_bytes: bytes, original_image_bytes: bytes, context_image_bytes: bytes) -> tuple[str, str]:
        """
        Evaluates the edited image against the original (or context) image using multiple judges.
        Returns majority vote and aggregated reasons.
        """
        choices = {}
        reasons = {}
        comparison_image_bytes = context_image_bytes if context_image_bytes else original_image_bytes
        
        # Create evaluation tasks
        eval_tasks = []
        for _ in range(self.num_judges):
            for is_edited_first in [True, False]:
                eval_tasks.append(is_edited_first)
        
        def evaluate_single(is_edited_first: bool):
            """Single evaluation task"""
            logging.info(f"Evaluating with edited image as {'first' if is_edited_first else 'second'} image.\n")
            
            images = [edited_image_bytes, comparison_image_bytes] if is_edited_first else [comparison_image_bytes, edited_image_bytes]
            choice_map = {
                "first": "edited" if is_edited_first else "original",
                "second": "original" if is_edited_first else "edited",
            }
            
            logging.info(f"Choice map: {choice_map}\n")
            
            evaluation = self.evaluator_model.get_response(
                task=self.evaluator_prompt,
                images=images
            )
            
            real_choice = choice_map.get(evaluation.choice.lower())
            
            if not real_choice:
                logging.warning("Evaluation failed. Skipping to next judge.\n")
                return None
            
            logging.info(f"{evaluation}\n")
            return (real_choice, evaluation.reason)
        
        # Run evaluations in parallel with nested ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(eval_tasks), 8)) as eval_executor:
            eval_futures = [eval_executor.submit(evaluate_single, task) for task in eval_tasks]
            
            for future in as_completed(eval_futures):
                result = future.result()
                if result:
                    real_choice, reason = result
                    choices[real_choice] = choices.get(real_choice, 0) + 1
                    reasons[real_choice] = reasons.get(real_choice, []) + [reason]
        
        # Choose majority
        choice = max(choices, key=choices.get)
        reason = "\n".join(reasons[choice])
        
        return choice, reason

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the optimization loop for each image, optionally in parallel.
        """
        total_iterations = len(image_paths) * (self.iterations + 1)

        with tqdm(total=total_iterations, desc="Total progress", unit="iter", position=0) as pbar_total:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_single_image, image_path, idx, len(image_paths), results_dir, pbar_total): image_path
                          for idx, image_path in enumerate(image_paths)}

                for future in tqdm(as_completed(futures), total=len(image_paths), desc="Images completed", unit="image", position=1, leave=False):
                    future.result()
