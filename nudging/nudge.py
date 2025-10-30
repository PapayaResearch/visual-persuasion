import os
import io
import logging
from PIL import Image
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel, LanguageModel

class VisualNudge:
    """
    Orchestrates the visual nudging pipeline.
    Hydra instantiates this class and its model dependencies automatically.
    """
    def __init__(
        self,
        enable_optimization: bool,
        iterations: int,
        enable_editing_context: bool,
        editing_context_prompt: str,
        enable_tournament_mode: bool,
        save_best_prompts: bool,
        initial_prompt: str,
        background_state_prompt: str,
        image_editing_model: ImageModel,
        evaluator_prompt: str,
        evaluator_model: LanguageModel,
        loss_model: LanguageModel,
        optimizer_model: LanguageModel,
    ):
        self.enable_optimization = enable_optimization
        self.iterations = iterations
        self.enable_editing_context = enable_editing_context
        self.editing_context_prompt = editing_context_prompt
        self.enable_tournament_mode = enable_tournament_mode
        self.save_best_prompts = save_best_prompts
        self.initial_prompt = initial_prompt
        self.background_state_prompt = background_state_prompt
        self.image_editing_model = image_editing_model
        self.evaluator_prompt = evaluator_prompt
        self.evaluator_model = evaluator_model
        self.loss_model = loss_model
        self.optimizer_model = optimizer_model

    def _process_single_image(self, image_path: str, img_idx: int, total_images: int, results_dir: str, pbar=None):
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
            context_image_bytes = None

            for iter in range(self.iterations + 1):
                logging.info("\n>> ITERATION " + ("BEST" if iter == self.iterations else f"{iter + 1}/{self.iterations}") + " <<\n")
                logging.info(f"PROMPT:\n{current_prompt}\n")

                # 1. Edit image with current prompt
                if self.enable_editing_context and context_image_bytes:
                    if self.enable_tournament_mode and self.save_best_prompts:
                        # Regenerate context image from best prompt so far
                        context_image, context_image_bytes = self.image_editing_model.edit(f"{best_prompt}\n{self.background_state_prompt}", original_image_bytes)
                        if not context_image:
                            logging.error("Context image regeneration failed. Skipping to next iteration.\n")
                            continue
                        context_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter+1}-context.jpg")
                        context_image.save(context_image_save_path)
                        logging.info(f"Regenerated and saved context image to: {context_image_save_path}\n")

                    # Use original and previous edited images for context
                    edited_prompt = f"{current_prompt}\n{self.editing_context_prompt}"
                    edited_image, edited_image_bytes = self.image_editing_model.edit(f"{edited_prompt}\n{self.background_state_prompt}", original_image_bytes, context_image_bytes)
                else:
                    # Use only the original image
                    edited_image, edited_image_bytes = self.image_editing_model.edit(f"{current_prompt}\n{self.background_state_prompt}", original_image_bytes)

                if not edited_image:
                    logging.error("Image editing failed. Skipping to next iteration.\n")
                    continue
                
                if iter == self.iterations:
                    best_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-n-edit.jpg")
                    edited_image.save(best_image_save_path)
                    logging.info(f"Saved best image to: {best_image_save_path}\n")
                    break

                edited_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter+1}-edit.jpg")
                edited_image.save(edited_image_save_path)
                logging.info(f"Saved edited image to: {edited_image_save_path}\n")

                if self.enable_optimization:
                    # 2. Evaluate the edit
                    if self.enable_tournament_mode and context_image_bytes:
                        # Use the previous and current edited images
                        evaluation = self.evaluator_model.get_response(
                            task=self.evaluator_prompt,
                            images=[context_image_bytes, edited_image_bytes]
                        )
                    else:
                        # Use the original and current edited images
                        evaluation = self.evaluator_model.get_response(
                            task=self.evaluator_prompt,
                            images=[original_image_bytes, edited_image_bytes]
                        )

                    if not evaluation:
                        logging.error("Evaluation failed. Skipping to next iteration.\n")
                        continue
                    logging.info(f"{evaluation}\n")

                    # 3. Get critique (loss)
                    critique = self.loss_model.get_response(
                        choice=evaluation.choice,
                        reason=evaluation.reason,
                    )

                    if not critique:
                        logging.error("Critique generation failed. Skipping to next iteration.")
                        continue
                    logging.info(f"{critique}\n")

                    # 4. Get new prompt from optimizer
                    response = self.optimizer_model.get_response(
                        current_prompt=current_prompt,
                        suggestions=critique.suggestions
                    )

                    if not response:
                        logging.error("Prompt optimization failed. Skipping to next iteration.\n")
                        continue
                    logging.info(f"{response}\n")

                    new_prompt = response.new_prompt

                    if self.enable_tournament_mode:
                        # In tournament mode, only update context if the new image was preferred
                        if evaluation.choice.lower() == "edited":
                            context_image_bytes = edited_image_bytes
                            best_prompt = current_prompt
                            logging.info("VLM preferred the new image. Updating context for next iteration.\n")
                        else:
                            logging.info("VLM preferred the old image. Retaining previous context for next iteration.\n")
                    else:
                        # Otherwise update context to the latest edit
                        context_image_bytes = edited_image_bytes
                        best_prompt = current_prompt

                    # Update the prompt for the next iteration
                    current_prompt = new_prompt

                # Update progress at end of iteration
                if pbar:
                    pbar.update(1)

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the optimization loop for each image, optionally in parallel.
        """
        total_iterations = len(image_paths) * (self.iterations + 1)

        with tqdm(total=total_iterations, desc="Total progress", unit="iter") as pbar_total:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_single_image, image_path, idx, len(image_paths), results_dir, pbar_total): image_path
                          for idx, image_path in enumerate(image_paths)}

                for future in as_completed(futures):
                    future.result()