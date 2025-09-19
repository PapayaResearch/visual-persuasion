import os
import io
import logging
from PIL import Image
from wrappers import ImageEditingModel, EvaluatorModel, LossModel, OptimizerModel

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
        enable_evaluation_context: bool,
        enable_tournament_mode: bool,
        save_best_prompts: bool,
        enhance_original: bool,
        initial_prompt: str,
        enhance_prompt: str,
        image_editing_model: ImageEditingModel,
        no_context_prompt: str,
        context_prompt: str,
        evaluator_model: EvaluatorModel, 
        loss_model: LossModel, 
        optimizer_model: OptimizerModel,
    ):
        self.enable_optimization = enable_optimization
        self.iterations = iterations
        self.enable_editing_context = enable_editing_context
        self.enable_evaluation_context = enable_evaluation_context
        self.enable_tournament_mode = enable_tournament_mode
        self.save_best_prompts = save_best_prompts
        self.enhance_original = enhance_original
        self.initial_prompt = initial_prompt
        self.enhance_prompt = enhance_prompt
        self.image_editing_model = image_editing_model
        self.no_context_prompt = no_context_prompt
        self.context_prompt = context_prompt
        self.evaluator_model = evaluator_model
        self.loss_model = loss_model
        self.optimizer_model = optimizer_model

    def run(self, image_paths: list, results_dir: str):
        """
        Runs the optimization loop for each image.
        """
        for img_idx, image_path in enumerate(image_paths):
            base_filename, _ = os.path.splitext(os.path.basename(image_path))
            logging.info(f"\n===== Processing Image {img_idx + 1}/{len(image_paths)}: {base_filename} =====\n")

            with open(image_path, "rb") as f:
                original_image_bytes = f.read()

            original_image = Image.open(io.BytesIO(original_image_bytes))
            original_save_path = os.path.join(results_dir, f"{base_filename}_original.jpg")
            original_image.save(original_save_path)
            logging.info(f"Saved original image to: {original_save_path}\n")

            # Optionally enhance the original image before starting
            if self.enhance_original:
                logging.info("\n--- Enhancing Original Image ---\n")
                enhanced_image, enhanced_image_bytes = self.image_editing_model.edit(self.enhance_prompt, original_image_bytes)
                
                if enhanced_image is None or enhanced_image_bytes is None:
                    logging.error("Enhancement failed. Proceeding with the original image.\n")
                else:
                    enhanced_save_path = os.path.join(results_dir, f"{base_filename}_enhanced.jpg")
                    enhanced_image.save(enhanced_save_path)
                    logging.info(f"Saved enhanced image to: {enhanced_save_path}\n")
                    # Use the enhanced image for subsequent edits
                    original_image = enhanced_image
                    original_image_bytes = enhanced_image_bytes
            
            current_prompt = self.initial_prompt
            logging.info("\n--- Starting Run ---\n")

            # Store the most recent successful iteration for context if needed
            context_image_bytes = None
            best_prompt = current_prompt

            for i in range(self.iterations):
                logging.info("-" * 30 + "\n")
                logging.info(f">> ITERATION {i + 1}/{self.iterations} <<\n")
                logging.info(f"Prompt:\n{current_prompt}\n")

                # 1. Edit image with current prompt
                if self.enable_editing_context and context_image_bytes:
                    if self.enable_tournament_mode and self.save_best_prompts:
                        # Regenerate context image from best prompt so far
                        _, context_image_bytes = self.image_editing_model.edit(best_prompt, original_image_bytes)
                        logging.info("Regenerating context image from best prompt so far\n")

                    # Use original and previous edited images for context
                    logging.info("Using previous edited image for context during editing\n")
                    edited_prompt = current_prompt + "\nThe first image is the original, the second image is the previous edit."
                    edited_image, edited_image_bytes = self.image_editing_model.edit_with_context(edited_prompt, original_image_bytes, context_image_bytes)
                else:
                    # Use only the original image
                    edited_image, edited_image_bytes = self.image_editing_model.edit(current_prompt, original_image_bytes)

                if edited_image is None or edited_image_bytes is None:
                    logging.error("Image editing failed. Skipping to next iteration.\n")
                    continue

                edited_image_save_path = os.path.join(results_dir, f"{base_filename}_iter_{i+1}.jpg")
                edited_image.save(edited_image_save_path)
                logging.info(f"Saved edited image to: {edited_image_save_path}\n")

                if self.enable_optimization:
                    # 2. Evaluate the edit
                    if context_image_bytes and not self.enable_tournament_mode:
                        # Use original, previous, and current edited images
                        self.evaluator_model.system_prompt = self.context_prompt
                        logging.info("Using context-aware evaluation prompt\n")          

                        evaluation = self.evaluator_model.evaluate_with_context(
                            "Compare the original, previous edited, and current edited images.", 
                            original_image_bytes, 
                            context_image_bytes, 
                            edited_image_bytes
                        )
                    else:
                        # Use only original and current edited images
                        self.evaluator_model.system_prompt = self.no_context_prompt
                        logging.info("Using no-context evaluation prompt\n")

                        evaluation = self.evaluator_model.evaluate(
                            "Compare the original and edited images.",
                            original_image_bytes,
                            edited_image_bytes
                        )

                    if evaluation is None:
                        logging.error("Evaluation failed. Skipping to next iteration.\n")
                        continue

                    logging.info(f"VLM Evaluation:\n{evaluation}\n")

                    # Parse the choice from the evaluation
                    if 'new' in evaluation.split("REASON")[0].lower():
                        vlm_choice = "new"
                    else:
                        vlm_choice = "old"

                    # 3. Get critique (loss)
                    critique = self.loss_model.get_critique(evaluation)

                    if critique is None:
                        logging.error("Critique generation failed. Skipping to next iteration.")
                        continue

                    logging.info(f"Critique (Loss):\n{critique}\n")

                    # 4. Get new prompt from optimizer
                    if self.enable_tournament_mode:
                        # In tournmanent mode, optimize the best prompt so far
                        optimizer_context = (
                            "ORIGINAL PROMPT:\n"
                            f"{best_prompt}\n\n"
                            "CRITIQUE:\n"
                            f"{critique}\n"
                        )
                        logging.info("Optimizing best prompt so far\n")
                    else:
                        # Otherwise optimize the current prompt
                        optimizer_context = (
                            "ORIGINAL PROMPT:\n"
                            f"{current_prompt}\n\n"
                            "CRITIQUE:\n"
                            f"{critique}\n"
                        )
                        logging.info("Optimizing current prompt\n")
                    new_prompt = self.optimizer_model.update_prompt(optimizer_context)

                    if new_prompt is None:
                        logging.error("Prompt optimization failed. Skipping to next iteration.")
                        continue
                    
                    if self.enable_tournament_mode:
                        # In tournament mode, only update context if the new image was preferred
                        if vlm_choice == "new":
                            context_image_bytes = edited_image_bytes
                            best_prompt = current_prompt
                            logging.info("VLM preferred the new image. Updating context for next iteration.\n")
                        else:
                            logging.info("VLM preferred the original image. Retaining previous context for next iteration.\n")
                    else:
                        # Otherwise update context to the latest edit
                        context_image_bytes = edited_image_bytes
                        best_prompt = current_prompt

                    # Update the prompt for the next iteration
                    current_prompt = new_prompt
                    
                    logging.info(f"Optimized Prompt:\n{current_prompt}\n")

            logging.info("-" * 30 + "\n")