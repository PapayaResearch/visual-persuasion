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
        enhance_original: bool,
        iterations: int,
        initial_prompt: str,
        enhance_prompt: str,
        image_editing_model: ImageEditingModel, 
        evaluator_model: EvaluatorModel, 
        loss_model: LossModel, 
        optimizer_model: OptimizerModel,
    ):
        self.enable_optimization = enable_optimization
        self.enhance_original = enhance_original
        self.iterations = iterations
        self.initial_prompt = initial_prompt
        self.enhance_prompt = enhance_prompt
        self.image_editing_model = image_editing_model
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
            logging.info(f"Saved original image to: {original_save_path}")

            # Optionally enhance the original image before starting
            if self.enhance_original:
                logging.info("\n--- Enhancing Original Image ---\n")
                enhanced_image, enhanced_image_bytes = self.image_editing_model.edit(self.enhance_prompt, original_image_bytes)
                
                if enhanced_image is None or enhanced_image_bytes is None:
                    logging.error("Enhancement failed. Proceeding with the original image.")
                else:
                    enhanced_save_path = os.path.join(results_dir, f"{base_filename}_enhanced.jpg")
                    enhanced_image.save(enhanced_save_path)
                    logging.info(f"Saved enhanced image to: {enhanced_save_path}")
                    # Use the enhanced image for subsequent edits
                    original_image = enhanced_image
                    original_image_bytes = enhanced_image_bytes
            
            current_prompt = self.initial_prompt
            logging.info("\n--- Starting Run ---\n")

            for i in range(self.iterations):
                logging.info(f"\n>> ITERATION {i + 1}/{self.iterations} <<\n")
                logging.info(f"Prompt:\n{current_prompt}")

                # 1. Edit image with current prompt
                edited_image, edited_image_bytes = self.image_editing_model.edit(current_prompt, original_image_bytes)

                if edited_image is None or edited_image_bytes is None:
                    logging.error("Image editing failed. Skipping to next iteration.")
                    continue

                edited_image_save_path = os.path.join(results_dir, f"{base_filename}_iter_{i+1}.jpg")
                edited_image.save(edited_image_save_path)
                logging.info(f"Saved edited image to: {edited_image_save_path}")

                if self.enable_optimization:
                    # 2. Evaluate the edit
                    evaluation = self.evaluator_model.evaluate(original_image_bytes, edited_image_bytes)
                    logging.info(f"\nVLM Evaluation:\n{evaluation}\n")

                    # 3. Get critique (loss)
                    loss_context = (
                        "CURRENT PROMPT:\n"
                        f"{current_prompt}\n\n"
                        "VLM EVALUATION:\n"
                        f"{evaluation}\n"
                    )
                    critique = self.loss_model.get_critique(loss_context)
                    logging.info(f"\nCritique (Loss):\n{critique}\n")

                    # 4. Get new prompt from optimizer
                    optimizer_context = (
                        "ORIGINAL PROMPT:\n"
                        f"{current_prompt}\n\n"
                        "CRITIQUE:\n"
                        f"{critique}\n"
                    )
                    new_prompt = self.optimizer_model.update_prompt(optimizer_context)
                    
                    # Update the prompt for the next iteration
                    current_prompt = new_prompt
                    
                    logging.info(f"\nOptimized Prompt:\n{current_prompt}\n")

                logging.info("-" * 30)