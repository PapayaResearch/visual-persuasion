import os
import io
import logging
from PIL import Image
from models import ImageEditingModel, EvaluatorModel, LossModel, OptimizerModel

class VisualNudge:
    """
    Orchestrates the visual nudging pipeline.
    Hydra instantiates this class and its model dependencies automatically.
    """
    def __init__(
        self, 
        image_editing_model: ImageEditingModel, 
        evaluator_model: EvaluatorModel, 
        loss_model: LossModel, 
        optimizer_model: OptimizerModel,
        iterations: int
    ):
        self.image_editing_model = image_editing_model
        self.evaluator_model = evaluator_model
        self.loss_model = loss_model
        self.optimizer_model = optimizer_model
        self.iterations = iterations

    def run(self, image_paths: list, results_dir: str, initial_prompt: str):
        """
        Runs the optimization loop for each image.
        """
        current_prompt = initial_prompt

        for img_idx, image_path in enumerate(image_paths):
            base_filename, _ = os.path.splitext(os.path.basename(image_path))
            logging.info(f"===== Processing Image {img_idx + 1}/{len(image_paths)}: {base_filename} =====")
            
            with open(image_path, "rb") as f:
                original_image_bytes = f.read()

            original_image = Image.open(io.BytesIO(original_image_bytes))
            original_save_path = os.path.join(results_dir, f"{base_filename}_original.jpg")
            original_image.save(original_save_path)
            logging.info(f"Saved original image to: {original_save_path}")
            
            logging.info("--- Starting Optimization ---")

            for i in range(self.iterations):
                logging.info(f"\n>> ITERATION {i + 1}/{self.iterations} <<")
                logging.info(f"Current Prompt:\n{current_prompt}")

                # 1. Edit image with current prompt
                edited_image, edited_image_bytes = self.image_editing_model.edit(current_prompt, original_image_bytes)
                edited_image_save_path = os.path.join(results_dir, f"{base_filename}_iter_{i+1}.jpg")
                edited_image.save(edited_image_save_path)
                logging.info(f"Saved edited image to: {edited_image_save_path}")

                # 2. Evaluate the edit
                evaluation = self.evaluator_model.evaluate(original_image_bytes, edited_image_bytes)
                logging.info(f"VLM Evaluation:\n{evaluation}")

                # 3. Get critique (loss)
                loss_context = (
                    "CURRENT_PROMPT:\n"
                    f"{current_prompt}\n\n"
                    "VLM_EVALUATION:\n"
                    f"{evaluation}\n"
                )
                critique = self.loss_model.get_critique(loss_context)
                logging.info(f"Critique (Loss):\n{critique}")

                # 4. Get new prompt from optimizer
                optimizer_context = (
                    "ORIGINAL_PROMPT:\n"
                    f"{current_prompt}\n\n"
                    "CRITIQUE:\n"
                    f"{critique}\n"
                )
                new_prompt = self.optimizer_model.update_prompt(optimizer_context)
                
                # Update the prompt for the next iteration
                current_prompt = new_prompt
                
                logging.info(f"New Optimized Prompt:\n{current_prompt}")
                logging.info("-" * 30)