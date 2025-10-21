import os
import logging
import random
import re
from shared.wrappers import LanguageModel

class EvaluationPipeline:
    """
    Evaluation pipeline to assess the visual nudges.
    """
    def __init__(
        self, 
        num_images: int,
        evaluator_prompt: str,
        evaluator_model: LanguageModel
    ):
        self.num_images = num_images
        self.evaluator_prompt = evaluator_prompt
        self.evaluator_model = evaluator_model
    
    def run(self, image_dir: str, model: str):
        """
        Runs the evaluation pipeline for each image.
        """
        # Create results directory if it doesn't exist
        results_dir = os.path.join(image_dir, "evaluation", model)
        os.makedirs(results_dir, exist_ok=True)

        # Get all image files in the directory
        all_files = [f for f in os.listdir(image_dir) if (os.path.isfile(os.path.join(image_dir, f))
                                                            and f.lower().endswith(('.jpg', '.jpeg', '.png')))]
        
        # Get all the original images
        original_images = [f for f in all_files if "_original" in f]
        
        if not original_images:
            logging.error(f"No original images found in directory: {image_dir}\n")
            return
        
        logging.info(f"Found {len(original_images)} original images to evaluate\n")
        
        # Limit the number of images to evaluate if specified
        if self.num_images > 0:
            original_images = original_images[:self.num_images]
        
        for original_image_name in original_images:
            base_name = original_image_name.replace("_original.jpg", "")
            logging.info(f"\n===== Evaluating Image: {base_name} =====\n")
            
            # Initialize a string to collect all evaluation results for this base image
            all_evaluations = f"Evaluation Results for {base_name}\n"
            all_evaluations += "=" * 50 + "\n\n"
            
            original_image_path = os.path.join(image_dir, original_image_name)
            with open(original_image_path, "rb") as f:
                original_image_bytes = f.read()
            
            # Find all iteration images for this base image
            iter_images = [f for f in all_files if f.startswith(base_name) and "_iter_" in f and "context" not in f and "best" not in f]
            
            if not iter_images:
                logging.info(f"No iteration images found for {base_name}\n")
                all_evaluations += f"No iteration images found for {base_name}. Skipping evaluation.\n\n"
                continue
            
            logging.info(f"Found {len(iter_images)} iteration images for {base_name}\n")
            all_evaluations += f"Found {len(iter_images)} iteration images for {base_name}\n\n"
            all_evaluations += "-" * 40 + "\n\n"
            
            # Sort iteration images by iteration number
            iter_images.sort(key=lambda x: int(re.search(r'_iter_(\d+)', x).group(1)))
            
            for iter_image_name in iter_images:
                iter_num = re.search(r'_iter_(\d+)', iter_image_name).group(1)
                logging.info(f"\n--- Evaluating original vs iteration {iter_num} ---\n")
                all_evaluations += f"Evaluating original vs iteration {iter_num}\n"
                
                iter_image_path = os.path.join(image_dir, iter_image_name)
                with open(iter_image_path, "rb") as f:
                    iter_image_bytes = f.read()
                
                # Randomly decide which image is first and which is second to avoid bias
                is_original_first = random.choice([True, False])
                
                image1_bytes = original_image_bytes if is_original_first else iter_image_bytes
                image2_bytes = iter_image_bytes if is_original_first else original_image_bytes
                
                # Evaluate the images without telling the VLM which is which
                evaluation = self.evaluator_model.get_response(
                    task=self.evaluator_prompt,
                    images=[image1_bytes, image2_bytes]
                )

                if evaluation is None:
                    logging.error("Evaluation failed. Skipping to next image.\n")
                    all_evaluations += f"ERROR: Evaluation failed for {iter_image_name}. Skipping.\n\n"
                    continue
                
                vlm_choice = evaluation.choice.lower()
                vlm_reason = evaluation.reason
                
                if vlm_choice:
                    # Determine which image was chosen by the VLM
                    original_chosen = ((vlm_choice == "first" and is_original_first) or
                        (vlm_choice == "second" and not is_original_first))
                    
                    choice_text = "original" if original_chosen else "edited"
                    reason_text = vlm_reason
                    
                    result = f"VLM Choice: {choice_text}\n"
                    result += f"Reason (first: {'original' if is_original_first else 'edited'}, second: {'edited' if is_original_first else 'original'}):\n{reason_text}\n"
                    
                    logging.info(result)
                    
                    # Append this result to our collection for this base image
                    all_evaluations += result + "\n" + "-" * 40 + "\n\n"
                else:
                    error_msg = f"Could not parse choice from VLM response for {iter_image_name}\nResponse was:\n{evaluation}\n"
                    logging.error(error_msg)
                    all_evaluations += f"ERROR: {error_msg}\n\n"
            
            # After processing all iterations, save the combined results to a single log file
            log_save_path = os.path.join(results_dir, f"{base_name}.log")
            with open(log_save_path, "w") as log_file:
                log_file.write(all_evaluations)
            logging.info(f"Saved all evaluation results to: {log_save_path}\n")
        
        return results_dir