import os
import logging
import random
import itertools
from collections import defaultdict
from shared.wrappers import LanguageModel

class EvaluationPipeline:
    """
    Evaluation pipeline to assess the visual nudges.
    """
    def __init__(
        self, 
        evaluator_prompt: str,
        evaluator_model: LanguageModel
    ):
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
        
        # Sort all files for consistent ordering
        all_files.sort()
        
        # Group images by class (foldername_filename)
        class_groups = defaultdict(list)
        for filename in all_files:
            name_without_ext = os.path.splitext(filename)[0]
            split = name_without_ext.split('_')
            image_class = '_'.join(split[:-1])
            base = split[-1]
            class_groups[image_class].append((filename, base))
        
        if not class_groups:
            logging.error(f"No images found in directory: {image_dir}\n")
            return
        
        logging.info(f"Found {len(class_groups)} image classes to evaluate\n")
        
        class_list = sorted(class_groups.keys())
        
        for image_class in class_list:
            comparable_images = sorted(class_groups[image_class], key=lambda x: x[1])
            
            if len(comparable_images) < 2:
                logging.info(f"Only 1 image found for class {image_class}. Skipping evaluation.\n")
                continue
            
            logging.info(f"\n===== Evaluating Image Class: {image_class} =====\n")
            
            # Initialize a string to collect all evaluation results for this class
            all_evaluations = f"Evaluation Results for {image_class}\n"
            all_evaluations += "=" * 50 + "\n\n"
            
            logging.info(f"Found {len(comparable_images)} comparable images for {image_class}\n")
            all_evaluations += f"Found {len(comparable_images)} comparable images for {image_class}\n\n"
            all_evaluations += "-" * 40 + "\n\n"
            
            # Perform round-robin comparison
            for (img1_name, base1), (img2_name, base2) in itertools.combinations(comparable_images, 2):
                logging.info(f"\n--- Evaluating {base1} vs {base2} ---\n")
                all_evaluations += f"Evaluating {base1} vs {base2}\n"
                
                img1_path = os.path.join(image_dir, img1_name)
                img2_path = os.path.join(image_dir, img2_name)
                
                with open(img1_path, "rb") as f:
                    img1_bytes = f.read()
                with open(img2_path, "rb") as f:
                    img2_bytes = f.read()
                
                # Randomly decide which image is first and which is second to avoid bias
                is_img1_first = random.choice([True, False])
                
                image1_bytes = img1_bytes if is_img1_first else img2_bytes
                image2_bytes = img2_bytes if is_img1_first else img1_bytes
                
                # Evaluate the images without telling the VLM which is which
                evaluation = self.evaluator_model.get_response(
                    task=self.evaluator_prompt,
                    images=[image1_bytes, image2_bytes]
                )

                if evaluation is None:
                    logging.error("Evaluation failed. Skipping to next comparison.\n")
                    all_evaluations += f"ERROR: Evaluation failed for {base1} vs {base2}. Skipping.\n\n"
                    continue
                
                vlm_choice = evaluation.choice.lower()
                vlm_reason = evaluation.reason
                
                if vlm_choice:
                    # Determine which image was chosen by the VLM
                    img1_chosen = ((vlm_choice == "first" and is_img1_first) or
                        (vlm_choice == "second" and not is_img1_first))
                    
                    choice_text = base1 if img1_chosen else base2
                    reason_text = vlm_reason
                    
                    result = f"VLM Choice: {choice_text}\n"
                    result += f"Reason (first: {base1 if is_img1_first else base2}, second: {base2 if is_img1_first else base1}):\n{reason_text}\n"
                    
                    logging.info(result)
                    
                    # Append this result to our collection for this class
                    all_evaluations += result + "\n" + "-" * 40 + "\n\n"
                else:
                    error_msg = f"Could not parse choice from VLM response for {base1} vs {base2}\nResponse was:\n{evaluation}\n"
                    logging.error(error_msg)
                    all_evaluations += f"ERROR: {error_msg}\n\n"
            
            # After processing all comparisons, save the combined results to a single log file
            log_save_path = os.path.join(results_dir, f"{image_class}.log")
            with open(log_save_path, "w") as log_file:
                log_file.write(all_evaluations)
            logging.info(f"Saved all evaluation results to: {log_save_path}\n")
        
        return results_dir