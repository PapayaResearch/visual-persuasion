import os
import logging
import random
import itertools
from typing import List
from collections import defaultdict
from utils.wrappers import LanguageModel

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

    def run(self, image_paths: List[str], results_dir: str):
        """
        Runs the evaluation pipeline for each image.
        """
        # Group images by class
        class_groups = defaultdict(list)
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            split = name_without_ext.split('_')
            image_class = '_'.join(split[:-1])
            base = split[-1]
            class_groups[image_class].append((img_path, base))

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
            for (img1_path, base1), (img2_path, base2) in itertools.combinations(comparable_images, 2):
                logging.info(f"\n--- Evaluating {base1} vs {base2} ---\n")
                all_evaluations += f"Evaluating {base1} vs {base2}\n"

                with open(img1_path, "rb") as f:
                    img1_bytes = f.read()
                with open(img2_path, "rb") as f:
                    img2_bytes = f.read()

                # Randomly decide which image is first and which is second to avoid bias
                is_img1_first = random.choice([True, False])

                image1_bytes = img1_bytes if is_img1_first else img2_bytes
                image2_bytes = img2_bytes if is_img1_first else img1_bytes

                # Evaluate the images without telling the VLM which is which
                evaluation, usage = self.evaluator_model.get_response(
                    task=self.evaluator_prompt,
                    images=[image1_bytes, image2_bytes]
                )

                if not evaluation:
                    logging.error("Evaluation failed. Skipping to next comparison.\n")
                    all_evaluations += f"ERROR: Evaluation failed for {base1} vs {base2}. Skipping.\n\n"
                    continue

                vlm_choice = evaluation.choice.lower()
                vlm_reason = evaluation.reason
                # Determine which image was chosen by the VLM
                img1_chosen = ((vlm_choice == "first" and is_img1_first) or
                               (vlm_choice == "second" and not is_img1_first))

                result = (
                    f"Choice: {base1 if img1_chosen else base2}\n"
                    f"Reason (first={base1 if is_img1_first else base2}, "
                    f"second={base2 if is_img1_first else base1}):\n"
                    f"{vlm_reason}\n"
                )
                logging.info(result)

                # Append this result to our collection for this class
                all_evaluations += result + "\n" + "-" * 40 + "\n\n"

                # Log usage data
                completion_tokens = usage.completion_tokens
                prompt_tokens = usage.prompt_tokens
                total_tokens = usage.total_tokens
                if not usage.completion_tokens_details:
                    reasoning_tokens = 0
                else:
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens

            # After processing all comparisons, save the combined results to a single log file
            log_save_path = os.path.join(results_dir, f"{image_class}.log")
            with open(log_save_path, "w", encoding="utf-8") as log_file:
                log_file.write(all_evaluations)
            logging.info(f"Saved all evaluation results to: {log_save_path}\n")