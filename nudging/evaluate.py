import os
import re
import logging
import random
import itertools
import pandas as pd
from typing import List, Tuple
from collections import defaultdict
from utils.wrappers import LanguageModel

class EvaluationPipeline:
    """
    Evaluation pipeline to assess the visual nudges.
    """
    def __init__(
        self,
        evaluator_prompt: str,
        evaluator_model: LanguageModel,
        allow_same_image_comparison: bool,
        only_allow_same_image_comparison: bool,
        only_allow_first_last_comparison: bool
    ):
        self.evaluator_prompt = evaluator_prompt
        self.evaluator_model = evaluator_model
        self.allow_same_image_comparison = allow_same_image_comparison
        self.only_allow_same_image_comparison = only_allow_same_image_comparison
        self.only_allow_first_last_comparison = only_allow_first_last_comparison

    def _parse_filename(self, filename: str) -> Tuple[str, str, str, int]:
        """
        Parse filename to extract class_name, image_id, edit_type, and prior_index.
        """
        match = re.match(r'([A-Za-z0-9_]+)_([A-Za-z0-9]+)_([A-Za-z0-9-]+)\.jpg', filename)

        class_name = match.group(1)
        image_id = match.group(2)
        edit_type = match.group(3)

        return class_name, image_id, edit_type

    def _get_images_to_evaluate(self, image_paths: List[str]) -> List[str]:
        """
        Retrieves all image files to be evaluated based on naming conventions.
        """
        if not self.only_allow_first_last_comparison:
            return image_paths

        first_iteration, last_iteration = {}, {}
        for img_path in image_paths:
            if not img_path.lower().endswith('.jpg'):
                continue
            _, image_id, _ = self._parse_filename(os.path.basename(img_path))
            if (image_id not in first_iteration) or (img_path < first_iteration[image_id]):
                first_iteration[image_id] = img_path
            if (image_id not in last_iteration) or (img_path > last_iteration[image_id]):
                last_iteration[image_id] = img_path

        images_to_evaluate = list(first_iteration.values()) + list(last_iteration.values())

        return images_to_evaluate

    def run(self, image_paths: List[str], results_dir: str):
        """
        Runs the evaluation pipeline for each image.
        """
        # Initialize DataFrame to collect all results
        results_data = []

        # Group images by class
        class_groups = defaultdict(set)
        images_to_evaluate = self._get_images_to_evaluate(image_paths)
        for img_path in images_to_evaluate:
            with open(img_path, "rb") as f:
                    img_bytes = f.read()
            filename = os.path.basename(img_path)
            class_name, image_id, edit_type = self._parse_filename(filename)
            class_groups[class_name].add((image_id, edit_type, img_bytes))

        logging.info(f"Found {len(class_groups)} image classes to evaluate\n")

        class_list = sorted(class_groups.keys())

        for image_class in class_list:
            comparable_images = sorted(class_groups[image_class], key=lambda x: '_'.join(x[:2]))

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
            for (image_id_1, edit_type_1, img_bytes_1), (image_id_2, edit_type_2, img_bytes_2)\
                in itertools.combinations(comparable_images, 2):
                # Skip comparison if comparisons between same image IDs are not allowed
                if (not self.allow_same_image_comparison) and (image_id_1 == image_id_2):
                    continue
                # Skip comparison if only comparisons between same image IDs are allowed
                if (self.allow_same_image_comparison and
                    self.only_allow_same_image_comparison and (image_id_1 != image_id_2)):
                    continue

                base_1 = f"{image_id_1}_{edit_type_1}"
                base_2 = f"{image_id_2}_{edit_type_2}"

                logging.info(f"\n--- Evaluating {base_1} vs {base_2} ---\n")
                all_evaluations += f"Evaluating {base_1} vs {base_2}\n"

                # Randomly decide which image is first and which is second to avoid bias
                is_img1_first = random.choice([True, False])

                image1_bytes = img_bytes_1 if is_img1_first else img_bytes_2
                image2_bytes = img_bytes_2 if is_img1_first else img_bytes_1

                # Evaluate the images without telling the VLM which is which
                evaluation, usage = self.evaluator_model.get_response(
                    task=self.evaluator_prompt,
                    images=[image1_bytes, image2_bytes]
                )

                if not evaluation:
                    logging.error("Evaluation failed. Skipping to next comparison.\n")
                    all_evaluations += f"ERROR: Evaluation failed for {base_1} vs {base_2}. Skipping.\n\n"
                    continue

                vlm_choice = evaluation.choice.lower()
                vlm_reason = evaluation.reason
                # Determine which image was chosen by the VLM
                img1_chosen = ((vlm_choice == "first" and is_img1_first) or
                               (vlm_choice == "second" and not is_img1_first))

                choice = base_1 if img1_chosen else base_2
                first = base_1 if is_img1_first else base_2
                second = base_2 if is_img1_first else base_1

                result = (
                    f"Choice: {choice}\n"
                    f"Reason (first={first}, second={second}):\n"
                    f"{vlm_reason}\n"
                )
                logging.info(result)

                # Append this result to our collection for this class
                all_evaluations += result + "\n" + "-" * 40 + "\n\n"

                # Extract usage data
                completion_tokens = usage.completion_tokens
                prompt_tokens = usage.prompt_tokens
                total_tokens = usage.total_tokens
                reasoning_tokens = 0
                if usage.completion_tokens_details:
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens

                # Add record to results data
                results_data.append({
                    'image_class': image_class,
                    'base1': base_1,
                    'base2': base_2,
                    'choice': choice,
                    'reason': vlm_reason,
                    'first': first,
                    'second': second,
                    'completion_tokens': completion_tokens,
                    'prompt_tokens': prompt_tokens,
                    'total_tokens': total_tokens,
                    'reasoning_tokens': reasoning_tokens
                })

            # After processing all comparisons, save the combined results to a single log file
            log_save_path = os.path.join(results_dir, f"{image_class}.log")
            with open(log_save_path, "w", encoding="utf-8") as log_file:
                log_file.write(all_evaluations)
            logging.info(f"Saved all evaluation results to: {log_save_path}\n")

        # Export results to CSV
        results_df = pd.DataFrame(results_data, columns=[
            'image_class', 'base1', 'base2', 'choice', 'reason', 'first', 'second',
            'completion_tokens', 'prompt_tokens', 'total_tokens', 'reasoning_tokens'
        ])
        csv_save_path = os.path.join(results_dir, 'results.csv')
        results_df.to_csv(csv_save_path, index=False, encoding='utf-8')
        logging.info(f"Saved results CSV with {len(results_df)} records to: {csv_save_path}\n")