import os
import re
import logging
import random
import itertools
import pandas as pd
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse filename to extract class_name, image_id, edit_type.
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

    def _evaluate_comparison(
        self,
        image_class: str,
        image_id_1: str,
        edit_type_1: str,
        img_bytes_1: bytes,
        image_id_2: str,
        edit_type_2: str,
        img_bytes_2: bytes
    ) -> dict:
        """
        Evaluate a single comparison between two images.
        Returns a dictionary with the evaluation result.
        """
        base_1 = f"{image_id_1}_{edit_type_1}"
        base_2 = f"{image_id_2}_{edit_type_2}"

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
            return None

        vlm_choice = evaluation.choice.lower()
        vlm_reason = evaluation.reason

        # Determine which image was chosen by the VLM
        img1_chosen = ((vlm_choice == "first" and is_img1_first) or
                       (vlm_choice == "second" and not is_img1_first))

        choice = base_1 if img1_chosen else base_2
        first = base_1 if is_img1_first else base_2
        second = base_2 if is_img1_first else base_1

        # Extract usage data
        completion_tokens = usage.completion_tokens if usage else 0
        prompt_tokens = usage.prompt_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        reasoning_tokens = 0
        if usage and usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens

        # Return result as dictionary
        return {
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
            'reasoning_tokens': reasoning_tokens,
            # For log file generation
            'log_text': (
                f"Evaluating {base_1} vs {base_2}\n"
                f"Choice: {choice}\n"
                f"Reason (first={first}, second={second}):\n"
                f"{vlm_reason}\n"
                f"{'-' * 40}\n\n"
            )
        }

    def _generate_class_log(self, image_class: str, results: List[dict], results_dir: str):
        """Generate log file for a single image class."""
        all_evaluations = f"Evaluation Results for {image_class}\n"
        all_evaluations += "=" * 50 + "\n\n"
        all_evaluations += f"Found {len(results)} comparable image pairs for {image_class}\n\n"
        all_evaluations += "-" * 40 + "\n\n"

        for result in results:
            if result:
                all_evaluations += result['log_text']

        log_save_path = os.path.join(results_dir, f"{image_class}.log")
        with open(log_save_path, "w", encoding="utf-8") as log_file:
            log_file.write(all_evaluations)

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the evaluation pipeline for each image comparison in parallel.
        """
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

        # Collect all comparison tasks
        comparison_tasks = []
        for image_class in sorted(class_groups.keys()):
            comparable_images = sorted(class_groups[image_class], key=lambda x: '_'.join(x[:2]))

            if len(comparable_images) < 2:
                logging.info(f"Only 1 image found for class {image_class}. Skipping evaluation.\n")
                continue

            for (image_id_1, edit_type_1, img_bytes_1), (image_id_2, edit_type_2, img_bytes_2) in \
                    itertools.combinations(comparable_images, 2):

                # Skip comparison based on settings
                if (not self.allow_same_image_comparison) and (image_id_1 == image_id_2):
                    continue
                if (self.allow_same_image_comparison and
                    self.only_allow_same_image_comparison and (image_id_1 != image_id_2)):
                    continue

                comparison_tasks.append((
                    image_class, image_id_1, edit_type_1, img_bytes_1,
                    image_id_2, edit_type_2, img_bytes_2
                ))

        total_comparisons = len(comparison_tasks)
        logging.info(f"Total comparisons to evaluate: {total_comparisons}\n")

        # Execute comparisons in parallel
        results_data = []
        class_results = defaultdict(list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._evaluate_comparison, *task): task for task in comparison_tasks}

            for future in tqdm(as_completed(futures), total=total_comparisons, desc="Evaluating comparisons", unit="comparison"):
                result = future.result()
                if result:
                    results_data.append(result)
                    class_results[result['image_class']].append(result)
                else:
                    task = futures[future]
                    image_class = task[0]
                    base_1 = f"{task[1]}_{task[2]}"
                    base_2 = f"{task[4]}_{task[5]}"
                    logging.error(f"Evaluation failed for {image_class}: {base_1} vs {base_2}. Skipping.\n")

        # Generate log files for each class
        logging.info("\nGenerating log files...\n")
        for image_class, results in class_results.items():
            self._generate_class_log(image_class, results, results_dir)

        # Export results to CSV
        results_df = pd.DataFrame(results_data, columns=[
            'image_class', 'base1', 'base2', 'choice', 'reason', 'first', 'second',
            'completion_tokens', 'prompt_tokens', 'total_tokens', 'reasoning_tokens'
        ])
        csv_save_path = os.path.join(results_dir, 'results.csv')
        results_df.to_csv(csv_save_path, index=False, encoding='utf-8')
        logging.info(f"Saved results CSV with {len(results_df)} records to: {csv_save_path}\n")
