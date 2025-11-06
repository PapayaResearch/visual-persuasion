import os
import re
import logging
import random
import itertools
import threading
import pandas as pd
from typing import List, Tuple, Set
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
        evaluator_model: LanguageModel,
        only_allow_first_last_comparison: bool
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
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

    def _load_completed_comparisons(self, csv_path: str) -> Set[Tuple[str, str, str]]:
        """
        Load completed comparisons from existing CSV.
        Returns a set of (image_class, base1, base2) tuples with normalized ordering.
        """
        if not os.path.exists(csv_path):
            return set()

        completed = set()
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Normalize order so (A, B) and (B, A) are treated as the same
            pair = tuple(sorted([row['base1'], row['base2']]))
            completed.add((row['image_class'], pair[0], pair[1]))

        logging.info(f"Loaded {len(completed)} completed comparisons from existing CSV\n")
        return completed

    def _get_images_to_evaluate(self, image_paths: List[str]) -> List[str]:
        """
        Retrieves all image files to be evaluated based on naming conventions.
        """
        if not self.only_allow_first_last_comparison:
            return image_paths

        image_groups = defaultdict(list)
        for img_path in image_paths:
            if not img_path.lower().endswith('.jpg'):
                continue
            _, image_id, edit_type = self._parse_filename(os.path.basename(img_path))
            iter_num = int(re.search(r'iter-(\d+)', edit_type).group(1))
            image_groups[image_id].append((iter_num, img_path))

        images_to_evaluate = []
        for image_id, paths in image_groups.items():
            paths.sort()
            images_to_evaluate.append(paths[0][1])   # first
            images_to_evaluate.append(paths[-1][1])  # last

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
            'reasoning_tokens': reasoning_tokens
        }

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the evaluation pipeline for each image comparison in parallel.
        Supports resumption by skipping already completed comparisons.
        """
        csv_save_path = os.path.join(results_dir, 'results.csv')

        # Load completed comparisons from existing CSV
        completed_comparisons = self._load_completed_comparisons(csv_save_path)

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

        # Collect all comparison tasks, skipping completed ones
        comparison_tasks = []
        for image_class in sorted(class_groups.keys()):
            comparable_images = sorted(class_groups[image_class], key=lambda x: '_'.join(x[:2]))

            for (image_id_1, edit_type_1, img_bytes_1), (image_id_2, edit_type_2, img_bytes_2) in \
                    itertools.combinations(comparable_images, 2):

                # Skip same image comparisons
                if image_id_1 == image_id_2:
                    continue

                # Check if this comparison was already completed
                base_1 = f"{image_id_1}_{edit_type_1}"
                base_2 = f"{image_id_2}_{edit_type_2}"
                comparison_key = (image_class, *tuple(sorted([base_1, base_2])))

                if comparison_key in completed_comparisons:
                    continue

                comparison_tasks.append((
                    image_class, image_id_1, edit_type_1, img_bytes_1,
                    image_id_2, edit_type_2, img_bytes_2
                ))

        total_comparisons = len(comparison_tasks)
        logging.info(f"Total comparisons to evaluate: {total_comparisons}\n")

        # Prepare CSV file for incremental writing
        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        # Open CSV in append mode for incremental writing
        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_class', 'base1', 'base2', 'choice', 'reason', 'first', 'second',
                'completion_tokens', 'prompt_tokens', 'total_tokens', 'reasoning_tokens'
            ]
            writer = pd.DataFrame(columns=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.to_csv(csvfile, index=False, header=True, mode='a')
                csvfile.flush()

            # Execute comparisons in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_comparison,
                        *task
                    ): task for task in comparison_tasks
                }

                for future in tqdm(
                        as_completed(futures),
                        total=total_comparisons,
                        desc="Evaluating comparisons",
                        unit="comparison"
                ):
                    result = future.result()

                    # Write result immediately to CSV with thread safety
                    with csv_lock:
                        result_df = pd.DataFrame([result])
                        result_df.to_csv(csvfile, index=False, header=False, mode='a')
                        csvfile.flush()  # Force write to disk

        logging.info(f"Evaluation completed. Results saved to: {csv_save_path}\n")
