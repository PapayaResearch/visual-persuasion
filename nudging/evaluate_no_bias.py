import os
import re
import logging
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
        strategy_name: str,
        valid_statuses: List[str],
        n_evaluations: int = 1
        # judge_prompts: List[str]
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.strategy_name = strategy_name
        self.valid_statuses = valid_statuses
        self.n_evaluations = n_evaluations
        # self.judge_prompts = judge_prompts

    def _parse_filename_competition(self, filename: str):
        """
        Parse competition-no-bias filename: CATEGORY_ID_VARIANT_STATUS.jpg or CATEGORY_ID_STATUS.jpg
        Example: SOFA_12345_A_final.jpg -> category=SOFA, image_id=12345_A, status=final
        Example: SOFA_12345_original.jpg -> category=SOFA, image_id=12345, status=original
        Returns (category, image_id, status).
        """
        # Match: CATEGORY_ID_[VARIANT_]STATUS.jpg
        match = re.match(r'^([A-Z]+)_([a-z0-9]+)_(?:([AB])_)?(.+)\.jpg$', filename)

        category = match.group(1)
        image_id = match.group(2)
        variant = match.group(3)  # None if no variant
        status = match.group(4)

        # Only process files with valid statuses
        if status not in self.valid_statuses:
            return None

        # Include variant in full_image_id for unique identification
        full_image_id = f"{image_id}_{variant}" if variant else image_id

        return category, image_id, full_image_id, status

    def _load_completed_comparisons(self, csv_path: str) -> Set[Tuple[str, str, str]]:
        """
        Load completed comparisons from existing CSV.
        Returns a set of (image_class, base1, base2) tuples.
        """
        if not os.path.exists(csv_path):
            return set()

        completed = set()
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            completed.add((row['image_class'], row['base1'], row['base2']))

        logging.info(f"Loaded {len(completed)} completed comparisons from existing CSV\n")
        return completed


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
        Evaluate a single comparison between two images using multiple judges.
        Returns a dictionary with the evaluation result based on majority vote.
        """
        base_1 = f"{image_id_1}_{edit_type_1}"
        base_2 = f"{image_id_2}_{edit_type_2}"

        def evaluate_single(is_1_first: bool):
            """Single judge evaluation"""
            images = [img_bytes_1, img_bytes_2] if is_1_first else [img_bytes_2, img_bytes_1]
            choice_map = {
                "first": base_1 if is_1_first else base_2,
                "second": base_2 if is_1_first else base_1
            }

            logging.info(f"Evaluating with {base_1} as {'first' if is_1_first else 'second'} image.\n")

            evaluation, usage = self.evaluator_model.get_response(
                images=images,
                metadata="The product here is a(n) %s." % image_class
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            logging.info(f"Judge chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason, usage)


        choice_1_first, reason_1_first, usage_1_first = evaluate_single(is_1_first=True)
        choice_2_first, reason_2_first, usage_2_first = evaluate_single(is_1_first=False)

        choice = None
        if choice_1_first == choice_2_first:
            choice = choice_1_first
            vlm_reason = reason_1_first if choice_1_first == base_1 else reason_2_first

        else:
            logging.warning(f"Inconsistent judge results for {base_1} vs {base_2}: {choice_1_first} vs {choice_2_first}\n")
            choice = "inconsistent"
            vlm_reason = " ".join([f"Judge 1: {reason_1_first}", f"Judge 2: {reason_2_first}"])

        completion_tokens = usage_1_first.completion_tokens + usage_2_first.completion_tokens
        prompt_tokens = usage_1_first.prompt_tokens + usage_2_first.prompt_tokens
        total_tokens = usage_1_first.total_tokens + usage_2_first.total_tokens

        reasoning_tokens = 0
        if usage_1_first.completion_tokens_details and usage_2_first.completion_tokens_details:
            reasoning_tokens = usage_1_first.completion_tokens_details.reasoning_tokens + usage_2_first.completion_tokens_details.reasoning_tokens

        return {
            'image_class': image_class,
            'base1': base_1,
            'base2': base_2,
            'choice': choice,
            'reason': vlm_reason,
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

        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(img_path)

            parsed = self._parse_filename_competition(filename)
            if parsed:
                category, base_id, full_id, status = parsed
                class_groups[category].add((base_id, full_id, status, img_bytes))

        logging.info(f"Found {len(class_groups)} image classes to evaluate\n")

        # Collect all comparison tasks, skipping completed ones
        comparison_tasks = []
        for image_class in sorted(class_groups.keys()):
            comparable_images = sorted(class_groups[image_class], key=lambda x: '_'.join(x[:2]))

            for (base_id_1, full_id_1, edit_type_1, img_bytes_1), (base_id_2, full_id_2, edit_type_2, img_bytes_2) in \
                    itertools.combinations(comparable_images, 2):

                # Skip same base image comparisons
                if base_id_1 == base_id_2:
                    continue

                # Check if this comparison was already completed
                base_1 = f"{full_id_1}_{edit_type_1}"
                base_2 = f"{full_id_2}_{edit_type_2}"
                comparison_key = (image_class, base_1, base_2)

                if comparison_key in completed_comparisons:
                    continue

                comparison_tasks.append((
                    image_class, full_id_1, edit_type_1, img_bytes_1,
                    full_id_2, edit_type_2, img_bytes_2
                ))

        comparison_tasks = list(comparison_tasks) * self.n_evaluations  # Repeat tasks for multiple evaluations if >1 specified
        total_comparisons = len(comparison_tasks)
        logging.info(f"Total comparisons to evaluate: {total_comparisons}\n")

        # Prepare CSV file for incremental writing
        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        # Open CSV in append mode for incremental writing
        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_class', 'base1', 'base2', 'choice', 'reason', 'first', 'second',
                'completion_tokens', 'prompt_tokens', 'total_tokens', 'reasoning_tokens',
                'consistent_judges', 'winner_score'
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
