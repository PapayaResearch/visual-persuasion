import os
import re
import logging
import random
import threading
import pandas as pd
from typing import List
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import LanguageModel

class EvaluationPipeline:
    """
    Chain evaluation pipeline to assess optimization progression.
    For each product, compares: original vs zero-shot, original vs final, zero-shot vs final.
    """
    def __init__(
        self,
        evaluator_model: LanguageModel,
        strategy_name: str,
        n_evaluations: int = 1,
        max_comparisons: int = -1,
        sampling_seed: int = 42
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.strategy_name = strategy_name
        self.n_evaluations = n_evaluations
        self.max_comparisons = max_comparisons
        self.sampling_seed = sampling_seed

    def _parse_filename_competition(self, filename: str):
        """
        Parse competition-no-bias filename: CATEGORY_ID_VARIANT_STATUS.jpg or CATEGORY_ID_STATUS.jpg
        Example: SOFA_12345_A_final.jpg -> category=SOFA, image_id=12345, status=final
        Example: SOFA_12345_original.jpg -> category=SOFA, image_id=12345, status=original
        Returns (category, image_id, status) or None.
        Only accepts: original, zero-shot, final
        """
        # Match: CATEGORY_ID_[VARIANT_]STATUS.jpg
        match = re.match(r'^([A-Z]+)_([a-z0-9]+)_(?:([AB])_)?(.+)\.jpg$', filename)

        if not match:
            return None

        category = match.group(1)
        image_id = match.group(2)
        variant = match.group(3)  # None if no variant
        status = match.group(4)

        # Only process chain-relevant files
        if status not in ['original', 'zero-shot', 'final']:
            return None

        return category, image_id, status

    def _evaluate_comparison(
        self,
        base: str,
        category: str,
        first_status: str,
        second_status: str,
        img_bytes_first: bytes,
        img_bytes_second: bytes
    ) -> dict:
        """
        Evaluate a single comparison between two stages of the same product.
        Returns a dictionary with the evaluation result.
        """
        def evaluate_single(is_first_order: bool):
            """Single judge evaluation"""
            images = [img_bytes_first, img_bytes_second] if is_first_order else [img_bytes_second, img_bytes_first]
            choice_map = {
                "first": "first" if is_first_order else "second",
                "second": "second" if is_first_order else "first"
            }

            logging.info(f"Evaluating {base}: {first_status} vs {second_status} ({'first-second order' if is_first_order else 'second-first order'})\n")

            evaluation, usage = self.evaluator_model.get_response(
                images=images,
                metadata="The product here is a(n) %s." % category
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            logging.info(f"Judge chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason, usage)

        choice_1_first, reason_1_first, usage_1_first = evaluate_single(is_first_order=True)
        choice_2_first, reason_2_first, usage_2_first = evaluate_single(is_first_order=False)

        if choice_1_first == choice_2_first:
            choice = choice_1_first
            vlm_reason = reason_1_first
        else:
            logging.warning(f"Inconsistent judge results for {base} {first_status} vs {second_status}: {choice_1_first} vs {choice_2_first}\n")
            choice = "inconsistent"
            vlm_reason = f"Judge 1: {reason_1_first} | Judge 2: {reason_2_first}"

        completion_tokens = usage_1_first.completion_tokens + usage_2_first.completion_tokens
        prompt_tokens = usage_1_first.prompt_tokens + usage_2_first.prompt_tokens
        total_tokens = usage_1_first.total_tokens + usage_2_first.total_tokens

        reasoning_tokens = 0
        if usage_1_first.completion_tokens_details and usage_2_first.completion_tokens_details:
            reasoning_tokens = usage_1_first.completion_tokens_details.reasoning_tokens + usage_2_first.completion_tokens_details.reasoning_tokens

        return {
            'image_class': category,
            'base': base,
            'first': first_status,
            'second': second_status,
            'choice': choice,
            'reason': vlm_reason,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs chain evaluation: for each product, compares original vs zero-shot, original vs final, zero-shot vs final.
        """
        csv_save_path = os.path.join(results_dir, 'chain_results.csv')

        # Group images by (category, base_id) to form chains
        product_images = defaultdict(dict)  # key: (category, base_id), value: {status: img_bytes}

        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(img_path)

            parsed = self._parse_filename_competition(filename)
            if parsed:
                category, base_id, status = parsed
                product_key = (category, base_id)
                product_images[product_key][status] = img_bytes

        logging.info(f"Found {len(product_images)} products to evaluate\n")

        # Collect comparison tasks
        comparison_tasks = []
        for (category, base_id), status_dict in sorted(product_images.items()):
            base = f"{category}_{base_id}"

            # Get the three images if they exist
            original = status_dict.get('original')
            zero_shot = status_dict.get('zero-shot')
            final = status_dict.get('final')

            # Create the 3 comparisons
            if original and zero_shot:
                comparison_tasks.append((
                    base, category, 'original', 'zero-shot', original, zero_shot
                ))

            if original and final:
                comparison_tasks.append((
                    base, category, 'original', 'final', original, final
                ))

            if zero_shot and final:
                comparison_tasks.append((
                    base, category, 'zero-shot', 'final', zero_shot, final
                ))

        # Sample comparisons if max_comparisons is set and we have more tasks
        if self.max_comparisons > 0 and len(comparison_tasks) > self.max_comparisons:
            logging.info(f"Sampling {self.max_comparisons} comparisons from {len(comparison_tasks)} total (balanced by comparison type, seed={self.sampling_seed})\n")

            # Seed the random number generator for reproducibility
            random.seed(self.sampling_seed)

            # Group by comparison type
            comparison_groups = defaultdict(list)
            for task in comparison_tasks:
                first_status = task[2]
                second_status = task[3]
                comp_type = tuple(sorted([first_status, second_status]))
                comparison_groups[comp_type].append(task)

            # Sample equally from each group
            num_groups = len(comparison_groups)
            per_group = self.max_comparisons // num_groups

            sampled_tasks = []
            for comp_type, tasks in comparison_groups.items():
                sample_size = min(per_group, len(tasks))
                sampled = random.sample(tasks, sample_size)
                sampled_tasks.extend(sampled)
                logging.info(f"  {comp_type}: sampled {sample_size}/{len(tasks)}")

            comparison_tasks = sampled_tasks

        comparison_tasks = list(comparison_tasks) * self.n_evaluations
        total_comparisons = len(comparison_tasks)
        logging.info(f"Total comparisons to evaluate: {total_comparisons}\n")

        # Prepare CSV file for incremental writing
        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        # Open CSV in append mode for incremental writing
        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_class', 'base', 'first', 'second', 'choice', 'reason',
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
                        desc="Evaluating chain comparisons",
                        unit="comparison"
                ):
                    result = future.result()

                    # Write result immediately to CSV with thread safety
                    with csv_lock:
                        result_df = pd.DataFrame([result])
                        result_df.to_csv(csvfile, index=False, header=False, mode='a')
                        csvfile.flush()  # Force write to disk

        logging.info(f"Chain evaluation completed. Results saved to: {csv_save_path}\n")
