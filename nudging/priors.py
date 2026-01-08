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

class PriorsPipeline:
    """
    Pipeline to assess default model preferences between product images.
    Compares all pairs of products within a category to determine baseline biases.
    """
    def __init__(
        self,
        evaluator_model: LanguageModel,
        judge_prompts: List[str],
        category_pattern: str = r'^([^_]+)_'
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.judge_prompts = judge_prompts
        self.category_pattern = category_pattern

    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse filename to extract category and image identifier.
        Default pattern: CATEGORY_ID.jpg
        Returns (category, image_id) or (None, None) if parsing fails.
        """
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return None, None

        # Extract category using the pattern
        match = re.match(self.category_pattern, filename)
        if not match:
            logging.warning(f"Could not parse category from filename: {filename}")
            return None, None

        category = match.group(1)
        # Use the full filename (without extension) as the image_id
        image_id = os.path.splitext(filename)[0]

        return category, image_id

    def _load_completed_comparisons(self, csv_path: str) -> Set[Tuple[str, str, str]]:
        """
        Load completed comparisons from existing CSV.
        Returns a set of (category, image_id1, image_id2) tuples.
        """
        if not os.path.exists(csv_path):
            return set()

        completed = set()
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            completed.add((row['category'], row['image_id1'], row['image_id2']))

        logging.info(f"Loaded {len(completed)} completed comparisons from existing CSV\n")
        return completed

    def _evaluate_comparison(
        self,
        category: str,
        image_id_1: str,
        img_bytes_1: bytes,
        image_id_2: str,
        img_bytes_2: bytes
    ) -> dict:
        """
        Evaluate a single comparison between two images using multiple judges.
        Returns a dictionary with the evaluation result based on majority vote.
        """
        def evaluate_single(judge_id: int, is_1_first: bool):
            """Single judge evaluation"""
            images = [img_bytes_1, img_bytes_2] if is_1_first else [img_bytes_2, img_bytes_1]
            choice_map = {
                "first": image_id_1 if is_1_first else image_id_2,
                "second": image_id_2 if is_1_first else image_id_1,
            }

            logging.info(f"Judge {judge_id}: Evaluating {image_id_1} vs {image_id_2} "
                        f"with {image_id_1} as {'first' if is_1_first else 'second'} image.\n")

            evaluation, usage = self.evaluator_model.get_response(
                images=images,
                judge_prompt=self.judge_prompts[judge_id]
            )

            if not evaluation:
                return None

            real_choice = choice_map.get(evaluation.choice.lower())

            logging.info(f"Judge {judge_id}: Chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason, usage)

        # Run all evaluations in parallel
        judge_results = {}  # judge_id -> {True: result, False: result}

        num_judges = len(self.judge_prompts)
        with ThreadPoolExecutor(max_workers=min(num_judges * 2, 8)) as eval_executor:
            future_to_judge = {}
            for judge_id in range(num_judges):
                for is_1_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, judge_id, is_1_first)
                    future_to_judge[future] = (judge_id, is_1_first)

            for future in as_completed(future_to_judge):
                judge_id, is_1_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_1_first] = result

        # Aggregate consistent judges
        votes = {image_id_1: 0, image_id_2: 0}
        feedback_by_choice = {image_id_1: [], image_id_2: []}
        total_consistent_judges = 0
        all_usages = []

        for judge_id, results in judge_results.items():
            result_1_first = results.get(True)
            result_2_first = results.get(False)

            if result_1_first is None or result_2_first is None:
                continue

            choice_1_first, reason_1_first, usage_1_first = result_1_first
            choice_2_first, reason_2_first, usage_2_first = result_2_first

            # Collect usage data
            if usage_1_first:
                all_usages.append(usage_1_first)
            if usage_2_first:
                all_usages.append(usage_2_first)

            # Only count consistent judges
            if choice_1_first == choice_2_first:
                logging.info(f"Judge {judge_id}: Consistent - chose '{choice_1_first}'\n")
                total_consistent_judges += 1
                votes[choice_1_first] += 1
                feedback_by_choice[choice_1_first].append(reason_1_first)
            else:
                logging.warning(f"Judge {judge_id}: Inconsistent - skipping.\n")

        # Determine winner
        if total_consistent_judges == 0:
            logging.warning(f"No consistent judges for {image_id_1} vs {image_id_2}. "
                          f"Defaulting to {image_id_1}.\n")
            choice = image_id_1
            winner_score = 0.5
            vlm_reason = "No consistent preference detected."
        else:
            choice = max(votes, key=votes.get)
            winner_score = votes[choice] / total_consistent_judges
            vlm_reason = "\n".join(feedback_by_choice[choice])
            logging.info(f"🏆 WINNER: {choice} ({votes[choice]}/{total_consistent_judges} "
                        f"= {winner_score:.2%})\n")

        # Aggregate usage data
        completion_tokens = sum(u.completion_tokens for u in all_usages if u) if all_usages else 0
        prompt_tokens = sum(u.prompt_tokens for u in all_usages if u) if all_usages else 0
        total_tokens = sum(u.total_tokens for u in all_usages if u) if all_usages else 0
        reasoning_tokens = 0
        for usage in all_usages:
            if usage and usage.completion_tokens_details:
                reasoning_tokens += usage.completion_tokens_details.reasoning_tokens

        return {
            'category': category,
            'image_id1': image_id_1,
            'image_id2': image_id_2,
            'winner': choice,
            'reason': vlm_reason,
            'winner_score': winner_score,
            'consistent_judges': total_consistent_judges,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the priors evaluation pipeline for each image comparison in parallel.
        Supports resumption by skipping already completed comparisons.
        """
        csv_save_path = os.path.join(results_dir, 'priors_results.csv')

        # Load completed comparisons from existing CSV
        completed_comparisons = self._load_completed_comparisons(csv_save_path)

        # Group images by category
        category_groups = defaultdict(set)

        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(img_path)

            category, image_id = self._parse_filename(filename)
            if category is None or image_id is None:
                continue

            category_groups[category].add((image_id, img_bytes))

        logging.info(f"Found {len(category_groups)} categories to evaluate\n")

        # Collect all comparison tasks, skipping completed ones
        comparison_tasks = []
        for category in sorted(category_groups.keys()):
            images_in_category = sorted(category_groups[category], key=lambda x: x[0])

            for (image_id_1, img_bytes_1), (image_id_2, img_bytes_2) in \
                    itertools.combinations(images_in_category, 2):

                # Check if this comparison was already completed
                comparison_key = (category, image_id_1, image_id_2)

                if comparison_key in completed_comparisons:
                    continue

                comparison_tasks.append((
                    category, image_id_1, img_bytes_1, image_id_2, img_bytes_2
                ))

        total_comparisons = len(comparison_tasks)
        logging.info(f"Total comparisons to evaluate: {total_comparisons}\n")

        # Prepare CSV file for incremental writing
        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        # Open CSV in append mode for incremental writing
        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'category', 'image_id1', 'image_id2', 'winner', 'reason',
                'winner_score', 'consistent_judges',
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
                        desc="Evaluating priors",
                        unit="comparison"
                ):
                    result = future.result()

                    # Write result immediately to CSV with thread safety
                    with csv_lock:
                        result_df = pd.DataFrame([result])
                        result_df.to_csv(csvfile, index=False, header=False, mode='a')
                        csvfile.flush()  # Force write to disk

        logging.info(f"Priors evaluation completed. Results saved to: {csv_save_path}\n")
