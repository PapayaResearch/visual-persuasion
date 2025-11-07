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
        strategy_name: str,
        judge_prompts: List[str]
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.strategy_name = strategy_name
        self.judge_prompts = judge_prompts

    def _parse_filename_zero_shot(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse zero-shot filename: CLASS_ID_EDITTYPE.jpg
        Returns (class_name, image_id, edit_type)
        """
        match = re.match(r'([A-Za-z0-9_]+)_([A-Za-z0-9]+)_([A-Za-z0-9-]+)\.jpg', filename)
        class_name = match.group(1)
        image_id = match.group(2)
        edit_type = match.group(3)
        return class_name, image_id, edit_type

    def _parse_filename_competition(self, filename: str):
        """
        Parse competition filename: CATEGORY_ID_STATUS.jpg or pair-X_..._CATEGORY_ID_STATUS.jpg
        Returns (category, image_id, status) or None if should skip.
        """
        if not filename.endswith('.jpg'):
            return None

        valid_statuses = ['final', 'original', 'no-prior', 'round-1_candidate-1']

        for status in valid_statuses:
            if filename.endswith(f'_{status}.jpg'):
                base = filename[:-4]
                parts = base.split('_')
                status_parts = status.split('_')
                num_status_parts = len(status_parts)
                image_id = parts[-(num_status_parts + 1)]
                category = parts[-(num_status_parts + 2)]
                return category, image_id, status

        return None

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

        def evaluate_single(judge_id: int, is_1_first: bool):
            """Single judge evaluation"""
            images = [img_bytes_1, img_bytes_2] if is_1_first else [img_bytes_2, img_bytes_1]
            choice_map = {
                "first": base_1 if is_1_first else base_2,
                "second": base_2 if is_1_first else base_1,
            }

            logging.info(f"Judge {judge_id}: Evaluating with {base_1} as {'first' if is_1_first else 'second'} image.\n")

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
        votes = {base_1: 0, base_2: 0}
        feedback_by_choice = {base_1: [], base_2: []}
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
            logging.warning(f"No consistent judges for {base_1} vs {base_2}. Defaulting to {base_1}.\n")
            choice = base_1
            winner_score = 0.5
            vlm_reason = "No consistent preference detected."
        else:
            choice = max(votes, key=votes.get)
            winner_score = votes[choice] / total_consistent_judges
            vlm_reason = "\n".join(feedback_by_choice[choice])
            logging.info(f"🏆 WINNER: {choice} ({votes[choice]}/{total_consistent_judges} = {winner_score:.2%})\n")

        # Aggregate usage data
        completion_tokens = sum(u.completion_tokens for u in all_usages if u) if all_usages else 0
        prompt_tokens = sum(u.prompt_tokens for u in all_usages if u) if all_usages else 0
        total_tokens = sum(u.total_tokens for u in all_usages if u) if all_usages else 0
        reasoning_tokens = 0
        for usage in all_usages:
            if usage and usage.completion_tokens_details:
                reasoning_tokens += usage.completion_tokens_details.reasoning_tokens

        # For backward compatibility with existing code, randomly assign first/second
        is_img1_first = random.choice([True, False])
        first = base_1 if is_img1_first else base_2
        second = base_2 if is_img1_first else base_1

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
            'consistent_judges': total_consistent_judges,
            'winner_score': winner_score
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

            if self.strategy_name == 'zero-shot':
                class_name, image_id, edit_type = self._parse_filename_zero_shot(filename)
                class_groups[class_name].add((image_id, edit_type, img_bytes))
            elif self.strategy_name == 'competition':
                parsed = self._parse_filename_competition(filename)
                if parsed is None:
                    continue
                category, image_id, status = parsed
                class_groups[category].add((image_id, status, img_bytes))

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
                comparison_key = (image_class, base_1, base_2)

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
