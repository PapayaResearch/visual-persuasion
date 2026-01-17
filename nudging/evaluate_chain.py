import os
import json
import logging
import threading
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import LanguageModel


class ChainEvaluationPipeline:
    """
    Evaluation pipeline to assess progression along optimization chains.
    Tests if images get progressively better: final > winners > zero-shot > original
    """
    def __init__(
        self,
        evaluator_model: LanguageModel,
        strategy_name: str
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.strategy_name = strategy_name

    def _extract_chain_from_log(self, log_path: str, images_dir: str) -> list[dict]:
        """
        Extract the chain of winner images from the competition log.
        Returns list of dicts with 'name', 'path', 'stage' for each image in the chain.
        """
        with open(log_path, 'r') as f:
            log_data = json.load(f)

        base_name = os.path.basename(log_path).replace('_log.json', '')
        chain = []

        if self.strategy_name == 'competition-no-bias':
            # Format: CATEGORY_ID_log.json
            # Original: CATEGORY_ID_original.jpg
            original_path = os.path.join(images_dir, f"{base_name}_original.jpg")
            if os.path.exists(original_path):
                chain.append({
                    'name': f"{base_name}_original",
                    'path': original_path,
                    'stage': 'original',
                    'round': 0
                })

            # Zero-shot winner (after round 1)
            zero_shot_path = os.path.join(images_dir, f"{base_name}_zero-shot.jpg")
            if os.path.exists(zero_shot_path):
                chain.append({
                    'name': f"{base_name}_zero-shot",
                    'path': zero_shot_path,
                    'stage': 'zero-shot',
                    'round': 1
                })

            # Extract winners from each round
            for round_log in log_data:
                round_num = round_log['round_number']
                if 'contest' in round_log and 'winner' in round_log['contest']:
                    winner_name = round_log['contest']['winner']
                    winner_path = os.path.join(images_dir, f"{winner_name}_round-{round_num}_WINNER.jpg")

                    if os.path.exists(winner_path):
                        chain.append({
                            'name': f"{winner_name}_round-{round_num}",
                            'path': winner_path,
                            'stage': f'round-{round_num}-winner',
                            'round': round_num
                        })

            # Final image
            final_path = os.path.join(images_dir, f"{base_name}_final.jpg")
            if os.path.exists(final_path):
                chain.append({
                    'name': f"{base_name}_final",
                    'path': final_path,
                    'stage': 'final',
                    'round': len(log_data) + 1
                })

        elif self.strategy_name == 'competition':
            # Format: pair-X_CATEGORY_ID_vs_CATEGORY_ID_log.json
            # For now, skip - can be extended later
            logging.warning("Competition strategy chain extraction not yet implemented")
            return []

        return chain

    def _evaluate_comparison(
        self,
        product: str,
        earlier_stage: str,
        earlier_round: int,
        earlier_bytes: bytes,
        later_stage: str,
        later_round: int,
        later_bytes: bytes
    ) -> dict:
        """
        Evaluate a single comparison between two images in the chain.
        Returns a dictionary with the evaluation result.
        """
        def evaluate_single(is_earlier_first: bool):
            """Single judge evaluation"""
            images = [earlier_bytes, later_bytes] if is_earlier_first else [later_bytes, earlier_bytes]
            choice_map = {
                "first": "earlier" if is_earlier_first else "later",
                "second": "later" if is_earlier_first else "earlier"
            }

            logging.info(f"Evaluating {earlier_stage} vs {later_stage} ({'earlier first' if is_earlier_first else 'later first'})\n")

            evaluation, usage = self.evaluator_model.get_response(
                images=images
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            logging.info(f"Judge chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason, usage)

        choice_1_first, reason_1_first, usage_1_first = evaluate_single(is_earlier_first=True)
        choice_2_first, reason_2_first, usage_2_first = evaluate_single(is_earlier_first=False)

        choice = None
        if choice_1_first == choice_2_first:
            choice = choice_1_first
            vlm_reason = reason_1_first
            consistent = True
        else:
            logging.warning(f"Inconsistent judge results for {earlier_stage} vs {later_stage}: {choice_1_first} vs {choice_2_first}\n")
            choice = "inconsistent"
            vlm_reason = f"Judge 1: {reason_1_first} | Judge 2: {reason_2_first}"
            consistent = False

        completion_tokens = usage_1_first.completion_tokens + usage_2_first.completion_tokens
        prompt_tokens = usage_1_first.prompt_tokens + usage_2_first.prompt_tokens
        total_tokens = usage_1_first.total_tokens + usage_2_first.total_tokens

        reasoning_tokens = 0
        if usage_1_first.completion_tokens_details and usage_2_first.completion_tokens_details:
            reasoning_tokens = usage_1_first.completion_tokens_details.reasoning_tokens + usage_2_first.completion_tokens_details.reasoning_tokens

        return {
            'product': product,
            'earlier_stage': earlier_stage,
            'earlier_round': earlier_round,
            'later_stage': later_stage,
            'later_round': later_round,
            'choice': choice,
            'consistent': consistent,
            'correct': (choice == 'later'),
            'reason': vlm_reason,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the chain evaluation pipeline.
        Finds log files in the data directory and evaluates progression chains.
        """
        csv_save_path = os.path.join(results_dir, 'chain_results.csv')

        # Determine images directory from image_paths
        if not image_paths:
            logging.error("No image paths provided")
            return

        images_dir = os.path.dirname(image_paths[0])
        logging.info(f"Looking for log files in: {images_dir}\n")

        # Find log files in the images directory
        log_files = list(Path(images_dir).glob("*_log.json"))
        logging.info(f"Found {len(log_files)} log files to analyze\n")

        if not log_files:
            logging.warning("No log files found in the data directory")
            return

        # Extract chains and prepare comparison tasks
        comparison_tasks = []

        for log_file in sorted(log_files):
            logging.info(f"Processing log: {log_file.name}")

            # Extract chain
            chain = self._extract_chain_from_log(str(log_file), images_dir)

            if len(chain) < 2:
                logging.warning(f"Chain too short ({len(chain)} images), skipping\n")
                continue

            logging.info(f"Chain length: {len(chain)} images")
            for img in chain:
                logging.info(f"  - {img['stage']}: {img['name']}")

            product = log_file.stem.replace('_log', '')

            # Create all pairwise comparisons where i < j (earlier vs later)
            for i in range(len(chain)):
                for j in range(i + 1, len(chain)):
                    earlier = chain[i]
                    later = chain[j]

                    # Load images
                    with open(earlier['path'], 'rb') as f:
                        earlier_bytes = f.read()
                    with open(later['path'], 'rb') as f:
                        later_bytes = f.read()

                    comparison_tasks.append((
                        product,
                        earlier['stage'],
                        earlier['round'],
                        earlier_bytes,
                        later['stage'],
                        later['round'],
                        later_bytes
                    ))

        total_comparisons = len(comparison_tasks)
        logging.info(f"\nTotal comparisons to evaluate: {total_comparisons}\n")

        if total_comparisons == 0:
            logging.warning("No comparisons to evaluate")
            return

        # Prepare CSV file for incremental writing
        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        # Open CSV in append mode for incremental writing
        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'product', 'earlier_stage', 'earlier_round', 'later_stage', 'later_round',
                'choice', 'consistent', 'correct', 'reason',
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

        logging.info(f"\nChain evaluation completed. Results saved to: {csv_save_path}\n")

        # Print summary
        df = pd.read_csv(csv_save_path)
        total = len(df)
        correct = df['correct'].sum()
        consistent = df['consistent'].sum()

        logging.info(f"="*80)
        logging.info(f"SUMMARY")
        logging.info(f"="*80)
        logging.info(f"Total comparisons: {total}")
        logging.info(f"Correct (later wins): {correct} ({correct/total*100:.1f}%)")
        logging.info(f"Consistent: {consistent} ({consistent/total*100:.1f}%)")
        logging.info(f"="*80)
