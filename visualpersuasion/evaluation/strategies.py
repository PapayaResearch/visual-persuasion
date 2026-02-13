import os
import re
import yaml
import random
import logging
import itertools
import threading
import pandas as pd
import hydra
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple
from datetime import datetime
from omegaconf import OmegaConf
from dotenv import load_dotenv
from visualpersuasion.utils.wrappers import LanguageModel
from visualpersuasion.utils.misc import print_config


class StrategiesComparison:
    """
    Compare final images across different optimization strategies.
    """
    def __init__(
        self,
        evaluator_config: dict,
        task_names: List[str],
        name: str,
        data_dir: str,
        results_dir: str,
        model_name: str,
        max_comparisons: int = -1,
        sampling_seed: int = 42
    ):
        self.name = name
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.model_name = model_name
        self.max_comparisons = max_comparisons
        self.sampling_seed = sampling_seed

        # Get shared components (already instantiated by Hydra)
        input_schema = evaluator_config['input_schema']
        output_schema = evaluator_config['output_schema']
        api_call = evaluator_config['api_call']

        # Load task configs and create task-specific evaluators
        task_configs_dir = Path(__file__).parent / "conf" / "task"
        self.evaluator_models = {}

        for task_name in task_names:
            # Load task config using OmegaConf (proper Hydra way, no instantiation)
            task_cfg = OmegaConf.load(task_configs_dir / f"{task_name}.yaml")

            # Create evaluator model with task-specific system prompt
            evaluator_model = LanguageModel(
                system_prompt=task_cfg.evaluation.system_prompt,
                input_schema=input_schema,
                output_schema=output_schema,
                api_call=api_call
            )
            evaluator_model.return_usage_data = True

            self.evaluator_models[task_name] = evaluator_model

    def _load_completed_comparisons(self, csv_path: str) -> Set[Tuple[str, str, str, str, str]]:
        """
        Load completed comparisons from existing CSV.
        Returns a set of (task, category, strategy1, image_id1, strategy2, image_id2) tuples.
        """
        if not os.path.exists(csv_path):
            return set()

        completed = set()
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            completed.add((row['task'], row['category'], row['strategy1'], row['image_id1'], row['strategy2'], row['image_id2']))

        logging.info(f"Loaded {len(completed)} completed comparisons from existing CSV\n")
        return completed

    def _parse_final_filename(self, filename: str):
        """
        Parse final image filename to extract category and image_id.
        Example: WALLET_d9582141_final.jpg -> (WALLET, d9582141)
        Returns (category, image_id) or None if pattern doesn't match.
        """
        # Remove _final.jpg
        base = filename[:-10]

        # Match pattern: CATEGORY_ID
        match = re.match(r'^([A-Za-z]+)_([a-z0-9]+)$', base)

        if match is None:
            logging.warning(f"Skipping file with unexpected name format: {filename}")
            return None

        category = match.group(1)
        image_id = match.group(2)

        return category, image_id

    def _load_strategy_results(self):
        """
        Scan data_dir directory and load all final images with their metadata.
        Returns: list of dicts with {strategy, task, category, image_id, path, bytes}
        """
        all_images = []
        results_root = Path(self.data_dir)

        for strategy_path in results_root.iterdir():
            for timestamp_dir in strategy_path.iterdir():
                with open(timestamp_dir / "config.yaml") as f:
                    config = yaml.safe_load(f)

                task = config['task']['name']
                strategy = config['strategy']['name']

                for img_path in timestamp_dir.glob("*_final.jpg"):
                    parsed = self._parse_final_filename(img_path.name)
                    if parsed is None:
                        continue

                    category, image_id = parsed

                    with open(img_path, "rb") as f:
                        img_bytes = f.read()

                    all_images.append({
                        'strategy': strategy,
                        'task': task,
                        'category': category,
                        'image_id': image_id,
                        'path': str(img_path),
                        'bytes': img_bytes
                    })

        logging.info(f"Loaded {len(all_images)} final images across all strategies\n")
        return all_images

    def _create_comparison_pairs(self, images):
        """
        Create pairs of images for comparison:
        - Same category
        - Different strategies
        - Different image_ids (different originals)
        - Same task
        """
        # Group by (task, category)
        task_category_groups = defaultdict(list)
        for img in images:
            task_category_groups[(img['task'], img['category'])].append(img)

        comparison_pairs = []

        for (task, category), task_images in task_category_groups.items():
            logging.info(f"Task '{task}', Category '{category}': {len(task_images)} images")

            task_pairs = []
            for img1, img2 in itertools.combinations(task_images, 2):
                # Skip if same strategy
                if img1['strategy'] == img2['strategy']:
                    continue

                # Skip if same image_id (same original)
                if img1['image_id'] == img2['image_id']:
                    continue

                comparison_pairs.append((img1, img2))
                task_pairs.append((img1, img2))

            logging.info(f"  Created {len(task_pairs)} comparison pairs\n")

        return comparison_pairs

    def _evaluate_comparison(self, img1, img2):
        """
        Evaluate a single comparison using multi-judge evaluation.
        """
        # Get task-specific evaluator
        task = img1['task']
        evaluator_model = self.evaluator_models[task]

        def evaluate_single(is_first_order: bool):
            images = [img1['bytes'], img2['bytes']] if is_first_order else [img2['bytes'], img1['bytes']]
            choice_map = {
                "first": "first" if is_first_order else "second",
                "second": "second" if is_first_order else "first"
            }

            img1_label = f"{img1['strategy']}_{img1['category']}_{img1['image_id']}"
            img2_label = f"{img2['strategy']}_{img2['category']}_{img2['image_id']}"

            logging.info(f"Evaluating: {img1_label} vs {img2_label} ({'first-second' if is_first_order else 'second-first'})\n")

            evaluation, usage = evaluator_model.get_response(
                images=images,
                metadata=f"Comparing {img1['category']} images."
            )

            real_choice = choice_map.get(evaluation.choice.lower())
            logging.info(f"Judge chose {real_choice} - {evaluation.reason}\n")

            return (real_choice, evaluation.reason, usage)

        choice_1, reason_1, usage_1 = evaluate_single(is_first_order=True)
        choice_2, reason_2, usage_2 = evaluate_single(is_first_order=False)

        if choice_1 == choice_2:
            choice = choice_1
            vlm_reason = reason_1
        else:
            logging.warning(f"Inconsistent judges: {choice_1} vs {choice_2}\n")
            choice = "inconsistent"
            vlm_reason = f"Judge 1: {reason_1} | Judge 2: {reason_2}"

        return {
            'task': img1['task'],
            'category': img1['category'],
            'strategy1': img1['strategy'],
            'image_id1': img1['image_id'],
            'strategy2': img2['strategy'],
            'image_id2': img2['image_id'],
            'choice': choice,
            'reason': vlm_reason,
            'completion_tokens': usage_1.completion_tokens + usage_2.completion_tokens,
            'prompt_tokens': usage_1.prompt_tokens + usage_2.prompt_tokens,
            'total_tokens': usage_1.total_tokens + usage_2.total_tokens,
            'reasoning_tokens': (usage_1.completion_tokens_details.reasoning_tokens + usage_2.completion_tokens_details.reasoning_tokens)
                if usage_1.completion_tokens_details and usage_2.completion_tokens_details else 0
        }

    def run(self, max_workers: int = 1, results_dir: str = None):
        """
        Run cross-strategy comparison evaluation.
        """
        csv_save_path = os.path.join(results_dir, 'results_methods.csv')

        # Load completed comparisons
        completed_comparisons = self._load_completed_comparisons(csv_save_path)

        # Load all final images
        logging.info("Scanning results directory...\n")
        all_images = self._load_strategy_results()

        # Create ALL comparison pairs
        logging.info("Creating comparison pairs...\n")
        all_pairs = self._create_comparison_pairs(all_images)
        logging.info(f"Total comparison pairs: {len(all_pairs)}\n")

        # Sample pairs if max_comparisons is set
        pairs = all_pairs
        if self.max_comparisons > 0 and len(all_pairs) > self.max_comparisons:
            logging.info(f"Sampling {self.max_comparisons} comparisons from {len(all_pairs)} total (balanced by strategy pair, seed={self.sampling_seed})\n")

            random.seed(self.sampling_seed)

            # Group by strategy pair type
            strategy_pair_groups = defaultdict(list)
            for img1, img2 in all_pairs:
                strategy_pair = tuple(sorted([img1['strategy'], img2['strategy']]))
                strategy_pair_groups[strategy_pair].append((img1, img2))

            # Sample equally from each group
            per_group = self.max_comparisons // len(strategy_pair_groups)
            pairs = []
            for strategy_pair, group_pairs in strategy_pair_groups.items():
                sample_size = min(per_group, len(group_pairs))
                pairs.extend(random.sample(group_pairs, sample_size))
                logging.info(f"  {strategy_pair}: sampled {sample_size}/{len(group_pairs)}")

        # Filter out completed comparisons
        pairs = [
            (img1, img2) for img1, img2 in pairs
            if (img1['task'], img1['category'], img1['strategy'], img1['image_id'], img2['strategy'], img2['image_id']) not in completed_comparisons
        ]

        logging.info(f"Total comparisons to evaluate: {len(pairs)}\n")

        # Prepare CSV file for incremental writing
        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'task', 'category', 'strategy1', 'image_id1',
                'strategy2', 'image_id2',
                'choice', 'reason',
                'completion_tokens', 'prompt_tokens', 'total_tokens', 'reasoning_tokens'
            ]

            writer = pd.DataFrame(columns=fieldnames)

            if not file_exists:
                writer.to_csv(csvfile, index=False, header=True, mode='a')
                csvfile.flush()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_comparison, img1, img2): (img1, img2)
                    for img1, img2 in pairs
                }

                for future in tqdm(as_completed(futures), total=len(pairs), desc="Evaluating pairs", unit="pair"):
                    result = future.result()

                    with csv_lock:
                        result_df = pd.DataFrame([result])
                        result_df.to_csv(csvfile, index=False, header=False, mode='a')
                        csvfile.flush()

        logging.info(f"Method comparison complete! Results saved to: {csv_save_path}\n")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    # Load environment variables
    load_dotenv()

    # Load and print configuration
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Instantiate the comparison pipeline
    comparison = hydra.utils.instantiate(cfg.evaluate)

    # Create output directory structure
    output_dir = os.path.join(comparison.results_dir, comparison.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_dir = cfg.logging.log_dir
    log_file = os.path.join(log_dir, "methods", comparison.model_name + ".log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging_handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if cfg.logging.console:
        logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        handlers=logging_handlers,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    print(f"Logging to: {log_file}")
    logging.info(f"Logging to: {log_file}")

    # Save config to output directory
    with open(os.path.join(output_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Run the comparison
    logging.info(f"Starting methods comparison from {comparison.data_dir}\n")
    comparison.run(max_workers=cfg.general.max_workers, results_dir=output_dir)
    logging.info(f"Methods comparison completed: {output_dir}\n")


if __name__ == "__main__":
    main()
