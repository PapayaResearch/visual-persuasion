import os
import re
import logging
import threading
import pandas as pd
from typing import List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import LanguageModel


class SoloEvaluationPipeline:
    """
    Evaluate single images independently (go / no-go) instead of pairwise comparisons.
    """

    def __init__(
        self,
        evaluator_model: LanguageModel,
        strategy_name: str,
        n_evaluations: int = 1
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.strategy_name = strategy_name
        self.n_evaluations = n_evaluations

    def _parse_filename_zero_shot(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse zero-shot filename: CLASS_ID_EDITTYPE.jpg
        Returns (class_name, image_id, edit_type)
        """
        match = re.match(r'([A-Za-z0-9_]+)_([A-Za-z0-9]+)_([A-Za-z0-9-]+)\\.jpg', filename)
        if not match:
            raise ValueError(f"Filename {filename} does not match zero-shot convention.")
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

        valid_statuses = ['final', 'original', 'no-prior', 'zero-shot']

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

    def _load_completed_decisions(self, csv_path: str) -> Set[Tuple[str, str]]:
        """
        Load completed evaluations to avoid duplicate work.
        """
        if not os.path.exists(csv_path):
            return set()

        df = pd.read_csv(csv_path)
        completed = set()
        for _, row in df.iterrows():
            completed.add((row['image_class'], row['base']))

        logging.info(f"Loaded {len(completed)} completed solo evaluations from existing CSV\n")
        return completed

    def _evaluate_image(
        self,
        image_class: str,
        image_id: str,
        edit_type: str,
        img_bytes: bytes
    ) -> dict:
        """
        Run a single go/no-go evaluation for an image.
        """
        base = f"{image_id}_{edit_type}"
        metadata = f"The product here is a(n) {image_class}."

        evaluation, usage = self.evaluator_model.get_response(
            images=[img_bytes],
            metadata=metadata
        )

        decision = evaluation.choice.lower()
        reason = evaluation.reason

        completion_tokens = getattr(usage, "completion_tokens", 0)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        reasoning_tokens = 0
        if getattr(usage, "completion_tokens_details", None):
            reasoning_tokens = getattr(
                usage.completion_tokens_details,
                "reasoning_tokens",
                0
            )

        return {
            'image_class': image_class,
            'image_id': image_id,
            'edit_type': edit_type,
            'base': base,
            'decision': decision,
            'reason': reason,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Execute go/no-go evaluations for every image found in the supplied directory.
        """
        csv_save_path = os.path.join(results_dir, 'results_solo.csv')
        completed = self._load_completed_decisions(csv_save_path)

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
            else:
                # Default: treat entire dataset as a single class
                class_groups['default'].add((filename, 'original', img_bytes))

        logging.info(f"Found {sum(len(v) for v in class_groups.values())} images to evaluate in solo mode\n")

        tasks = []
        for image_class, images in sorted(class_groups.items()):
            for image_id, edit_type, img_bytes in images:
                base = f"{image_id}_{edit_type}"
                key = (image_class, base)
                if key in completed:
                    continue
                tasks.append((image_class, image_id, edit_type, img_bytes))

        tasks = tasks * self.n_evaluations
        total_tasks = len(tasks)
        logging.info(f"Total solo evaluations to run: {total_tasks}\n")

        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_class', 'image_id', 'edit_type', 'base', 'decision', 'reason',
                'completion_tokens', 'prompt_tokens', 'total_tokens', 'reasoning_tokens'
            ]
            if not file_exists:
                pd.DataFrame(columns=fieldnames).to_csv(csvfile, index=False, header=True, mode='a')
                csvfile.flush()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_image, *task): task for task in tasks
                }

                for future in tqdm(
                        as_completed(futures),
                        total=total_tasks,
                        desc="Solo evaluations",
                        unit="image"
                ):
                    result = future.result()
                    with csv_lock:
                        pd.DataFrame([result]).to_csv(csvfile, index=False, header=False, mode='a')
                        csvfile.flush()

        logging.info(f"Solo evaluation completed. Results saved to: {csv_save_path}\n")
