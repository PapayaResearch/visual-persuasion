import os
import re
import csv
import logging
import dataclasses
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from utils.wrappers import LanguageModel


@dataclasses.dataclass
class ComparabilityEvaluator:
    """
    Pipeline to evaluate and filter image pairs for comparability based on judge feedback.
    """
    comparability_threshold: float
    judge_prompts: list[str]
    evaluator_model: LanguageModel

    def __post_init__(self):
        """Initialize the pipeline."""
        # Track comparable pairs
        self.comparable_pairs: list[tuple[str, str]] = []
    
    def _evaluate_pair_comparability(
        self,
        image_id_1: str,
        img_bytes_1: bytes,
        image_id_2: str,
        img_bytes_2: bytes
    ) -> tuple[bool, float, str]:
        """
        Evaluate whether a pair of images is "comparable" on the target parameter.
        A pair is considered comparable if the votes are evenly split.
        """
        def evaluate_single(judge_id: int, is_1_first: bool):
            """Single judge evaluation with position tracking."""
            images = [img_bytes_1, img_bytes_2] if is_1_first else [img_bytes_2, img_bytes_1]
            choice_map = {
                "first": image_id_1 if is_1_first else image_id_2,
                "second": image_id_2 if is_1_first else image_id_1,
            }

            evaluation = self.evaluator_model.get_response(
                images=images,
                judge_prompt=self.judge_prompts[judge_id],
                metadata="Context: the user is looking for a(n) %s." % image_id_1.split("_")[0].lower()
            )

            if evaluation is None:
                return None

            real_choice = choice_map.get(evaluation.choice.lower())
            return (real_choice, evaluation.reason)

        # Run all evaluations in parallel (each judge evaluates twice with swapped positions)
        judge_results = {}

        num_judges = len(self.judge_prompts)
        with ThreadPoolExecutor(max_workers=min(num_judges * 2, 8)) as executor:
            future_to_judge = {}
            for judge_id in range(num_judges):
                for is_1_first in [True, False]:
                    future = executor.submit(evaluate_single, judge_id, is_1_first)
                    future_to_judge[future] = (judge_id, is_1_first)

            for future in as_completed(future_to_judge):
                judge_id, is_1_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_1_first] = result

        # Aggregate consistent judges
        votes = {image_id_1: 0, image_id_2: 0}
        total_consistent_judges = 0

        for judge_id, results in judge_results.items():
            result_1_first = results.get(True)
            result_2_first = results.get(False)

            if result_1_first is None or result_2_first is None:
                continue

            choice_1_first, _ = result_1_first
            choice_2_first, _ = result_2_first

            # Only count consistent judges
            if choice_1_first == choice_2_first:
                total_consistent_judges += 1
                votes[choice_1_first] += 1

        # Determine comparability
        if total_consistent_judges == 0:
            # No consistent judges = treat as comparable (high uncertainty)
            return True, 0.5, image_id_1

        winner = max(votes, key=votes.get)
        winner_score = votes[winner] / total_consistent_judges

        # Pair is comparable if the decision is close to even
        is_comparable = winner_score <= self.comparability_threshold

        logging.info(
            f"Pair {image_id_1} vs {image_id_2}: "
            f"winner={winner}, score={winner_score:.2%}, "
            f"comparable={is_comparable}"
        )

        return is_comparable, winner_score, winner

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Filter image pairs to identify comparable images
        """
        logging.info(f"Number of Images: {len(image_paths)}")

        image_dir = os.path.dirname(image_paths[0])

        # Group images by category
        image_categories = {}
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            match = re.match("^([^_]+)_", base_name)
            if match:
                category = match.group(1)
            else:
                category = "default"
            image_categories.setdefault(category, []).append(image_path)

        # Generate all pairs within each category
        all_pairs = []
        for category, paths in image_categories.items():
            category_pairs = list(combinations(paths, 2))
            all_pairs.extend(category_pairs)

        logging.info(f"Total pairs to evaluate: {len(all_pairs)}")

        # Load image bytes
        image_bytes_cache = {}
        for path in image_paths:
            with open(path, "rb") as f:
                image_bytes_cache[path] = f.read()

        # Evaluate all pairs for comparability
        comparable_pairs = []
        comparability_results = []

        def evaluate_pair(pair: tuple[str, str]):
            path_1, path_2 = pair
            id_1 = os.path.splitext(os.path.basename(path_1))[0]
            id_2 = os.path.splitext(os.path.basename(path_2))[0]

            is_comparable, score, winner = self._evaluate_pair_comparability(
                id_1, image_bytes_cache[path_1],
                id_2, image_bytes_cache[path_2]
            )

            return {
                "pair": pair,
                "id_1": id_1,
                "id_2": id_2,
                "is_comparable": is_comparable,
                "score": score,
                "winner": winner
            }

        with tqdm(total=len(all_pairs), desc="Evaluating pairs", unit="pair") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(evaluate_pair, pair): pair for pair in all_pairs}

                for future in as_completed(futures):
                    result = future.result()
                    comparability_results.append(result)

                    if result["is_comparable"]:
                        comparable_pairs.append(result["pair"])

                    pbar.update(1)

        # Save comparability results
        comparability_csv = os.path.join(image_dir, "comparability_results.csv")
        with open(comparability_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id_1", "id_2", "is_comparable", "score", "winner"])
            writer.writeheader()
            for r in comparability_results:
                writer.writerow({
                    "id_1": r["id_1"],
                    "id_2": r["id_2"],
                    "is_comparable": r["is_comparable"],
                    "score": r["score"],
                    "winner": r["winner"]
                })

        if not comparable_pairs:
            logging.warning("No comparable pairs found.")
            return

        logging.info(f"\n{len(comparable_pairs)} comparable pairs identified. Saved in {image_dir}\n")

        return comparable_pairs