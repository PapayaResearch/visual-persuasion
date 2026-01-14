import os
import re
import csv
import random
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
    
    def _conduct_contest(
        self,
        image_a_bytes: bytes,
        image_b_bytes: bytes,
        image_a_name: str,
        image_b_name: str
    ) -> tuple[str, float, str]:
        """
        Conduct a multi-judge contest between two images.
        Returns: (winner_name, winner_score, aggregated_feedback)
        winner_score is the proportion of consistent judges preferring the winner (0-1)
        """
        logging.info(f"\n🥊 CONTEST: {image_a_name} vs {image_b_name}\n")

        def evaluate_single(judge_id: int, is_a_first: bool):
            """Single judge evaluation"""
            images = [image_a_bytes, image_b_bytes] if is_a_first else [image_b_bytes, image_a_bytes]
            choice_map = {
                "first": image_a_name if is_a_first else image_b_name,
                "second": image_b_name if is_a_first else image_a_name,
            }

            logging.info(f"Judge {judge_id}: Evaluating with {image_a_name} as {'first' if is_a_first else 'second'} image.\n")

            evaluation = self.evaluator_model.get_response(
                images=images,
                judge_prompt=self.judge_prompts[judge_id],
                metadata="Context: the user is looking for a(n) %s." % image_a_name.split("_")[0].lower()
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            logging.info(f"Judge {judge_id}: Chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason)

        def break_tie(image_a_name: str, image_b_name: str):
            """Apply tie-breaking strategy"""
            if self.tie_breaking_strategy == "first":
                winner = image_a_name
            elif self.tie_breaking_strategy == "second":
                winner = image_b_name
            elif self.tie_breaking_strategy == "random":
                winner = random.choice([image_a_name, image_b_name])
            else:
                raise ValueError(f"Unknown tie-breaking strategy: {self.tie_breaking_strategy}")
            return winner

        # Run all evaluations in parallel
        judge_results = {}  # judge_id -> {True: result, False: result}

        num_judges = len(self.judge_prompts)
        with ThreadPoolExecutor(max_workers=min(num_judges * 2, 8)) as eval_executor:
            future_to_judge = {}
            for judge_id in range(num_judges):
                for is_a_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, judge_id, is_a_first)
                    future_to_judge[future] = (judge_id, is_a_first)

            for future in as_completed(future_to_judge):
                judge_id, is_a_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_a_first] = result

        # Aggregate consistent judges
        votes = {image_a_name: 0, image_b_name: 0}
        feedback_by_choice = {image_a_name: [], image_b_name: []}
        total_consistent_judges = 0

        for judge_id, results in judge_results.items():
            result_a_first = results.get(True)
            result_b_first = results.get(False)

            choice_a_first, reason_a_first = result_a_first
            choice_b_first, reason_b_first = result_b_first

            # Only count consistent judges
            if choice_a_first == choice_b_first:
                logging.info(f"Judge {judge_id}: Consistent - chose '{choice_a_first}'\n")
                total_consistent_judges += 1
                votes[choice_a_first] += 1
                feedback_by_choice[choice_a_first].append(reason_a_first)
            else:
                logging.warning(f"Judge {judge_id}: Inconsistent - skipping.\n")

        # Determine winner
        if total_consistent_judges == 0:
            logging.warning("No consistent judges. Applying tie-breaking strategy.\n")
            winner_score = 0.5
            feedback = "No consistent preference detected."
            winner = break_tie(image_a_name, image_b_name)
        else:
            winner = max(votes, key=votes.get)
            winner_score = votes[winner] / total_consistent_judges
            feedback = "\n".join(feedback_by_choice[winner])

            # Check for 50-50 tie and apply tie-breaking
            if winner_score == 0.5:
                logging.warning("Judges split 50-50. Applying tie-breaking strategy.\n")
                winner = break_tie(image_a_name, image_b_name)

            logging.info(f"🏆 WINNER: {winner} ({votes[winner]}/{total_consistent_judges} = {winner_score:.2%})\n")

        return winner, winner_score, feedback

    def run(self, image_paths: list[str], max_workers: int = 1):
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

            winner, winner_score, feedback = self._conduct_contest(
                image_bytes_cache[path_1],
                image_bytes_cache[path_2],
                id_1, id_2
            )

            is_comparable = (winner_score <= self.comparability_threshold)

            return {
                "pair": pair,
                "id_1": id_1,
                "id_2": id_2,
                "is_comparable": is_comparable,
                "score": winner_score,
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