import os
import io
import logging
import threading
import pandas as pd
from typing import List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import LanguageModel


class UIEvaluationPipeline:
    """
    Evaluate UI competition outputs by comparing final variants for each product.
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

    def _evaluate_pair(
        self,
        product: str,
        variant_a: str,
        bytes_a: bytes,
        variant_b: str,
        bytes_b: bytes
    ) -> dict:
        """
        Run a single pairwise evaluation with both orderings to ensure consistency.
        """

        def evaluate_single(is_a_first: bool):
            images = [bytes_a, bytes_b] if is_a_first else [bytes_b, bytes_a]
            choice_map = {
                "first": variant_a if is_a_first else variant_b,
                "second": variant_b if is_a_first else variant_a
            }

            evaluation, usage = self.evaluator_model.get_response(
                images=images,
                metadata=f"Product UI comparison for {product}"
            )
            real_choice = choice_map.get(evaluation.choice.lower())
            reason = evaluation.reason
            return real_choice, reason, usage

        choice_a, reason_a, usage_a = evaluate_single(True)
        choice_b, reason_b, usage_b = evaluate_single(False)

        if choice_a == choice_b:
            winning_variant = choice_a
            winning_reason = reason_a if winning_variant == variant_a else reason_b
        else:
            logging.warning(
                f"Inconsistent UI evaluation for {product}: {choice_a} vs {choice_b}\n"
            )
            winning_variant = "inconsistent"
            winning_reason = f"A: {reason_a}\nB: {reason_b}"

        completion_tokens = getattr(usage_a, "completion_tokens", 0) + getattr(usage_b, "completion_tokens", 0)
        prompt_tokens = getattr(usage_a, "prompt_tokens", 0) + getattr(usage_b, "prompt_tokens", 0)
        total_tokens = getattr(usage_a, "total_tokens", 0) + getattr(usage_b, "total_tokens", 0)

        reasoning_tokens = 0
        if getattr(usage_a, "completion_tokens_details", None):
            reasoning_tokens += getattr(usage_a.completion_tokens_details, "reasoning_tokens", 0)
        if getattr(usage_b, "completion_tokens_details", None):
            reasoning_tokens += getattr(usage_b.completion_tokens_details, "reasoning_tokens", 0)

        return {
            "product": product,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "choice": winning_variant,
            "reason": winning_reason,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens
        }

    def _collect_variants(self, image_paths: List[str]) -> dict:
        """
        Build a mapping of product -> {final variants, baseline variants}.
        """
        variants = defaultdict(lambda: {"final": [], "baseline": []})
        prefix = "ui-"
        final_suffix = "_final.jpg"
        baseline_suffix = "_round-1_candidate-1.jpg"

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            if not filename.startswith(prefix):
                continue

            if filename.endswith(final_suffix):
                bucket = "final"
                suffix = final_suffix
            elif filename.endswith(baseline_suffix):
                bucket = "baseline"
                suffix = baseline_suffix
            else:
                continue

            core = filename[len(prefix):-len(suffix)]
            if "_" not in core:
                continue
            product, variant = core.split("_", 1)

            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
            except OSError:
                logging.warning(f"Failed to read UI variant image: {img_path}\n")
                continue

            label = f"{variant} ({bucket})"
            variants[product][bucket].append({
                "label": label,
                "bytes": img_bytes,
                "filename": filename
            })

        return variants

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Evaluate UI finals across products and save results to CSV.
        """
        csv_path = os.path.join(results_dir, "results_ui.csv")
        os.makedirs(results_dir, exist_ok=True)

        variant_map = self._collect_variants(image_paths)
        tasks: List[Tuple[str, str, bytes, str, bytes]] = []

        for product, splits in variant_map.items():
            finals = splits["final"]
            baselines = splits["baseline"]
            if not finals or not baselines:
                continue

            for final in finals:
                for baseline in baselines:
                    tasks.append((
                        product,
                        f"{final['label']} [{final['filename']}]",
                        final["bytes"],
                        f"{baseline['label']} [{baseline['filename']}]",
                        baseline["bytes"]
                    ))

        tasks = tasks * self.n_evaluations
        total_tasks = len(tasks)

        if total_tasks == 0:
            logging.warning("No UI variants found to evaluate.\n")
            return

        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "product", "variant_a", "variant_b", "choice", "reason",
                "completion_tokens", "prompt_tokens", "total_tokens", "reasoning_tokens"
            ]
            if not file_exists:
                pd.DataFrame(columns=fieldnames).to_csv(csvfile, index=False, header=True, mode='a')
                csvfile.flush()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_pair, *task): task for task in tasks
                }

                for future in tqdm(
                        as_completed(futures),
                        total=total_tasks,
                        desc="Evaluating UI finals",
                        unit="comparison"
                ):
                    result = future.result()
                    with csv_lock:
                        pd.DataFrame([result]).to_csv(csvfile, index=False, header=False, mode='a')
                        csvfile.flush()

        logging.info(f"UI evaluation complete. Results stored at {csv_path}\n")
