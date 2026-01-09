import logging
import os
import re
import csv
import shutil
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from dotenv import load_dotenv
from config import Config
from utils.misc import print_config
from tqdm import tqdm
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import LanguageModel

# Load environment variables from .env
load_dotenv()

# Initialize Hydra config store
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)


def evaluate_pair_comparability(
    image_id_1: str,
    img_bytes_1: bytes,
    image_id_2: str,
    img_bytes_2: bytes,
    judge_prompts: list[str],
    evaluator_model: LanguageModel,
    comparability_threshold: float
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

        evaluation = evaluator_model.get_response(
            images=images,
            judge_prompt=judge_prompts[judge_id],
            metadata="Context: the user is looking for a(n) %s." % image_id_1.split("_")[0].lower()
        )

        if evaluation is None:
            return None

        real_choice = choice_map.get(evaluation.choice.lower())
        return (real_choice, evaluation.reason)

    # Run all evaluations in parallel (each judge evaluates twice with swapped positions)
    judge_results = {}

    num_judges = len(judge_prompts)
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
    is_comparable = winner_score <= comparability_threshold

    logging.info(
        f"Pair {image_id_1} vs {image_id_2}: "
        f"winner={winner}, score={winner_score:.2%}, "
        f"comparable={is_comparable}"
    )

    return is_comparable, winner_score, winner


def filter_comparable_pairs(
    image_paths: list[str],
    results_dir: str,
    judge_prompts: list[str],
    evaluator_model: LanguageModel,
    comparability_threshold: float,
    category_pattern: str,
    max_workers: int = 1
) -> list[tuple[str, str]]:
    """
    Phase 1: Filter all pairs to find comparable images.
    Returns list of (image_path_1, image_path_2) tuples for comparable pairs.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"PHASE 1: Filtering Comparable Pairs")
    logging.info(f"{'='*80}\n")

    # Group images by category
    image_categories = {}
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        match = re.match(category_pattern, base_name)
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

        is_comparable, score, winner = evaluate_pair_comparability(
            id_1, image_bytes_cache[path_1],
            id_2, image_bytes_cache[path_2],
            judge_prompts,
            evaluator_model,
            comparability_threshold
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
    comparability_csv = os.path.join(results_dir, "comparability_results.csv")
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

    logging.info(f"\n{'='*60}")
    logging.info(f"Phase 1 Complete: Found {len(comparable_pairs)}/{len(all_pairs)} comparable pairs")
    logging.info(f"Comparability results saved to: {comparability_csv}")
    logging.info(f"{'='*60}\n")

    return comparable_pairs


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    # Load and print configuration
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Instantiate the competition strategy from config
    competition = hydra.utils.instantiate(cfg.strategy)

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build directory path: strategy/model/timestamp
    base_dir = competition.name

    results_dir = os.path.join(cfg.logging.results_dir, base_dir, current_date)

    if cfg.general.resume:
        # Find most recent results directory first
        existing_dirs = []
        results_pdir = os.path.dirname(results_dir)
        if os.path.exists(results_pdir):
            for d in os.listdir(results_pdir):
                dir_path = os.path.join(results_pdir, d)
                if os.path.isdir(dir_path):
                    existing_dirs.append(d)

        latest_dir = max(existing_dirs)
        print(f"Found existing results directory: {latest_dir}")
        previous_results_path = os.path.join(results_pdir, latest_dir)
        shutil.copytree(previous_results_path, results_dir)
        print(f"Resuming from previous results at: {previous_results_path}")

    # Set up logging
    log_dir = cfg.logging.log_dir
    log_file = os.path.join(log_dir, base_dir, current_date + ".log")
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

    # Create the results directory
    os.makedirs(results_dir, exist_ok=True)
    # Save config to output directories
    with open(os.path.join(results_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)
    with open(os.path.join(os.path.dirname(log_file), "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Get list of images from data directory
    data_dir = cfg.general.data_dir
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if os.path.isfile(os.path.join(data_dir, f))
                    and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    logging.info(f"\n{'='*80}")
    logging.info(f"COMPETITION PIPELINE")
    logging.info(f"Number of Images: {len(image_paths)}")
    logging.info(f"{'='*80}\n")

    # Phase 1: Filter comparable pairs
    phase1_dir = os.path.join(results_dir, "phase1_comparability")
    os.makedirs(phase1_dir, exist_ok=True)

    comparable_pairs = filter_comparable_pairs(
        image_paths=image_paths,
        results_dir=phase1_dir,
        judge_prompts=competition.judge_prompts,
        evaluator_model=competition.evaluator_model,
        comparability_threshold=cfg.competition.comparability_threshold,
        category_pattern=cfg.competition.category_pattern,
        max_workers=cfg.general.max_workers
    )

    if not comparable_pairs:
        logging.warning("No comparable pairs found. Exiting competition.")
        return

    # Phase 2: Run competition on comparable pairs
    logging.info(f"\n{'='*80}")
    logging.info(f"PHASE 2: Running Competition on {len(comparable_pairs)} Comparable Pairs")
    logging.info(f"{'='*80}\n")

    phase2_dir = os.path.join(results_dir, "phase2_competition")
    os.makedirs(phase2_dir, exist_ok=True)

    # Run competition on the comparable pairs
    competition.run(
        pairs=comparable_pairs,
        results_dir=phase2_dir,
        max_workers=cfg.general.max_workers
    )

    logging.info(f"\nCompetition pipeline complete. Results saved to: {results_dir}\n")

if __name__ == "__main__":
    main()
