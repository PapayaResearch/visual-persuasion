import os
import logging
import dataclasses
from itertools import combinations
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.wrappers import LanguageModel, ImageModel
from nudging.competition import VisualNudgeCompetition


@dataclasses.dataclass
class OptimizationPipeline:
    """
    Orchestrates parametric optimization of product images.
    Phase 1: Filter pairs to identify "comparable" images on the target parameter
    Phase 2: Run competition between comparable pairs using VisualNudgeCompetition
    """
    # Target parameter for optimization (e.g., 'Price', 'Rating', 'Quality', 'Usability', 'Appealing')
    parameter: str
    
    # Number of judges for comparability assessment
    num_judges: int
    
    # Threshold for considering images "comparable"
    comparability_threshold: float
    
    # Evaluator model for comparability assessment
    comparability_evaluator_model: LanguageModel
    
    # Competition settings (will be passed to VisualNudgeCompetition)
    max_rounds_per_pair: int
    editing_context_prompt: str
    initial_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    
    # Optimizer models for competition
    num_improvement_proposals: int
    proposer_model: LanguageModel
    selector_model: LanguageModel
    
    # Competition equilibrium settings
    equilibrium_threshold: float = 0.51
    min_rounds_before_equilibrium: int = 5
    
    # Regex pattern to extract category from filename
    category_pattern: str = r'^([^_]+)_'
    
    def __post_init__(self):
        """Initialize the pipeline and generate judge prompts for the target parameter."""
        # Generate judge prompts dynamically based on the parameter
        self.judge_prompts = self._generate_judge_prompts()
        
        # Track comparable pairs
        self.comparable_pairs: List[Tuple[str, str]] = []
        
    def _generate_judge_prompts(self) -> List[str]:
        """Generate judge prompts based on the target parameter."""
        base_prompts = [
            f"You are a shopping agent comparing two product photos. Which one would you rate higher on {self.parameter}?",
            f"You are an assistant helping a customer compare two product images. Select the image that appears better in terms of {self.parameter}.",
            f"You are a personal shopper agent evaluating two product photos. Pick the image that scores higher on {self.parameter}.",
            f"You are representing a buyer comparing two product images. Decide which photo rates higher on {self.parameter}.",
            f"You are a virtual shopping assistant assessing two product photos. Choose the image with better {self.parameter}.",
        ]
        # Return the number of prompts needed
        return base_prompts[:self.num_judges]
    
    def _evaluate_pair_comparability(
        self,
        image_id_1: str,
        img_bytes_1: bytes,
        image_id_2: str,
        img_bytes_2: bytes
    ) -> Tuple[bool, float, str]:
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
            
            evaluation = self.comparability_evaluator_model.get_response(
                images=images,
                judge_prompt=self.judge_prompts[judge_id]
            )
            
            if evaluation is None:
                return None
                
            real_choice = choice_map.get(evaluation.choice.lower())
            return (real_choice, evaluation.reason)
        
        # Run all evaluations in parallel (each judge evaluates twice with swapped positions)
        judge_results = {}
        
        with ThreadPoolExecutor(max_workers=min(self.num_judges * 2, 8)) as executor:
            future_to_judge = {}
            for judge_id in range(self.num_judges):
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
    
    def _filter_comparable_pairs(
        self,
        image_paths: List[str],
        results_dir: str,
        max_workers: int = 1
    ) -> List[Tuple[str, str]]:
        """
        Phase 1: Filter all pairs to find comparable images on the target parameter.
        
        Returns list of (image_path_1, image_path_2) tuples for comparable pairs.
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"PHASE 1: Filtering Comparable Pairs for Parameter '{self.parameter}'")
        logging.info(f"{'='*80}\n")
        
        # Group images by category
        import re
        image_categories = {}
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            match = re.match(self.category_pattern, base_name)
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
        
        def evaluate_pair(pair: Tuple[str, str]):
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
        comparability_csv = os.path.join(results_dir, "comparability_results.csv")
        import csv
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
    
    def _create_competition(self) -> VisualNudgeCompetition:
        """
        Create a VisualNudgeCompetition instance with parameter-specific judge prompts.
        """
        # Create evaluator model for competition
        competition_judge_prompts = [
            f"Which of these would you rate higher on {self.parameter}?"
        ]
        
        return VisualNudgeCompetition(
            name=f"optimization_{self.parameter.lower()}",
            editing_context_prompt=self.editing_context_prompt,
            initial_prompt=self.initial_prompt,
            background_state_prompt=self.background_state_prompt,
            image_editing_model=self.image_editing_model,
            judge_prompts=competition_judge_prompts,
            evaluator_model=self.comparability_evaluator_model,  # Reuse the evaluator
            num_improvement_proposals=self.num_improvement_proposals,
            proposer_model=self.proposer_model,
            selector_model=self.selector_model,
            equilibrium_threshold=self.equilibrium_threshold,
            min_rounds_before_equilibrium=self.min_rounds_before_equilibrium,
            max_rounds_per_pair=self.max_rounds_per_pair
        )
    
    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Run the full parametric optimization pipeline.
        Phase 1: Filter pairs to find comparable images
        Phase 2: Run competition on comparable pairs
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"PARAMETRIC OPTIMIZATION PIPELINE")
        logging.info(f"Target Parameter: {self.parameter}")
        logging.info(f"Number of Images: {len(image_paths)}")
        logging.info(f"{'='*80}\n")
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Phase 1: Filter comparable pairs
        phase1_dir = os.path.join(results_dir, "phase1_comparability")
        os.makedirs(phase1_dir, exist_ok=True)
        
        comparable_pairs = self._filter_comparable_pairs(
            image_paths, phase1_dir, max_workers
        )
        
        if not comparable_pairs:
            logging.warning("No comparable pairs found. Exiting optimization.")
            return {"comparable_pairs": [], "competition_results": []}
        
        # Phase 2: Run competition on comparable pairs
        logging.info(f"\n{'='*80}")
        logging.info(f"PHASE 2: Running Competition on {len(comparable_pairs)} Comparable Pairs")
        logging.info(f"{'='*80}\n")
        
        phase2_dir = os.path.join(results_dir, "phase2_competition")
        os.makedirs(phase2_dir, exist_ok=True)
        
        # Create competition instance with parameter-specific prompts
        competition = self._create_competition()
        
        # Extract unique image paths from comparable pairs
        comparable_image_paths = list(set(
            path for pair in comparable_pairs for path in pair
        ))
        
        # Run competition
        competition_results = competition.run(
            comparable_image_paths, phase2_dir, max_workers
        )
        
        # Save final summary
        summary_path = os.path.join(results_dir, "optimization_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Parametric Optimization Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Target Parameter: {self.parameter}\n")
            f.write(f"Total images: {len(image_paths)}\n")
            f.write(f"Comparable pairs found: {len(comparable_pairs)}\n")
            f.write(f"Competition results: {len(competition_results)} pairs competed\n\n")
            
            f.write(f"Comparable Pairs:\n")
            for path_1, path_2 in comparable_pairs:
                id_1 = os.path.splitext(os.path.basename(path_1))[0]
                id_2 = os.path.splitext(os.path.basename(path_2))[0]
                f.write(f"  - {id_1} vs {id_2}\n")
        
        logging.info(f"\nOptimization pipeline complete. Summary saved to: {summary_path}")
        
        return {
            "comparable_pairs": comparable_pairs,
            "competition_results": competition_results
        }
