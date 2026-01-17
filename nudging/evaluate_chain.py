import os
import json
import logging
from pathlib import Path
from typing import List

from utils.wrappers import LanguageModel


class ChainEvaluationPipeline:
    """
    Evaluation pipeline to assess progression along optimization chains.
    Tests if images get progressively better: final > winners > zero-shot > original
    """
    def __init__(
        self,
        evaluator_model: LanguageModel,
        strategy: str,
        results_dir: str,
        specific_product: str = None
    ):
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.strategy = strategy
        self.results_dir = results_dir
        self.specific_product = specific_product

    def _extract_chain_from_log(self, log_path: str) -> list[dict]:
        """
        Extract the chain of winner images from the competition log.
        Returns list of dicts with 'name', 'path', 'stage' for each image in the chain.
        """
        with open(log_path, 'r') as f:
            log_data = json.load(f)

        base_name = os.path.basename(log_path).replace('_log.json', '')
        chain = []

        if self.strategy == 'competition-no-bias':
            # Format: CATEGORY_ID_log.json
            # Original: CATEGORY_ID_original.jpg
            original_path = os.path.join(self.results_dir, f"{base_name}_original.jpg")
            if os.path.exists(original_path):
                chain.append({
                    'name': f"{base_name}_original",
                    'path': original_path,
                    'stage': 'original',
                    'round': 0
                })

            # Zero-shot winner (after round 1)
            zero_shot_path = os.path.join(self.results_dir, f"{base_name}_zero-shot.jpg")
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
                    winner_path = os.path.join(self.results_dir, f"{winner_name}_round-{round_num}_WINNER.jpg")

                    if os.path.exists(winner_path):
                        chain.append({
                            'name': f"{winner_name}_round-{round_num}",
                            'path': winner_path,
                            'stage': f'round-{round_num}-winner',
                            'round': round_num
                        })

            # Final image
            final_path = os.path.join(self.results_dir, f"{base_name}_final.jpg")
            if os.path.exists(final_path):
                chain.append({
                    'name': f"{base_name}_final",
                    'path': final_path,
                    'stage': 'final',
                    'round': len(log_data) + 1
                })

        elif self.strategy == 'competition':
            # Format: pair-X_CATEGORY_ID_vs_CATEGORY_ID_log.json
            # Need to handle two competing images
            # For now, we'll trace one of them (could be extended to trace both)
            logging.warning("Competition strategy chain extraction not yet implemented - only competition-no-bias supported")
            return []

        return chain

    def _evaluate_chain(self, chain: list[dict]) -> dict:
        """
        Evaluate all pairwise comparisons in the chain.
        Returns results showing if progression is monotonic.
        """
        results = {
            'chain': chain,
            'comparisons': [],
            'progressive': True
        }

        # Evaluate all pairs where i < j (earlier vs later)
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                earlier = chain[i]
                later = chain[j]

                # Load images
                with open(earlier['path'], 'rb') as f:
                    earlier_bytes = f.read()
                with open(later['path'], 'rb') as f:
                    later_bytes = f.read()

                # Evaluate with both orderings to check consistency
                logging.info(f"Evaluating: {earlier['stage']} vs {later['stage']}")

                # First ordering: earlier first
                eval_1, usage_1 = self.evaluator_model.get_response(
                    images=[earlier_bytes, later_bytes]
                )

                # Second ordering: later first
                eval_2, usage_2 = self.evaluator_model.get_response(
                    images=[later_bytes, earlier_bytes]
                )

                # Map choices back to earlier/later
                choice_1 = 'later' if eval_1.choice.lower() == 'second' else 'earlier'
                choice_2 = 'later' if eval_2.choice.lower() == 'first' else 'earlier'

                consistent = (choice_1 == choice_2)
                winner = choice_1 if consistent else 'inconsistent'

                comparison = {
                    'earlier': earlier['stage'],
                    'later': later['stage'],
                    'earlier_round': earlier['round'],
                    'later_round': later['round'],
                    'winner': winner,
                    'consistent': consistent,
                    'expected_winner': 'later',
                    'correct': (winner == 'later'),
                    'reason_1': eval_1.reason,
                    'reason_2': eval_2.reason
                }

                results['comparisons'].append(comparison)

                # Check if progression holds
                if winner != 'later':
                    results['progressive'] = False
                    logging.warning(f"  ❌ Progression violated: {earlier['stage']} vs {later['stage']} -> winner: {winner}")
                else:
                    logging.info(f"  ✅ Progression holds: {later['stage']} beats {earlier['stage']}")

        return results

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Run chain evaluation for all products in the results directory.
        Note: image_paths parameter is ignored - we use log files from results_dir instead.
        """
        # Find log files in the configured results directory
        log_files = list(Path(self.results_dir).glob("*_log.json"))
        if self.specific_product:
            # Filter to specific product if specified
            log_files = [f for f in log_files if self.specific_product in f.name]

        logging.info(f"Found {len(log_files)} log files to analyze")

        # Process each chain
        all_results = []

        for log_file in sorted(log_files):
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing: {log_file.name}")
            logging.info(f"{'='*80}")

            # Extract chain
            chain = self._extract_chain_from_log(str(log_file))

            if len(chain) < 2:
                logging.warning(f"Chain too short ({len(chain)} images), skipping")
                continue

            logging.info(f"Chain length: {len(chain)} images")
            for img in chain:
                logging.info(f"  - {img['stage']}: {img['name']}")

            # Evaluate chain
            results = self._evaluate_chain(chain)
            results['product'] = log_file.stem.replace('_log', '')
            all_results.append(results)

            # Print summary
            total_comparisons = len(results['comparisons'])
            correct_comparisons = sum(1 for c in results['comparisons'] if c['correct'])
            inconsistent_comparisons = sum(1 for c in results['comparisons'] if not c['consistent'])

            logging.info(f"\n{'─'*80}")
            logging.info(f"Summary for {results['product']}")
            logging.info(f"{'─'*80}")
            logging.info(f"Progressive: {results['progressive']}")
            logging.info(f"Correct progression: {correct_comparisons}/{total_comparisons} ({correct_comparisons/total_comparisons*100:.1f}%)")
            logging.info(f"Inconsistent evaluations: {inconsistent_comparisons}/{total_comparisons} ({inconsistent_comparisons/total_comparisons*100:.1f}%)")

        # Overall summary
        if all_results:
            logging.info(f"\n{'='*80}")
            logging.info(f"OVERALL SUMMARY")
            logging.info(f"{'='*80}")

            total_products = len(all_results)
            progressive_products = sum(1 for r in all_results if r['progressive'])

            all_comparisons = [c for r in all_results for c in r['comparisons']]
            total_comparisons = len(all_comparisons)
            correct_comparisons = sum(1 for c in all_comparisons if c['correct'])
            inconsistent_comparisons = sum(1 for c in all_comparisons if not c['consistent'])

            logging.info(f"Products analyzed: {total_products}")
            logging.info(f"Products with full progression: {progressive_products}/{total_products} ({progressive_products/total_products*100:.1f}%)")
            logging.info(f"Total comparisons: {total_comparisons}")
            logging.info(f"Correct progression: {correct_comparisons}/{total_comparisons} ({correct_comparisons/total_comparisons*100:.1f}%)")
            logging.info(f"Inconsistent evaluations: {inconsistent_comparisons}/{total_comparisons} ({inconsistent_comparisons/total_comparisons*100:.1f}%)")

            # Save detailed results to the evaluation results_dir (passed in run())
            output_path = os.path.join(results_dir, 'chain_evaluation_results.json')
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            logging.info(f"\nDetailed results saved to: {output_path}")
