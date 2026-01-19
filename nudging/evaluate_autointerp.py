import os
import re
import io
import csv
import logging
import dataclasses
from PIL import Image
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.wrappers import LanguageModel

@dataclasses.dataclass
class ResultsInterpreter:
    """
    Analyzes optimization results by comparing original images to final optimized versions.
    Identifies thematic differences that emerge from the optimization process.
    Groups results by category and generates theme summaries.
    """
    results_dir: str
    difference_detector_model: LanguageModel
    theme_summarizer_model: LanguageModel
    name: str

    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse filename to extract category, base_id, and status.

        Examples:
            SOFA_9760b07b_original.jpg -> ('SOFA', '9760b07b', 'original')
            SOFA_9760b07b_final.jpg -> ('SOFA', '9760b07b', 'final')
        """
        match = re.match(r'([A-Z_]+)_([a-f0-9]+)_(original|final)\.jpg', filename)

        if not match:
            return None, None, None

        category = match.group(1)
        base_id = match.group(2)
        status = match.group(3)

        return category, base_id, status

    def _group_images(self, image_paths: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Group images by category and base_id.

        Args:
            image_paths: List of absolute paths to image files

        Returns:
            {
                'SOFA': {
                    '9760b07b': {
                        'original': '/path/to/SOFA_9760b07b_original.jpg',
                        'final': '/path/to/SOFA_9760b07b_final.jpg'
                    }
                }
            }
        """
        grouped = defaultdict(lambda: defaultdict(dict))

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            if not filename.endswith('.jpg'):
                continue

            category, base_id, status = self._parse_filename(filename)

            if category is None:
                continue

            grouped[category][base_id][status] = image_path

        return grouped

    def _compare_images(self, original_path: str, edited_path: str, label: str) -> Tuple[str, str]:
        """
        Compare original and edited images using VLM to identify differences.

        Returns:
            Tuple of (label, differences)
        """
        with open(original_path, "rb") as f:
            original_bytes = f.read()
        with open(edited_path, "rb") as f:
            edited_bytes = f.read()

        logging.info(f"  Comparing {label}...")

        response = self.difference_detector_model.get_response(
            images=[original_bytes, edited_bytes]
        )

        logging.info(f"    Differences: {response.differences}\n")

        return label, response.differences

    def _summarize_themes(self, all_differences: List[str]) -> str:
        """
        Summarize all differences into main themes using LLM.

        Args:
            all_differences: List of difference descriptions from all images

        Returns:
            String describing main themes
        """
        differences_text = "\n".join([f"- {diff}" for diff in all_differences])

        response = self.theme_summarizer_model.get_response(
            differences=differences_text
        )

        return response.themes

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 8):
        """
        Analyze optimization results by comparing original vs final images.
        Generates theme summaries per category showing what the optimization changed.

        Args:
            image_paths: List of absolute paths to image files to analyze
            results_dir: Directory where results will be saved
            max_workers: Maximum number of parallel workers
        """
        os.makedirs(results_dir, exist_ok=True)

        grouped_images = self._group_images(image_paths)

        # Collect all results for CSV
        all_results = []

        for category, images_in_category in grouped_images.items():
            logging.info(f"\n===== Processing Category: {category} =====\n")

            # Prepare all comparison tasks
            comparison_tasks = []
            for base_id, images in images_in_category.items():
                logging.info(f"Image: {base_id}")

                # Get available images
                original = images.get('original')
                final = images.get('final')

                # Compare original vs final to identify optimization themes
                if original and final:
                    comparison_tasks.append((original, final, base_id))

            # Run all comparisons in parallel
            category_differences = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._compare_images, orig, edited, label): label
                          for orig, edited, label in comparison_tasks}

                for future in tqdm(
                        as_completed(futures),
                        total=len(comparison_tasks),
                        desc=f"{category} comparisons",
                        unit="comparison"
                ):
                    label, diff = future.result()
                    category_differences.append(diff)
                    # Store temporarily without themes
                    all_results.append({
                        'category': category,
                        'base_id': label,
                        'differences': diff,
                        'themes': None  # Will be filled in after summarization
                    })

            # Summarize themes for this category
            logging.info(f"Summarizing themes for {category}...\n")
            themes = self._summarize_themes(category_differences)
            logging.info(f"Themes for {category}:\n{themes}\n")

            # Update all results for this category with the themes
            for result in all_results:
                if result['category'] == category and result['themes'] is None:
                    result['themes'] = themes

        # Save all results to CSV
        csv_save_path = os.path.join(results_dir, 'results_autointerp.csv')
        with open(csv_save_path, 'w', newline='') as csvfile:
            fieldnames = ['category', 'base_id', 'differences', 'themes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        logging.info(f"Interpretation complete. Results saved to: {csv_save_path}\n")
