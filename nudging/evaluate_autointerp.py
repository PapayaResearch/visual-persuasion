import os
import re
import io
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
    Analyzes zero-shot nudging results by comparing edited images against originals.
    Groups results by category and identifies thematic differences.
    """
    results_dir: str
    difference_detector_model: LanguageModel
    theme_summarizer_model: LanguageModel
    difference_prompt: str
    theme_prompt: str

    def _parse_filename(self, filename: str) -> Tuple[str, str, str, int]:
        """
        Parse filename to extract category, product_id, edit_type, and prior_index.

        Examples:
            SOFA_9760b07b_original.jpg -> ('SOFA', '9760b07b', 'original', -1)
            SOFA_9760b07b_no-prior.jpg -> ('SOFA', '9760b07b', 'no-prior', -1)
            SOFA_9760b07b_prior-0.jpg -> ('SOFA', '9760b07b', 'prior', 0)
        """
        match = re.match(r'([A-Z_]+)_([a-f0-9]+)_(original|no-prior|prior-(\d+))\.jpg', filename)

        category = match.group(1)
        product_id = match.group(2)
        edit_type_full = match.group(3)

        if edit_type_full == 'original':
            edit_type = 'original'
            prior_index = None
        elif edit_type_full == 'no-prior':
            edit_type = 'no-prior'
            prior_index = None
        else:
            edit_type = 'prior'
            prior_index = int(match.group(4))

        return category, product_id, edit_type, prior_index

    def _group_images(self, image_paths: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Group images by category and product.

        Args:
            image_paths: List of absolute paths to image files

        Returns:
            {
                'SOFA': {
                    '9760b07b': {
                        'original': '/path/to/SOFA_9760b07b_original.jpg',
                        'no-prior': '/path/to/SOFA_9760b07b_no-prior.jpg',
                        'prior-0': '/path/to/SOFA_9760b07b_prior-0.jpg',
                        ...
                    }
                }
            }
        """
        grouped = defaultdict(lambda: defaultdict(dict))

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            if not filename.endswith('.jpg'):
                continue

            category, product_id, edit_type, prior_index = self._parse_filename(filename)

            if edit_type == 'original':
                key = 'original'
            elif edit_type == 'no-prior':
                key = 'no-prior'
            else:
                key = f'prior-{prior_index}'

            grouped[category][product_id][key] = image_path

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
            task=self.difference_prompt,
            images=[original_bytes, edited_bytes]
        )

        logging.info(f"    Differences: {response.differences}\n")

        return label, response.differences

    def _summarize_themes(self, all_differences: List[str]) -> str:
        """
        Summarize all differences into main themes using LLM.

        Args:
            all_differences: List of difference descriptions from all products

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
        Analyze all results and generate theme summaries per category.

        Args:
            image_paths: List of absolute paths to image files to analyze
            results_dir: Directory where results will be saved
            max_workers: Maximum number of parallel workers
        """
        os.makedirs(results_dir, exist_ok=True)

        grouped_images = self._group_images(image_paths)

        for category, products in grouped_images.items():
            logging.info(f"\n===== Processing Category: {category} =====\n")

            # Prepare all comparison tasks
            comparison_tasks = []
            for product_id, images in products.items():
                logging.info(f"Product: {product_id}")

                original_path = images['original']

                # Add no-prior comparison
                comparison_tasks.append((original_path, images['no-prior'], 'no-prior'))

                # Add all prior comparisons
                prior_keys = sorted([k for k in images.keys() if k.startswith('prior-')])
                for prior_key in prior_keys:
                    comparison_tasks.append((original_path, images[prior_key], prior_key))

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
                    category_differences.append(f"[{label}] {diff}")

            # Summarize themes for this category
            logging.info(f"Summarizing themes for {category}...\n")
            themes = self._summarize_themes(category_differences)
            logging.info(f"Themes for {category}:\n{themes}\n")

            # Save to file
            output_file = os.path.join(results_dir, f"{category}_themes.txt")
            with open(output_file, "w") as f:
                f.write(f"Category: {category}\n\n")
                f.write(f"Main Themes:\n{themes}\n\n")
                f.write(f"All Differences ({len(category_differences)} comparisons):\n")
                for diff in category_differences:
                    f.write(f"{diff}\n")

            logging.info(f"Saved themes to: {output_file}\n")

        logging.info(f"Interpretation complete. Results saved to: {results_dir}\n")
