import os
import logging
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel

class ImageStandardizer:
    def __init__(
        self,
        standardization_model: ImageModel,
        standardization_prompt: str
    ):
        self.standardization_model = standardization_model
        self.standardization_prompt = standardization_prompt

    def _standardize_single_image(self, src_image_path: str, dst_image_path: str, category: str):
        """Standardize a single image and save it to the destination path."""
        with open(src_image_path, 'rb') as f:
            src_image_bytes = f.read()

        # Format the prompt with the category
        prompt = self.standardization_prompt.format(category=category)

        standardized_image, _ = self.standardization_model.edit(
            prompt,
            src_image_bytes
        )

        if not standardized_image:
            logging.error(f"Standardization failed for image: {src_image_path}, skipping.\n")
            return False

        standardized_image.save(dst_image_path)
        return True

    def standardize_images(self, sampled_images: List[Tuple[str, str, str]], dst_dir: str, max_workers: int):
        """
        Standardizes sampled images and saves them to the destination directory.
        Skips images that already exist in dst_dir.

        Args:
            sampled_images: List of tuples (source_image_path, destination_filename, category)
            dst_dir: Directory where standardized images will be saved
            max_workers: Number of worker threads for parallel processing
        """
        os.makedirs(dst_dir, exist_ok=True)

        # Filter out images that already exist in dst_dir
        existing_images = set(os.listdir(dst_dir))
        images_to_standardize = [
            (src_path, dst_filename, category)
            for src_path, dst_filename, category in sampled_images
            if dst_filename not in existing_images
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._standardize_single_image,
                    src_path,
                    os.path.join(dst_dir, dst_filename),
                    category
                ): dst_filename
                for src_path, dst_filename, category in images_to_standardize
            }

            for future in tqdm(
                    as_completed(futures),
                    total=len(images_to_standardize),
                    desc="Standardizing images",
                    unit="image"
            ):
                future.result()
