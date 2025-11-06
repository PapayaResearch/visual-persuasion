import os
import logging
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel

class ImageEnhancer:
    def __init__(
        self,
        enhancement_model: ImageModel,
        enhancement_prompt: str
    ):
        self.enhancement_model = enhancement_model
        self.enhancement_prompt = enhancement_prompt

    def _enhance_single_image(self, src_image_path: str, dst_image_path: str, category: str):
        """Enhance a single image and save it to the destination path."""
        with open(src_image_path, 'rb') as f:
            original_image_bytes = f.read()

        # Format the prompt with the category
        prompt = self.enhancement_prompt.format(category=category)

        enhanced_image, enhanced_image_bytes = self.enhancement_model.edit(
            prompt,
            original_image_bytes
        )

        if not enhanced_image:
            logging.error(f"Enhancement failed for image: {src_image_path}, skipping.\n")
            return False

        enhanced_image.save(dst_image_path)
        return True

    def enhance_images(self, sampled_images: List[Tuple[str, str, str]], dst_dir: str, max_workers: int):
        """
        Enhances sampled images and saves them to the destination directory.
        Skips images that already exist in dst_dir.

        Args:
            sampled_images: List of tuples (source_image_path, destination_filename, category)
            dst_dir: Directory where enhanced images will be saved
            max_workers: Number of worker threads for parallel processing
        """
        os.makedirs(dst_dir, exist_ok=True)

        # Filter out images that already exist in dst_dir
        existing_images = set(os.listdir(dst_dir))
        images_to_enhance = [
            (src_path, dst_filename, category)
            for src_path, dst_filename, category in sampled_images
            if dst_filename not in existing_images
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._enhance_single_image,
                    src_path,
                    os.path.join(dst_dir, dst_filename),
                    category
                ): dst_filename
                for src_path, dst_filename, category in images_to_enhance
            }

            for future in tqdm(
                    as_completed(futures),
                    total=len(images_to_enhance),
                    desc="Enhancing images",
                    unit="image"
            ):
                future.result()
