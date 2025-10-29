import os
import logging
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

    def _enhance_single_image(self, base_dir, enhanced_dir, img_file):
        """Enhance a single image."""
        img_path = os.path.join(base_dir, img_file)
        with open(img_path, 'rb') as f:
            original_image_bytes = f.read()
        enhanced_image, enhanced_image_bytes = self.enhancement_model.edit(
            self.enhancement_prompt,
            original_image_bytes
        )
        if not enhanced_image:
            logging.error(f"Enhancement failed for image: {img_file}, skipping.\n")
            return False
        enhanced_image.save(os.path.join(enhanced_dir, img_file))
        return True

    def enhance_images(self, src_dir, max_workers=10):
        """
        Enhances the quality of images in the source directory using the specified enhancement model.
        """
        base_dir = os.path.join(src_dir, "base")
        enhanced_dir = os.path.join(src_dir, "enhanced")
        os.makedirs(enhanced_dir, exist_ok=True)

        image_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._enhance_single_image, base_dir, enhanced_dir, img_file): img_file
                      for img_file in image_files}

            for future in tqdm(as_completed(futures), total=len(image_files), desc="Enhancing images", unit="image"):
                future.result()

        logging.info(f"Enhanced images saved to {enhanced_dir}\n")
