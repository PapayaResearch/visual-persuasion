import os
import logging
from wrappers import ImageEditingModel

class ImageEnhancer:
    def __init__(
        self,
        enhancement_model: ImageEditingModel,
        enhancement_prompt: str
    ):
        self.enhancement_model = enhancement_model
        self.enhancement_prompt = enhancement_prompt

    def enhance_images(self, src_dir):
        """
        Enhances the quality of images in the source directory using the specified enhancement model.
        """
        base_dir = os.path.join(src_dir, "base")
        enhanced_dir = os.path.join(src_dir, "enhanced")
        os.makedirs(enhanced_dir, exist_ok=True)

        image_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in image_files:
            img_path = os.path.join(base_dir, img_file)
            original_image_bytes = open(img_path, 'rb').read()
            enhanced_image, enhanced_image_bytes = self.enhancement_model.edit(
                self.enhancement_prompt,
                original_image_bytes
            )
            if enhanced_image is None:
                logging.error(f"Enhancement failed for image: {img_file}, skipping.\n")
                continue
            enhanced_image.save(os.path.join(enhanced_dir, img_file))

        print(f"Enhanced images saved to {enhanced_dir}")