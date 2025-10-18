import os
import io
import logging
from wrappers import ImageEditingModel

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import random
import matplotlib.pyplot as plt

class BackgroundProcessor:
    def __init__(
        self,
        image_editing_model: ImageEditingModel,
        num_previews_with_background: int,
        num_previews_without_background: int,
        background_removal_prompt: str,
        ssim_threshold: float,
        enable_background_normalization: bool,
        background_normalization_prompt: str
    ):
        self.image_editing_model = image_editing_model
        self.num_previews_with_background = num_previews_with_background
        self.num_previews_without_background = num_previews_without_background
        self.background_removal_prompt = background_removal_prompt
        self.ssim_threshold = ssim_threshold
        self.enable_background_normalization = enable_background_normalization
        self.background_normalization_prompt = background_normalization_prompt

    def _compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Computes the Structural Similarity Index (SSIM) between two images.
        """
        img1_gray = np.array(img1.convert("L"))
        img2_gray = np.array(img2.convert("L"))

        ssim_value, _ = ssim(img1_gray, img2_gray, full=True)
        return ssim_value
    
    def _generate_previews(self, src_dir: str, dst_dir_with_bg: str, dst_dir_without_bg: str, dst_dir_normalized_bg: str):
        """
        Generates preview images for the with-background and without-background datasets.
        """
        def calculate_subplot_dims(num_images):
            """Calculate rows and cols for a near-square subplot layout."""
            if num_images == 0:
                return 1, 1
            rows = int(np.sqrt(num_images))
            cols = int(np.ceil(num_images / rows))
            return rows, cols
        
        # Generate previews for with-background images
        with_bg_images = [f for f in os.listdir(dst_dir_with_bg) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))]
        random.shuffle(with_bg_images)
        selected_with_bg = with_bg_images[:min(self.num_previews_with_background, len(with_bg_images))]
        
        if selected_with_bg:
            rows, cols = calculate_subplot_dims(len(selected_with_bg))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            
            # Handle single image case
            if len(selected_with_bg) == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if rows > 1 else axes
            
            for i, img_name in enumerate(selected_with_bg):
                img_path = os.path.join(dst_dir_with_bg, img_name)
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(img_name, fontsize=8)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(selected_with_bg), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(src_dir, "preview_with_bg.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Generated preview_with_bg.png with {len(selected_with_bg)} images")

        # Generate previews for without-background images
        without_bg_images = [f for f in os.listdir(dst_dir_without_bg) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))]
        random.shuffle(without_bg_images)
        selected_without_bg = without_bg_images[:min(self.num_previews_without_background, len(without_bg_images))]
        
        if selected_without_bg:
            rows, cols = calculate_subplot_dims(len(selected_without_bg))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            
            # Handle single image case
            if len(selected_without_bg) == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if rows > 1 else axes
            
            for i, img_name in enumerate(selected_without_bg):
                img_path = os.path.join(dst_dir_without_bg, img_name)
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(img_name, fontsize=8)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(selected_without_bg), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(src_dir, "preview_without_bg.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Generated preview_without_bg.png with {len(selected_without_bg)} images")

            # Generate normalized previews using the same images
            if self.enable_background_normalization and os.path.exists(dst_dir_normalized_bg):
                rows, cols = calculate_subplot_dims(len(selected_without_bg))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                
                # Handle single image case
                if len(selected_without_bg) == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if rows > 1 else axes
                
                for i, img_name in enumerate(selected_without_bg):
                    normalized_img_path = os.path.join(dst_dir_normalized_bg, img_name)
                    if os.path.exists(normalized_img_path):
                        img = Image.open(normalized_img_path)
                        axes[i].imshow(img)
                        axes[i].set_title(f"Normalized {img_name}", fontsize=8)
                        axes[i].axis('off')
                
                # Hide unused subplots
                for i in range(len(selected_without_bg), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(src_dir, "preview_normalized.png"), dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Generated preview_normalized.png with {len(selected_without_bg)} images")

    def split_by_background(self, src_dir: str):
        """
        Splits images in the source directory into two directories based on whether the image has a normalized background.
        """
        enhanced_dir = os.path.join(src_dir, "enhanced")
        if os.path.exists(enhanced_dir):
            self.src_dir = enhanced_dir
        else:
            self.src_dir = os.path.join(src_dir, "base")
        self.dst_dir_with_bg = os.path.join(src_dir, "with_background")
        self.dst_dir_without_bg = os.path.join(src_dir, "without_background")
        self.dst_dir_normalized_bg = os.path.join(src_dir, "normalized_background")

        os.makedirs(self.dst_dir_with_bg, exist_ok=True)
        os.makedirs(self.dst_dir_without_bg, exist_ok=True)
        os.makedirs(self.dst_dir_normalized_bg, exist_ok=True)

        for file in os.listdir(self.src_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue

            original_img_path = os.path.join(self.src_dir, file)
            with open(original_img_path, 'rb') as f:
                original_image_bytes = f.read()

            # Use the background remover model to get the normalized image
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                self.background_removal_prompt,
                original_image_bytes
            )

            # Compute SSIM between original and edited image
            original_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
            ssim_value = self._compute_ssim(original_image, edited_image)

            # Decide destination based on SSIM threshold
            if ssim_value < self.ssim_threshold:
                dst_path = os.path.join(self.dst_dir_with_bg, file)
                with open(dst_path, 'wb') as out_f:
                    out_f.write(edited_image_bytes)
                    logging.info(f"Image {file} (SSIM value: {ssim_value}) copied to with-background directory.")
            else:
                dst_path = os.path.join(self.dst_dir_without_bg, file)
                with open(dst_path, 'wb') as out_f:
                    out_f.write(original_image_bytes)
                    logging.info(f"Image {file} (SSIM value: {ssim_value}) copied to without-background directory.")

                if self.enable_background_normalization:
                    # Further normalize the background using the normalization prompt
                    normalized_image, normalized_image_bytes = self.image_editing_model.edit(
                        self.background_normalization_prompt,
                        original_image_bytes
                    )
                    dst_normalized_path = os.path.join(self.dst_dir_normalized_bg, file)
                    with open(dst_normalized_path, 'wb') as norm_f:
                        norm_f.write(normalized_image_bytes)
                        logging.info(f"Image {file} normalized and saved to normalized-background directory.")
        
        # Generate preview images
        self._generate_previews(src_dir, self.dst_dir_with_bg, self.dst_dir_without_bg, self.dst_dir_normalized_bg)