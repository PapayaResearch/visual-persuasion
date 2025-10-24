import os
import io
import logging
from utils.wrappers import ImageModel
from typing import Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import random
import matplotlib.pyplot as plt

class BackgroundProcessor:
    def __init__(
        self,
        image_editing_model: ImageModel,
        max_previews: int,
        background_removal_prompt: str,
        ssim_threshold: float,
        enable_background_normalization: bool,
        background_normalization_prompt: str
    ):
        self.image_editing_model = image_editing_model
        self.max_previews = max_previews
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

    def _calculate_subplot_dims(self, num_images: int) -> Tuple[int, int]:
        """Calculate rows and cols for a near-square subplot layout."""
        if num_images == 0:
            return 1, 1
        rows = int(np.sqrt(num_images))
        cols = int(np.ceil(num_images / rows))
        return rows, cols
    
    def _generate_preview(self, image_dir: str, normalized_dir: str, dst_file: str, title: str):
        """Generates a single preview image showing samples from the specified directory."""
        images = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))]
        if not images:
            logging.warning(f"No images found in {image_dir} for preview generation.\n")
            return
        
        random.shuffle(images)
        num_previews = self.max_previews if (self.max_previews != -1 and self.max_previews < len(images)) else len(images)
        selected_images = images[:num_previews]

        show_normalized = self.enable_background_normalization and normalized_dir is not None
        
        # Calculate subplot dimensions
        rows, cols = self._calculate_subplot_dims(len(selected_images))
        
        # Adjust figure size and column count based on whether we show normalized versions
        if show_normalized:
            fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        # Add plot title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Ensure axes is always 2D array for consistent indexing
        total_cols = cols * 2 if show_normalized else cols
        if rows == 1 and total_cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif total_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot images
        for i, img_name in enumerate(selected_images):
            row = i // cols
            col = i % cols  
            if show_normalized:
                # Original image (left)
                original_img_path = os.path.join(image_dir, img_name)
                original_img = Image.open(original_img_path)
                axes[row, col * 2].imshow(original_img)
                axes[row, col * 2].set_title(f"Original\n{img_name}", fontsize=8)
                axes[row, col * 2].axis('off')
                # Normalized image (right)
                normalized_img_path = os.path.join(normalized_dir, img_name)
                if os.path.exists(normalized_img_path):
                    normalized_img = Image.open(normalized_img_path)
                    axes[row, col * 2 + 1].imshow(normalized_img)
                    axes[row, col * 2 + 1].set_title(f"Normalized\n{img_name}", fontsize=8)
                else:
                    axes[row, col * 2 + 1].text(0.5, 0.5, 'Missing', ha='center', va='center',
                                                transform=axes[row, col * 2 + 1].transAxes)
                axes[row, col * 2 + 1].axis('off')
            else:
                # Single image view
                img_path = os.path.join(image_dir, img_name)
                img = Image.open(img_path)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"{img_name}", fontsize=8)
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(len(selected_images), rows * cols):
            row = i // cols
            col = i % cols
            if show_normalized:
                axes[row, col * 2].axis('off')
                axes[row, col * 2 + 1].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        preview_path = os.path.join(dst_file)
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        img_count_text = f"{len(selected_images)} image pairs" if show_normalized else f"{len(selected_images)} images"
        logging.info(f"Generated {title} preview with {img_count_text} at: {preview_path}\n")

    def split_by_background(self, src_dir: str):
        """
        Splits images in the source directory into two directories based on whether the image has a normalized background.
        """
        base_dir, enhanced_dir = os.path.join(src_dir, "base"), os.path.join(src_dir, "enhanced")
        self.src_dir = enhanced_dir if os.path.exists(enhanced_dir) else base_dir

        self.dst_dir_bg = os.path.join(src_dir, "with_background")
        self.dst_dir_no_bg = os.path.join(src_dir, "without_background")
        os.makedirs(self.dst_dir_bg, exist_ok=True)
        os.makedirs(self.dst_dir_no_bg, exist_ok=True)

        # Create normalized directories if normalization is enabled
        if self.enable_background_normalization:
            self.dst_dir_bg_normalized = os.path.join(src_dir, "with_background_normalized")
            self.dst_dir_no_bg_normalized = os.path.join(src_dir, "without_background_normalized")
            os.makedirs(self.dst_dir_bg_normalized, exist_ok=True)
            os.makedirs(self.dst_dir_no_bg_normalized, exist_ok=True)
        else:
            self.dst_dir_bg_normalized = None
            self.dst_dir_no_bg_normalized = None

        for file in os.listdir(self.src_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue

            original_img_path = os.path.join(self.src_dir, file)
            with open(original_img_path, 'rb') as f:
                original_image_bytes = f.read()

            # Use the background remover model to get the edited image
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                self.background_removal_prompt,
                original_image_bytes
            )
            if not edited_image:
                logging.error(f"Background removal failed for image: {file}, skipping.\n")
                continue

            # Compute SSIM between original and edited image
            original_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
            ssim_value = self._compute_ssim(original_image, edited_image)

            # Decide if the image has a background based on SSIM threshold
            has_bg = ssim_value < self.ssim_threshold
            dst_path = os.path.join(self.dst_dir_bg if has_bg else self.dst_dir_no_bg, file)
            with open(dst_path, 'wb') as out_f:
                out_f.write(original_image_bytes)
                bg_status = 'with-background' if has_bg else 'without-background'
                logging.info(f"Image {file} (SSIM value: {ssim_value:.4f}) copied to {bg_status} directory.\n")

            # Generate normalized version if enabled
            if self.enable_background_normalization:
                normalized_image, normalized_image_bytes = self.image_editing_model.edit(
                    self.background_normalization_prompt,
                    original_image_bytes
                )
                if not normalized_image:
                    logging.error(f"Background normalization failed for image: {file}, skipping.\n")
                    continue
                # Save normalized image to appropriate directory
                dst_normalized_path = os.path.join(self.dst_dir_bg_normalized if has_bg else self.dst_dir_no_bg_normalized, file)
                with open(dst_normalized_path, 'wb') as norm_f:
                    norm_f.write(normalized_image_bytes)
                    bg_status = 'with-background-normalized' if has_bg else 'without-background-normalized'
                    logging.info(f"Image {file} normalized and saved to {bg_status} directory.\n")
        
        # Generate preview for with-background images
        self._generate_preview(
            image_dir=self.dst_dir_bg,
            normalized_dir=self.dst_dir_bg_normalized,
            dst_file=os.path.join(src_dir, "preview_with_bg.png"),
            title="With Background Subset"
        )
        # Generate preview for without-background images
        self._generate_preview(
            image_dir=self.dst_dir_no_bg,
            normalized_dir=self.dst_dir_no_bg_normalized,
            dst_file=os.path.join(src_dir, "preview_without_bg.png"),
            title="Without Background Subset"
        )