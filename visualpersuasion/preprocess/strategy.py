import random
import logging
import os
from typing import List
from tqdm import tqdm
from abc import ABC, abstractmethod

class SamplingStrategy(ABC):
    def __init__(self, num_folders: int):
        self.num_folders = num_folders

    def sample_images(self, all_folders: List[str]) -> List[tuple]:
        """
        Main dataset sampling workflow with common logic.
        Returns a list of tuples: (source_image_path, destination_filename, category)
        """
        self.sampled_images = []

        # Handle -1 case (process all folders)
        if self.num_folders == -1:
            self.num_folders = len(all_folders)

        # Check if requested folders exceed available folders
        if self.num_folders > len(all_folders):
            logging.warning(f"Requested number of folders ({self.num_folders}) exceeds available folders ({len(all_folders)}). Adjusting to available count.\n")
            self.num_folders = len(all_folders)

        # Select random folders
        random.shuffle(all_folders)
        selected_folders = all_folders[:self.num_folders]

        # Process each selected folder using strategy-specific logic
        for folder_path in tqdm(selected_folders, desc="Sampling images", unit="folder"):
            folder_name = os.path.basename(folder_path)

            # Get all images in the folder
            images = [img for img in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, img)) and
                      img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Call strategy-specific processing
            self.process_folder(folder_path, folder_name, images)

        return self.sampled_images

    @abstractmethod
    def process_folder(self, folder_path: str, folder_name: str, images: List[str]):
        """Strategy-specific logic for processing a single folder."""
        pass


class RandomSampling(SamplingStrategy):
    def __init__(
        self,
        num_folders: int,
        num_process_per_folder: int,
    ):
        super().__init__(num_folders)
        self.num_process_per_folder = num_process_per_folder

    def process_folder(self, folder_path: str, folder_name: str, images: List[str]):
        """Randomly select images from the folder."""
        # Handle -1 case for num_process_per_folder
        num_to_process = len(images) if self.num_process_per_folder == -1 else self.num_process_per_folder

        if len(images) < num_to_process:
            logging.warning(f"Requested number of images ({num_to_process}) exceeds available images ({len(images)}) in folder {folder_name}. Adjusting to available count.\n")
            selected_images = images
        else:
            random.shuffle(images)
            selected_images = images[:num_to_process]

        # Add selected images to sampled list
        for image in selected_images:
            src_image_path = os.path.join(folder_path, image)
            dst_filename = f"{folder_name.replace('_', '')}_{image.replace('_', '')}"
            self.sampled_images.append((src_image_path, dst_filename, folder_name))
