import random
import logging
import os
from typing import List
from tqdm import tqdm
from utils.wrappers import LanguageModel
from abc import ABC, abstractmethod

class SamplingStrategy(ABC):
    def __init__(self, num_folders: int):
        self.num_folders = num_folders
        self.dst_dir = None

    def create_dataset(self, all_folders: List[str], dst_dir: str):
        """Main dataset creation workflow with common logic."""
        self.dst_dir = os.path.join(dst_dir, "base")
        os.makedirs(self.dst_dir, exist_ok=True)

        logging.info(f"Using {self.__class__.__name__} strategy\n")

        # Handle -1 case (process all folders)
        if self.num_folders == -1:
            self.num_folders = len(all_folders)
            logging.info(f"num_folders set to -1, processing all {self.num_folders} folders\n")

        # Check if requested folders exceed available folders
        if self.num_folders > len(all_folders):
            logging.warning(f"Requested number of folders ({self.num_folders}) exceeds available folders ({len(all_folders)}). Adjusting to available count.\n")
            self.num_folders = len(all_folders)

        # Select random folders
        random.shuffle(all_folders)
        selected_folders = all_folders[:self.num_folders]
        logging.info(f"Selected {len(selected_folders)} folders for processing\n")

        # Process each selected folder using strategy-specific logic
        for folder_path in tqdm(selected_folders, desc="Processing folders", unit="folder"):
            folder_name = os.path.basename(folder_path)

            # Get all images in the folder
            images = [img for img in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, img))]

            # Call strategy-specific processing
            self.process_folder(folder_path, folder_name, images)

    @abstractmethod
    def process_folder(self, folder_path: str, folder_name: str, images: List[str]):
        """Strategy-specific logic for processing a single folder."""
        pass

    def copy_image(self, folder_path: str, folder_name: str, image: str):
        """Copy a single image to the destination directory."""
        dst_image_path = os.path.join(self.dst_dir, f"{folder_name.replace('_', '')}_{image.replace('_', '')}")
        with open(os.path.join(folder_path, image), "rb") as src_f:
            with open(dst_image_path, "wb") as dst_f:
                dst_f.write(src_f.read())


class RandomSampling(SamplingStrategy):
    def __init__(
        self,
        num_folders: int,
        num_process_per_folder: int,
    ):
        super().__init__(num_folders)
        self.num_process_per_folder = num_process_per_folder

    def process_folder(self, folder_path: str, folder_name: str, images: List[str]):
        """Randomly select and copy images from the folder."""
        # Handle -1 case for num_process_per_folder
        num_to_process = len(images) if self.num_process_per_folder == -1 else self.num_process_per_folder

        if len(images) < num_to_process:
            logging.warning(f"Requested number of images ({num_to_process}) exceeds available images ({len(images)}) in folder {folder_name}. Adjusting to available count.\n")
            selected_images = images
        else:
            random.shuffle(images)
            selected_images = images[:num_to_process]

        # Copy selected images
        for image in selected_images:
            self.copy_image(folder_path, folder_name, image)


class VLMFiltering(SamplingStrategy):
    def __init__(
        self,
        num_folders: int,
        num_evaluate_per_folder: int,
        num_process_per_folder: int,
        evaluator_prompt: str,
        evaluator_model: LanguageModel,
    ):
        super().__init__(num_folders)
        self.num_evaluate_per_folder = num_evaluate_per_folder
        self.num_process_per_folder = num_process_per_folder
        self.evaluator_prompt = evaluator_prompt
        self.evaluator_model = evaluator_model

    def process_folder(self, folder_path: str, folder_name: str, images: List[str]):
        """Use VLM to select best images from chunks."""
        # Handle -1 case for num_evaluate_per_folder
        if self.num_evaluate_per_folder == -1:
            num_evaluate = len(images)
            logging.info(f"num_evaluate_per_folder set to -1, evaluating all {num_evaluate} images in folder {folder_name}\n")
        else:
            num_evaluate = self.num_evaluate_per_folder

        if len(images) < num_evaluate:
            logging.warning(f"Requested number of images ({num_evaluate}) exceeds available images ({len(images)}) in folder {folder_name}. Adjusting to available count.\n")
            num_evaluate = len(images)

        random.shuffle(images)
        selected_images = images[:num_evaluate]

        # Handle -1 case for num_process_per_folder
        actual_num_process = len(selected_images) if self.num_process_per_folder == -1 else self.num_process_per_folder

        # Split selected images into chunks and process each chunk
        chunk_size = len(selected_images) // actual_num_process
        remainder = len(selected_images) % actual_num_process

        start_idx = 0
        for i in range(actual_num_process):
            # Calculate chunk size (distribute remainder across first chunks)
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size

            # Get current chunk of images
            chunk_images = selected_images[start_idx:end_idx]
            logging.info(f"Evaluating chunk {i+1}/{actual_num_process} with {len(chunk_images)} images in folder {folder_name}\n")

            # Evaluate the chunk images and choose the best one
            image_bytes_list = []
            for img in chunk_images:
                with open(os.path.join(folder_path, img), "rb") as f:
                    image_bytes_list.append(f.read())

            response = self.evaluator_model.get_response(
                task=self.evaluator_prompt,
                images=image_bytes_list
            )

            if not response:
                logging.error(f"Evaluator returned no response, defaulting to first image.\n")
                best_index = 0
            else:
                best_index = int(response.choice) - 1

            best_image = chunk_images[best_index]
            logging.info(f"Selected best image {best_image} (index {best_index}) from chunk {i+1} in folder {folder_name}")

            # Copy the best image to the destination directory
            self.copy_image(folder_path, folder_name, best_image)

            start_idx = end_idx
