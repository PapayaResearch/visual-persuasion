import random
import logging
import os
from typing import List
from wrappers import EvaluatorModel

class RandomSampling:
    def __init__(
        self,
        num_folders: int,
        num_process_per_folder: int,
    ):
        self.num_folders = num_folders
        self.num_process_per_folder = num_process_per_folder

    def create_dataset(self, all_folders: List[str], dst_dir: str):
        self.dst_dir = dst_dir
        os.makedirs(self.dst_dir, exist_ok=True)
        logging.info(f"Using Random Sampling strategy")

        # Randomly select the specified number of folders
        if len(all_folders) < self.num_folders:
            logging.warning(f"Requested number of folders ({self.num_folders}) exceeds available folders ({len(all_folders)}). Adjusting to available count.")
            self.num_folders = len(all_folders)

        random.shuffle(all_folders)
        selected_folders = all_folders[:self.num_folders]
        logging.info(f"Selected {len(selected_folders)} folders for processing")

        # Process each selected folder
        for folder_path in selected_folders:
            folder_name = os.path.basename(folder_path)
            logging.info(f"Processing folder: {folder_name}")

            # List all images in the folder and randomly select the specified number
            images = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
            if len(images) < self.num_process_per_folder:
                logging.warning(f"Requested number of images ({self.num_process_per_folder}) exceeds available images ({len(images)}) in folder {folder_name}. Adjusting to available count.")
                selected_images = images
            else:
                random.shuffle(images)
                selected_images = images[:self.num_process_per_folder]

            # Copy selected images to destination directory
            for image in selected_images:
                dst_image_path = os.path.join(self.dst_dir, f"{folder_name}_{image}")
                with open(os.path.join(folder_path, image), "rb") as src_f:
                    with open(dst_image_path, "wb") as dst_f:
                        dst_f.write(src_f.read())
                logging.info(f"Copied image {image} from folder {folder_name}")

class VLMFiltering:
    def __init__(
        self,
        num_folders: int,
        num_evaluate_per_folder: int,
        num_process_per_folder: int,
        evaluator_model: EvaluatorModel,
    ):
        self.num_folders = num_folders
        self.num_evaluate_per_folder = num_evaluate_per_folder
        self.num_process_per_folder = num_process_per_folder
        self.evaluator_model = evaluator_model

    def create_dataset(self, all_folders: List[str], dst_dir: str):
        self.dst_dir = dst_dir
        os.makedirs(self.dst_dir, exist_ok=True)
        logging.info(f"Using VLM Filtering strategy")

        # Randomly select the specified number of folders
        if len(all_folders) < self.num_folders:
            logging.warning(f"Requested number of folders ({self.num_folders}) exceeds available folders ({len(all_folders)}). Adjusting to available count.")
            self.num_folders = len(all_folders)

        random.shuffle(all_folders)
        selected_folders = all_folders[:self.num_folders]
        logging.info(f"Selected {len(selected_folders)} folders for processing")

        # Process each selected folder
        for folder_path in selected_folders:
            folder_name = os.path.basename(folder_path)
            logging.info(f"Processing folder: {folder_name}")

            # List all images in the folder and randomly select the specified number for evaluation
            images = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
            if len(images) < self.num_evaluate_per_folder:
                logging.warning(f"Requested number of images ({self.num_evaluate_per_folder}) exceeds available images ({len(images)}) in folder {folder_name}. Adjusting to available count.")
                num_evaluate = len(images)
            else:
                num_evaluate = self.num_evaluate_per_folder

            random.shuffle(images)
            selected_images = images[:num_evaluate]

            # Split selected images into chunks and process each chunk
            chunk_size = len(selected_images) // self.num_process_per_folder
            remainder = len(selected_images) % self.num_process_per_folder
            
            start_idx = 0
            for i in range(self.num_process_per_folder):
                # Calculate chunk size (distribute remainder across first chunks)
                current_chunk_size = chunk_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_chunk_size
                
                # Get current chunk of images
                chunk_images = selected_images[start_idx:end_idx]
                
                if not chunk_images:
                    continue
                
                logging.info(f"Evaluating chunk {i+1}/{self.num_process_per_folder} with {len(chunk_images)} images in folder {folder_name}")

                # Evaluate the chunk images and choose the best one
                image_bytes_list = []
                for img in chunk_images:
                    with open(os.path.join(folder_path, img), "rb") as f:
                        image_bytes_list.append(f.read())
                
                best_index = self.evaluator_model.evaluate(image_bytes_list)
                best_image = chunk_images[best_index]
                logging.info(f"Selected best image {best_image} (index {best_index}) from chunk {i+1} in folder {folder_name}")
                
                # Copy the best image to the destination directory
                dst_image_path = os.path.join(self.dst_dir, f"{folder_name}_{best_image}")
                with open(os.path.join(folder_path, best_image), "rb") as src_f:
                    with open(dst_image_path, "wb") as dst_f:
                        dst_f.write(src_f.read())
                
                start_idx = end_idx