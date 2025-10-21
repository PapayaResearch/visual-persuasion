import random
import logging
import os
from typing import List
from shared.wrappers import LanguageModel


class RandomSampling:
    def __init__(
        self,
        num_folders: int,
        num_process_per_folder: int,
    ):
        self.num_folders = num_folders
        self.num_process_per_folder = num_process_per_folder

    def create_dataset(self, all_folders: List[str], dst_dir: str):
        self.dst_dir = os.path.join(dst_dir, "base")
        os.makedirs(self.dst_dir, exist_ok=True)
        logging.info(f"Using Random Sampling strategy\n")

        # Handle -1 case (process all folders)
        if self.num_folders == -1:
            self.num_folders = len(all_folders)
            logging.info(f"num_folders set to -1, processing all {self.num_folders} folders\n")
        
        # Check if requested folders exceed available folders
        if self.num_folders > len(all_folders):
            logging.warning(f"Requested number of folders ({self.num_folders}) exceeds available folders ({len(all_folders)}). Adjusting to available count.\n")
            self.num_folders = len(all_folders)

        random.shuffle(all_folders)
        selected_folders = all_folders[:self.num_folders]
        logging.info(f"Selected {len(selected_folders)} folders for processing\n")

        # Process each selected folder
        for folder_path in selected_folders:
            folder_name = os.path.basename(folder_path)
            logging.info(f"Processing folder: {folder_name}\n")

            # List all images in the folder and randomly select the specified number
            images = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
            
            # Handle -1 case for num_process_per_folder
            num_to_process = len(images) if self.num_process_per_folder == -1 else self.num_process_per_folder
            
            if len(images) < num_to_process:
                logging.warning(f"Requested number of images ({num_to_process}) exceeds available images ({len(images)}) in folder {folder_name}. Adjusting to available count.\n")
                selected_images = images
            else:
                random.shuffle(images)
                selected_images = images[:num_to_process]

            # Copy selected images to destination directory
            for image in selected_images:
                dst_image_path = os.path.join(self.dst_dir, f"{folder_name}_{image}")
                with open(os.path.join(folder_path, image), "rb") as src_f:
                    with open(dst_image_path, "wb") as dst_f:
                        dst_f.write(src_f.read())
                logging.info(f"Copied image {image} from folder {folder_name}\n")


class VLMFiltering:
    def __init__(
        self,
        num_folders: int,
        num_evaluate_per_folder: int,
        num_process_per_folder: int,
        evaluator_model: LanguageModel,
    ):
        self.num_folders = num_folders
        self.num_evaluate_per_folder = num_evaluate_per_folder
        self.num_process_per_folder = num_process_per_folder
        self.evaluator_model = evaluator_model

    def create_dataset(self, all_folders: List[str], dst_dir: str):
        self.dst_dir = os.path.join(dst_dir, "base")
        os.makedirs(self.dst_dir, exist_ok=True)
        logging.info(f"Using VLM Filtering strategy\n")

        # Handle -1 case (process all folders)
        if self.num_folders == -1:
            self.num_folders = len(all_folders)
            logging.info(f"num_folders set to -1, processing all {self.num_folders} folders\n")
        
        # Check if requested folders exceed available folders
        if self.num_folders > len(all_folders):
            logging.warning(f"Requested number of folders ({self.num_folders}) exceeds available folders ({len(all_folders)}). Adjusting to available count.\n")
            self.num_folders = len(all_folders)

        random.shuffle(all_folders)
        selected_folders = all_folders[:self.num_folders]
        logging.info(f"Selected {len(selected_folders)} folders for processing\n")

        # Process each selected folder
        for folder_path in selected_folders:
            folder_name = os.path.basename(folder_path)
            logging.info(f"Processing folder: {folder_name}\n")

            # List all images in the folder and randomly select the specified number for evaluation
            images = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
            
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
                
                if not chunk_images:
                    continue
                
                logging.info(f"Evaluating chunk {i+1}/{actual_num_process} with {len(chunk_images)} images in folder {folder_name}\n")

                # Evaluate the chunk images and choose the best one
                image_bytes_list = []
                for img in chunk_images:
                    with open(os.path.join(folder_path, img), "rb") as f:
                        image_bytes_list.append(f.read())
                
                response = self.evaluator_model.get_response(
                    task="Select the best image from the provided images.",
                    images=image_bytes_list
                )

                try:
                    best_index = int(response.choice) - 1
                    if not best_index in range(len(images)):
                        logging.error(f"Evaluator returned invalid index: {response}\nDefaulting to first image.")
                        best_index = 0  # Default to first image on error
                except ValueError:
                    logging.error(f"Evaluator response parsing failed: {response}\nDefaulting to first image.")
                    best_index = 0  # Default to first image on error
                
                best_image = chunk_images[best_index]
                logging.info(f"Selected best image {best_image} (index {best_index}) from chunk {i+1} in folder {folder_name}\n")
                
                # Copy the best image to the destination directory
                dst_image_path = os.path.join(self.dst_dir, f"{folder_name}_{best_image}")
                with open(os.path.join(folder_path, best_image), "rb") as src_f:
                    with open(dst_image_path, "wb") as dst_f:
                        dst_f.write(src_f.read())
                
                start_idx = end_idx