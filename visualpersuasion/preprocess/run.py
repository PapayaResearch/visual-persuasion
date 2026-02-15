import logging
import os
import shutil
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from dotenv import load_dotenv
from visualpersuasion.config import Config
from visualpersuasion.utils.misc import print_config

# Load environment variables from .env
load_dotenv()

# Initialize Hydra config store
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Config):
    # Load and print configuration
    OmegaConf.resolve(cfg)
    print_config(cfg)

    # Set up logging
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(
        cfg.logging.log_dir,
        "preprocess",
        current_date + ".log"
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Logging to: {log_file}")

    # Setup paths
    src_dir = cfg.general.data_dir
    results_dir = cfg.preprocess.results_dir

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Get all folders in the source directory with absolute paths
    all_folders = [
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, f))
    ]

    # Instantiate the preprocessing pipeline
    preprocess_pipeline = hydra.utils.instantiate(cfg.preprocess)

    # Check preprocessing mode
    if cfg.preprocess.name == 'random-sampling':
        logging.info("Running random sampling...")
        sampled_images = preprocess_pipeline.sample_images(all_folders)

        for src_image_path, dst_filename, category in sampled_images:
            dst_path = os.path.join(results_dir, dst_filename)
            shutil.copy2(src_image_path, dst_path)

        logging.info(f"Sampling completed. {len(sampled_images)} images saved to: {results_dir}")

    elif cfg.preprocess.name == 'enhance':
        logging.info("Running image enhancement...")

        all_images = []
        for folder_path in all_folders:
            folder_name = os.path.basename(folder_path)
            images = [
                img for img in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, img)) and
                img.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            for image in images:
                src_image_path = os.path.join(folder_path, image)
                dst_filename = f"{folder_name.replace('_', '')}_{image.replace('_', '')}"
                all_images.append((src_image_path, dst_filename, folder_name))

        preprocess_pipeline.enhance_images(all_images, results_dir, cfg.general.max_workers)
        logging.info(f"Enhancement completed. Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
