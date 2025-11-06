import logging
import os
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from dotenv import load_dotenv
from config import Config
from utils.misc import print_config

# Load environment variables from .env
load_dotenv()

# Initialize Hydra config store
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    # Load and print configuration
    OmegaConf.resolve(cfg)
    print_config(cfg)

    # Set up logging
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(
        "logs",
        "standardization",
        cfg.dataset.name,
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
    src_dir = cfg.general.src_dir
    dst_dir = os.path.join(cfg.general.dst_dir, cfg.dataset.name)

    # Load images from the source directory
    sampled_images = []
    for filename in os.listdir(src_dir):
        if os.path.isfile(os.path.join(src_dir, filename)) and filename.lower().endswith('.jpg'):
            class_name = filename.split('_')[0]
            src_image_path = os.path.join(src_dir, filename)
            sampled_images.append((src_image_path, filename, class_name))

    # Standardize images and save to destination
    standardizer = hydra.utils.instantiate(cfg.image_standardizer)
    standardizer.standardize_images(sampled_images, dst_dir, cfg.general.max_workers)

    logging.info(f"Image standardization completed.\nResults saved to: {dst_dir}")

if __name__ == "__main__":
    main()
