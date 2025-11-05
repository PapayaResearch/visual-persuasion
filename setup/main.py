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

    # Get all folders in the source directory with absolute paths (strategy-independent)
    all_folders = [
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, f))
    ]

    # Sample images using strategy
    strategy = hydra.utils.instantiate(cfg.strategy)
    sampled_images = strategy.sample_images(all_folders)

    # Enhance sampled images and save to destination
    enhancer = hydra.utils.instantiate(cfg.image_enhancer)
    enhancer.enhance_images(sampled_images, dst_dir, cfg.general.max_workers)

    logging.info(f"Dataset creation completed.\nResults saved to: {dst_dir}")

if __name__ == "__main__":
    main()
