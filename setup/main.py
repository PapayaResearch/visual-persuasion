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

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(
        cfg.dataset.name,
        cfg.llm.model
    )

    # Set up logging
    log_file = os.path.join("logs", base_dir, current_date + ".log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.info(f"Logging to: {log_file}")

    # Always append timestamp to destination directory
    src_path = cfg.general.src_dir
    dst_dir = os.path.join(cfg.general.dst_dir, cfg.dataset.name + '_' + current_date)

    # Get all folders in the source directory with absolute paths (strategy-independent)
    all_folders = [
        os.path.join(src_path, f)
        for f in os.listdir(src_path)
        if os.path.isdir(os.path.join(src_path, f))
    ]

    logging.info(f"Starting dataset creation for: {cfg.dataset.name}")
    logging.info(f"Found {len(all_folders)} folders in source directory: {src_path}")

    # Create strategy instance and create dataset
    strategy = hydra.utils.instantiate(cfg.strategy)
    strategy.create_dataset(all_folders, dst_dir)

    # Enhance image quality if required
    if cfg.general.enhance_image_quality:
        enhancer = hydra.utils.instantiate(cfg.image_enhancer)
        enhancer.enhance_images(dst_dir)

    # Split the dataset by background if required
    if cfg.general.split_by_background:
        background_processor = hydra.utils.instantiate(cfg.background_processor)
        background_processor.split_by_background(dst_dir)

    logging.info(f"Dataset creation completed.\nResults saved to: {dst_dir}")

if __name__ == "__main__":
    main()
