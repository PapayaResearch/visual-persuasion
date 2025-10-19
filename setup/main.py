import logging
import os
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from config import Config
from utils.misc import print_config

# Initialize Hydra config store
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    # Load and print configuration
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(
        cfg.dataset.name,
        cfg.provider.name,
        cfg.general.model
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
    
    # Set up provider API key
    try:
        with open(cfg.provider.key) as infile:
            os.environ[cfg.provider.key_name] = infile.read().strip()
        logging.info(f"Set API key from {cfg.provider.key}")
    except FileNotFoundError:
        logging.error(f"API key file not found at: {cfg.provider.key}")
        return
    
    # Set up image editing provider API key
    try:
        with open(cfg.provider_image.key) as infile:
            os.environ[cfg.provider_image.key_name] = infile.read().strip()
        logging.info(f"Set API key from {cfg.provider_image.key}")
    except FileNotFoundError:
        logging.error(f"API key file not found at: {cfg.provider_image.key}")
        return

    # Check if the source directory exists
    src_path = os.path.join(cfg.general.src_dir, cfg.dataset.subfolder)
    if not os.path.isdir(src_path):
        logging.error(f"Source directory not found: {src_path}")
        return
    
    # Check destination directory and modify if it already contains images
    original_dst_dir = os.path.join(cfg.general.dst_dir, cfg.dataset.name)
    final_dst_dir = original_dst_dir
    
    if os.path.exists(original_dst_dir):
        # Check if the directory contains any image files in immediate subfolders
        existing_files = []
        for item in os.listdir(original_dst_dir):
            item_path = os.path.join(original_dst_dir, item)
            if os.path.isdir(item_path):
                for f in os.listdir(item_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                        existing_files.append(os.path.join(item_path, f))
        
        if existing_files:
            final_dst_dir = original_dst_dir + '_' + current_date
            logging.info(f"Destination directory {original_dst_dir} already exists with {len(existing_files)} images. Using modified destination: {final_dst_dir}")
        else:
            logging.info(f"Destination directory {original_dst_dir} exists but is empty. Using original destination.")
    
    # Get all folders in the source directory with absolute paths (strategy-independent)
    all_folders = [os.path.join(src_path, f) for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f))]
    if not all_folders:
        logging.error(f"No folders found in source directory: {src_path}")
        return
    
    logging.info(f"Starting dataset creation for: {cfg.dataset.name}")
    logging.info(f"Found {len(all_folders)} folders in source directory: {src_path}")
    
    # Create strategy instance and create dataset
    strategy = hydra.utils.instantiate(cfg.strategy)
    strategy.create_dataset(all_folders, final_dst_dir)

    # Enhance image quality if required
    if cfg.general.enhance_image_quality:
        enhancer = hydra.utils.instantiate(cfg.image_enhancer)
        enhancer.enhance_images(final_dst_dir)

    # Split the dataset by background if required
    if cfg.general.split_by_background:
        background_processor = hydra.utils.instantiate(cfg.background_processor)
        background_processor.split_by_background(final_dst_dir)
    
    logging.info(f"Dataset creation completed.\nResults saved to: {final_dst_dir}")

if __name__ == "__main__":
    main()