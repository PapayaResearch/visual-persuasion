import logging
import os
import hydra
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from config import Config
from utils.misc import print_config

# Initialize Hydra config store
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load and print configuration
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Create results and log directories
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(
        cfg.provider.name,
        cfg.visual_nudge.evaluator_model.model
    )
    results_dir = os.path.join(cfg.logging.results_dir, base_dir, current_date)
    log_dir = os.path.join(cfg.logging.log_dir, base_dir)
    log_file = os.path.join(log_dir, current_date + ".log")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save config to output directories
    with open(os.path.join(results_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)
    with open(os.path.join(log_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Set up logging to file and console
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # Set up provider API key
    try:
        with open(cfg.provider.key) as infile:
            os.environ[cfg.provider.key_name] = infile.read().strip()
        logging.info(f"Set API key from {cfg.provider.key}")
    except FileNotFoundError:
        logging.error(f"API key file not found at: {cfg.provider.key}")
        return

    # Get list of images from data directory
    data_dir = cfg.general.data_dir
    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found at: {data_dir}")
        return
    
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    if not image_paths:
        logging.warning(f"No images found in data directory: {data_dir}")
        return

    # Instantiate the entire VisualNudge pipeline
    logging.info("Instantiating VisualNudge pipeline...")
    nudge = hydra.utils.instantiate(cfg.visual_nudge)
    logging.info("Pipeline instantiated.")
        
    logging.info(f"Starting visual nudge run with {len(image_paths)} image(s) from {data_dir}")
    # Pass runtime-specific parameters to the run method
    nudge.run(image_paths=image_paths, results_dir=results_dir)
    logging.info("Visual nudge run completed")


if __name__ == "__main__":
    main()