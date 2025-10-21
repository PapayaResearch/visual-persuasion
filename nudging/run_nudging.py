import logging
import os
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from config import Config
from shared.misc import print_config

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
        "nudging",
        cfg.provider.name,
        cfg.nudge.evaluator_model.api_call.model
    )
    results_dir = os.path.join(cfg.logging.results_dir, base_dir, current_date)
    
    # Set up logging
    log_dir = cfg.logging.log_dir
    log_file = os.path.join(log_dir, base_dir, current_date + ".log")
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
    
    # Create the results directory
    os.makedirs(results_dir, exist_ok=True)
    # Save config to output directories
    with open(os.path.join(results_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)
    with open(os.path.join(os.path.dirname(log_file), "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Set up provider API key
    try:
        with open(cfg.provider.key) as infile:
            os.environ[cfg.provider.key_name] = infile.read().strip()
        logging.info(f"Set API key from {cfg.provider.key}\n")
    except FileNotFoundError:
        logging.error(f"API key file not found at: {cfg.provider.key}\n")
        return

    # Set up image editing provider API key
    try:
        with open(cfg.provider_image.key) as infile:
            os.environ[cfg.provider_image.key_name] = infile.read().strip()
        logging.info(f"Set API key from {cfg.provider_image.key}")
    except FileNotFoundError:
        logging.error(f"API key file not found at: {cfg.provider_image.key}")
        return

    # Get list of images from data directory
    data_dir = cfg.general.data_dir
    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found at: {data_dir}\n")
        return
    
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                    if os.path.isfile(os.path.join(data_dir, f))]
    if not image_paths:
        logging.warning(f"No images found in data directory: {data_dir}\n")
        return

    # Instantiate the entire VisualNudge pipeline
    logging.info("Instantiating VisualNudge pipeline...\n")
    nudge_pipeline = hydra.utils.instantiate(cfg.nudge)
    logging.info("Pipeline instantiated\n")
        
    logging.info(f"Starting visual nudge run with {len(image_paths)} image(s) from {data_dir}\n")

    # Pass runtime-specific parameters to the run method
    nudge_results_dir = nudge_pipeline.run(image_paths, results_dir)
    logging.info(f"Visual nudge run completed: {nudge_results_dir}\n")

if __name__ == "__main__":
    main()
