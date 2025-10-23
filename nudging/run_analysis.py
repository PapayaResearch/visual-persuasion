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
    print_config(cfg)

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(
        "analysis",
        cfg.provider.name,
        cfg.nudge.evaluator_model.api_call.model
    )
    
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

    # Set up provider API key
    try:
        with open(cfg.provider.key) as infile:
            os.environ[cfg.provider.key_name] = infile.read().strip()
        logging.info(f"Set API key from {cfg.provider.key}\n")
    except FileNotFoundError:
        logging.error(f"API key file not found at: {cfg.provider.key}\n")
        return

    analysis_dir = cfg.general.analysis_dir
    
    # Check if the directory exists and is valid
    if not os.path.isdir(analysis_dir):
        logging.error(f"Analysis directory not found: {analysis_dir}\n")
        return
    
    logging.info(f"Starting analysis on results in: {analysis_dir}\n")

    # Create analysis pipeline
    analysis_pipeline = hydra.utils.instantiate(cfg.analyze)

    # Run analysis
    analysis_results_dir = analysis_pipeline.run(analysis_dir)
    logging.info(f"Analysis completed: {analysis_results_dir}\n")

if __name__ == "__main__":
    main()