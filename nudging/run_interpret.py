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
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Create output directory
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(cfg.logging.results_dir, "interpretation", current_date)
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_dir = cfg.logging.log_dir
    log_file = os.path.join(log_dir, "interpretation", current_date + ".log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    print(f"Logging to: {log_file}")
    logging.info(f"Logging to: {log_file}")

    # Save config to output directory
    with open(os.path.join(output_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)
    with open(os.path.join(os.path.dirname(log_file), "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Instantiate the interpreter pipeline
    interpreter = hydra.utils.instantiate(cfg.interpret)

    # Run the interpreter
    logging.info(f"Starting interpretation of results from {cfg.interpret.results_dir}\n")
    interpreter.run(output_dir, cfg.general.max_workers)

    logging.info(f"Interpretation completed: {output_dir}\n")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
