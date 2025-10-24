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

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(
        "evaluation",
        cfg.evaluate.evaluator_model.api_call.model
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

    # Get list of images from data directory
    data_dir = cfg.general.eval_dir
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if os.path.isfile(os.path.join(data_dir, f))
                    and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Create the results directory
    results_dir = os.path.join(
        data_dir,
        "evaluation",
        cfg.evaluate.evaluator_model.api_call.model,
        current_date
    )
    os.makedirs(results_dir, exist_ok=True)
    # Save config to output directories
    with open(os.path.join(results_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)
    with open(os.path.join(os.path.dirname(log_file), "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Create evaluation pipeline
    eval_pipeline = hydra.utils.instantiate(cfg.evaluate)

    # Run evaluation
    logging.info(f"Starting evaluation on results in: {data_dir}\n")
    eval_pipeline.run(image_paths, results_dir)

    logging.info(f"Evaluation completed: {results_dir}\n")

if __name__ == "__main__":
    main()
