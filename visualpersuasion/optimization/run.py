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

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    # Load and print configuration
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Instantiate the nudging pipeline from strategy config
    nudge_pipeline = hydra.utils.instantiate(cfg.strategy)

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build directory path: strategy/model/timestamp (model only if strategy uses one)
    base_dir = nudge_pipeline.name

    results_dir = os.path.join(cfg.logging.results_dir, base_dir, current_date)

    if cfg.general.resume:
        # Find most recent results directory first
        existing_dirs = []
        results_pdir = os.path.dirname(results_dir)
        if os.path.exists(results_pdir):
            for d in os.listdir(results_pdir):
                dir_path = os.path.join(results_pdir, d)
                if os.path.isdir(dir_path):
                    existing_dirs.append(d)

        latest_dir = max(existing_dirs)
        print(f"Found existing results directory: {latest_dir}")
        previous_results_path = os.path.join(results_pdir, latest_dir)
        shutil.copytree(previous_results_path, results_dir)
        print(f"Resuming from previous results at: {previous_results_path}")


    # Set up logging
    log_dir = cfg.logging.log_dir
    log_file = os.path.join(log_dir, base_dir, current_date + ".log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging_handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if cfg.logging.console:
        logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        handlers=logging_handlers,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    print(f"Logging to: {log_file}")
    logging.info(f"Logging to: {log_file}")

    # Create the results directory
    os.makedirs(results_dir, exist_ok=True)
    # Save config to output directories
    with open(os.path.join(results_dir, "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)
    with open(os.path.join(os.path.dirname(log_file), "config.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    # Get list of images from data directory
    data_dir = cfg.general.data_dir
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if os.path.isfile(os.path.join(data_dir, f))
                    and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Run the nudging pipeline
    logging.info(f"Starting nudging run with {len(image_paths)} image(s) from {data_dir}\n")
    nudge_pipeline.run(image_paths, results_dir, cfg.general.max_workers)

    logging.info(f"Nudging run completed: {results_dir}\n")

if __name__ == "__main__":
    main()
