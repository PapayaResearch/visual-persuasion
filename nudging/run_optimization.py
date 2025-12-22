import logging
import os
import shutil
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

    # Instantiate the optimization pipeline
    optimization_pipeline = hydra.utils.instantiate(cfg.optimization)
    
    # Get the target parameter for directory naming
    parameter = cfg.optimization.parameter
    
    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build directory path: optimization/{parameter}/timestamp
    base_dir = os.path.join("optimization", parameter.lower())

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

        if existing_dirs:
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
    logging.info(f"Target parameter for optimization: {parameter}")

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

    # Run the optimization pipeline
    logging.info(f"Starting parametric optimization with {len(image_paths)} image(s) from {data_dir}")
    logging.info(f"Optimizing for parameter: {parameter}\n")
    
    results = optimization_pipeline.run(image_paths, results_dir, cfg.general.max_workers)

    logging.info(f"Optimization run completed: {results_dir}")
    logging.info(f"Comparable pairs found: {len(results.get('comparable_pairs', []))}")
    logging.info(f"Competition results: {len(results.get('competition_results', []))} pairs\n")

if __name__ == "__main__":
    main()
