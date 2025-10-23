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

    eval_dir = cfg.general.eval_dir

    logging.info(f"Starting evaluation on results in: {eval_dir}\n")

    # Create evaluation pipeline
    eval_pipeline = hydra.utils.instantiate(cfg.evaluate)

    # Run evaluation
    eval_results_dir = eval_pipeline.run(eval_dir, cfg.evaluate.evaluator_model.api_call.model)
    logging.info(f"Evaluation completed: {eval_results_dir}\n")

if __name__ == "__main__":
    main()
