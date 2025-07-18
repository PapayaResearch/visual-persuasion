import pandas as pd
import logging
import os
import hydra
import litellm
from datetime import datetime
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from config import Config
from utils.misc import print_config

config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    ##############################################
    # Check compatibility with LiteLLM
    ##############################################

    assert litellm.supports_function_calling(model=cfg.general.model) == True

    ##############################################
    # Create results and log directories
    ##############################################

    current_date = datetime.now().strftime("%a-%b-%d-%Y_%I-%M-%S%p")

    base_dir = os.path.join(
        cfg.nudge.name,
        cfg.provider.name,
        cfg.general.model
    )

    results_dir = os.path.join(cfg.logging.results_dir, base_dir, current_date)
    results_file = os.path.join(results_dir, current_date + ".csv")

    log_dir = os.path.join(cfg.logging.log_dir, base_dir)
    log_file = os.path.join(log_dir, current_date + ".log")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "cfg.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)
    with open(os.path.join(log_dir, "cfg.yaml"), "w") as outfile:
        outfile.write(cfg_yaml)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,  # You can change this to DEBUG, ERROR, etc.
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ##############################################
    # Set up provider
    ##############################################

    with open(cfg.provider.key) as infile:
        os.environ[cfg.provider.key_name] = infile.read().strip()

    ##############################################
    # Instantiate nudge
    ##############################################

    nudge = instantiate(cfg.nudge.nudge)

    ##############################################
    # Check compatibility with LiteLLM
    ##############################################

    assert litellm.supports_function_calling(model=nudge.eval_model) == True

    ##############################################
    # Run simulation
    ##############################################


if __name__ == "__main__":
    main()