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

    # Determine which pipelines to run based on config
    run_nudge = cfg.general.enable_nudging
    run_evaluate = cfg.general.enable_evaluation
    run_analysis = cfg.general.enable_analysis

    # Create common directories and paths
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(
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
    
    # Create the results directory if needed
    if run_nudge:
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

    # Run the Nudging pipeline if enabled
    if run_nudge:
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

    # Run the Evaluation pipeline if enabled
    if run_evaluate:
        # Determine which directory to evaluate
        if run_nudge:
            # If we just ran the nudging pipeline, evaluate its results
            eval_dir = results_dir
        else:
            # If only evaluation is enabled, use the configured eval_dir
            eval_dir = cfg.general.eval_dir
        
        # Check if the directory exists and is valid
        if not os.path.isdir(eval_dir):
            logging.error(f"Evaluation directory not found: {eval_dir}\n")
            return
        
        logging.info(f"Starting evaluation on results in: {eval_dir}\n")
        
        # Create evaluation pipeline
        eval_pipeline = hydra.utils.instantiate(cfg.evaluate)
        
        # Run evaluation
        eval_results_dir = eval_pipeline.run(eval_dir, cfg.evaluate.evaluator_model.api_call.model)
        logging.info(f"Evaluation completed: {eval_results_dir}\n")

    # Run the Analysis pipeline if enabled
    if run_analysis:
        # Determine which directory to analyze
        if run_evaluate:
            # If we just ran the evaluation pipeline, analyze its results
            analysis_dir = eval_results_dir
        else:
            # If only analysis is enabled, use the configured analysis_dir
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