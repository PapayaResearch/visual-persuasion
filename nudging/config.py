from dataclasses import dataclass

#######################
# API Settings
#######################

@dataclass
class ApiCall:
    # Hydra target for API call function
    _target_: str
    # Model name for API calls
    model: str
    # Delay before API calls to avoid rate limits
    delay: float
    # Sampling temperature for responses
    temperature: float
    # Maximum tokens in API response
    max_tokens: int
    # Reasoning effort for API calls
    reasoning_effort: str
    # Additional parameters to drop for specific models
    additional_drop_params: list

#######################
# Schema Settings
#######################

@dataclass
class SchemaFactory:
    # Hydra target for schema factory function
    _target_: str
    # Field descriptions for the schema (as kwargs)
    # These will be passed to create_*_schema functions

#######################
# Model Components
#######################

@dataclass
class ImageModel:
    # Hydra target for image editing model class
    _target_: str
    # Additional model-specific parameters (from model configs)
    # These will be filled in by the model-specific YAML files

@dataclass
class LanguageModel:
    # Hydra target for language model class
    _target_: str
    # System prompt for the task
    system_prompt: str
    # Input schema factory configuration
    input_schema: SchemaFactory
    # Output schema factory configuration
    output_schema: SchemaFactory
    # API call configuration
    api_call: ApiCall
    # Enable JSON schema validation for models that don't natively support it
    enable_json_schema_validation: bool = True

#######################
# Strategy Settings
#######################

@dataclass
class Strategy:
    # Enable prompt optimization pipeline (disable for zero-shot testing)
    enable_optimization: bool
    # Enable tournament mode (keep track of the last chosen image instead of the previous image)
    enable_tournament_mode: bool
    # Save best prompts instead of the best images in tournament mode (regenerates images for every iteration)
    save_best_prompts: bool

#######################
# Main Pipeline
#######################

@dataclass
class VisualNudge:
    # Hydra target for main pipeline class
    _target_: str
    # Total number of iterations to run per image
    iterations: int
    # Enable previous image context (the last edited image) during editing
    enable_editing_context: bool
    # Additional prompt for editing context
    editing_context_prompt: str
    # Enable prompt optimization pipeline (disable for zero-shot testing)
    enable_optimization: bool
    # Enable tournament mode (keep track of the last chosen image instead of the previous image)
    enable_tournament_mode: bool
    # Save best prompts instead of the best images in tournament mode (regenerates images for every iteration)
    save_best_prompts: bool
    # Initial prompt for image editing
    initial_prompt: str
    # Additional prompt to retain background state during editing
    background_state_prompt: str
    # Image editing model configuration
    image_editing_model: ImageModel
    # Prompt for the evaluator model
    evaluator_prompt: str
    # Evaluator language model configuration
    evaluator_model: LanguageModel
    # Loss model configuration
    loss_model: LanguageModel
    # Optimizer model configuration
    optimizer_model: LanguageModel

#######################
# Evaluation Pipeline
#######################

@dataclass
class Evaluate:
    # Hydra target for evaluation pipeline class
    _target_: str
    # Prompt for the evaluator model
    evaluator_prompt: str
    # Evaluator model configuration
    evaluator_model: LanguageModel

#######################
# Analysis Pipeline
#######################

@dataclass
class Analyze:
    # Hydra target class for the analysis pipeline
    _target_: str
    # Number of preview images to generate
    num_previews: 5

#######################
# General Settings
#######################

@dataclass
class General:
    # Directory containing the images to be tested
    data_dir: str
    # Total number of iterations to run per image
    iterations: int
    # Enable previous image context during editing
    enable_editing_context: bool
    # Directory to evaluate (directory with images used for nudging)
    eval_dir: str
    # Directory to analyze (directory with evaluation log files)
    analysis_dir: str

#######################
# Logging Settings
#######################

@dataclass
class Logging:
    # Base directory for writing log files
    log_dir: str
    # Base directory for writing results (images, configs)
    results_dir: str

######################
# Main Config
######################

@dataclass
class Config:
    # LLM API call configuration
    llm: ApiCall
    # Image editor model configuration
    editor: ImageModel
    # Main pipeline configuration
    nudge: VisualNudge
    # Evaluation pipeline configuration
    evaluate: Evaluate
    # Analysis pipeline configuration
    analyze: Analyze
    # Strategy configuration
    strategy: Strategy
    # General experiment settings
    general: General
    # Logging configuration
    logging: Logging