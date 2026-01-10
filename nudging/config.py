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
# Priors Pipeline
#######################

@dataclass
class Priors:
    # Hydra target for priors pipeline class
    _target_: str
    # Threshold for considering a pair of images comparable
    comparability_threshold: float
    # List of judge prompts for multi-judge evaluation
    judge_prompts: list
    # Evaluator model configuration
    evaluator_model: LanguageModel
    # Regex pattern to extract category from filename
    category_pattern: str

#######################
# Interpretation Pipeline
#######################

@dataclass
class Interp:
    # Hydra target class for the interpretation pipeline
    _target_: str
    # Directory containing zero-shot results to interpret
    results_dir: str
    # Prompt for the difference detector model
    difference_prompt: str
    # Difference detector model configuration
    difference_detector_model: LanguageModel
    # Prompt for the theme summarizer model
    theme_prompt: str
    # Theme summarizer model configuration
    theme_summarizer_model: LanguageModel

#######################
# General Settings
#######################

@dataclass
class General:
    # Directory containing the images to be tested
    data_dir: str
    # Maximum number of parallel workers for processing images
    max_workers: int
    # Resume from latest run if some results exist
    resume: bool

#######################
# Logging Settings
#######################

@dataclass
class Logging:
    # Base directory for writing log files
    log_dir: str
    # Base directory for writing results (images, configs)
    results_dir: str
    # Also log to console
    console: bool

######################
# Main Config
######################

@dataclass
class Config:
    # LLM API call configuration
    llm: ApiCall
    # Image editor model configuration
    editor: ImageModel
    # Strategy configuration
    strategy: str
    # Evaluation mode configuration
    evaluate: str
    # Priors pipeline configuration
    priors: Priors
    # Interpretation pipeline configuration
    interp: Interp
    # General experiment settings
    general: General
    # Logging configuration
    logging: Logging
