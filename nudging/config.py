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
# Competition Pipeline
#######################

@dataclass
class Competition:
    # Threshold for considering images "comparable"
    comparability_threshold: float
    # Regex pattern to extract category from filename
    category_pattern: str
    # Thresholds for equilibrium detection
    equilibrium_threshold: float
    # Minimum rounds to run per image pair
    min_rounds_before_equilibrium: int
    # Maximum rounds to run per image pair
    max_rounds_per_pair: int
    # Tie-breaking strategy: "first", "second", "random"
    tie_breaking_strategy: str
    # Task-specific prompts
    base_prior: str
    evaluator_system_prompt: str
    evaluator_reason_description: str
    optimizer_system_prompt: str
    proposer_system_prompt: str
    selector_system_prompt: str
    # List of judge prompts for multi-judge evaluation
    judge_prompts: list

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
    # Valid statuses for image filenames
    valid_statuses: list

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
    llm: str
    # Image editor model configuration
    editor: str
    # Strategy configuration
    strategy: str
    # Evaluation mode configuration
    evaluate: str
    # Priors pipeline configuration
    priors: Priors
    # Interpretation pipeline configuration
    interp: Interp
    # Competition pipeline configuration
    competition: Competition
    # General experiment settings
    general: General
    # Logging configuration
    logging: Logging
