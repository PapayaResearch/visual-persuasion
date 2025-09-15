from dataclasses import dataclass
from typing import Any

#######################
# API Settings
#######################

@dataclass
class ApiCall:
    # Hydra target for API call function
    _target_: str
    # Model name for API calls
    model: str
    # Sampling temperature for responses
    temperature: float
    # Maximum tokens in API response
    max_tokens: int
    # Delay before API calls to avoid rate limits
    delay: int

#######################
# Model Components
#######################

@dataclass
class ImageEditingModel:
    # Hydra target for image editing model class
    _target_: str
    # Additional model-specific parameters (from model configs)
    # These will be filled in by the model-specific YAML files

@dataclass
class EvaluatorModel:
    # Hydra target for evaluator model class
    _target_: str
    # System prompt for image comparison task
    system_prompt: str
    # API call configuration
    api_call: ApiCall

@dataclass
class LossModel:
    # Hydra target for loss model class
    _target_: str
    # System prompt for critique generation
    system_prompt: str
    # API call configuration
    api_call: ApiCall

@dataclass
class OptimizerModel:
    # Hydra target for optimizer model class
    _target_: str
    # System prompt for prompt optimization
    system_prompt: str
    # API call configuration
    api_call: ApiCall

#######################
# Main Pipeline
#######################

@dataclass
class VisualNudge:
    # Hydra target for main pipeline class
    _target_: str
    # Enable prompt optimization pipeline (disable for zero-shot testing)
    enable_optimization: bool
    # Enable previous image context (the last edited image)
    enable_image_context: bool
    # Whether to enhance the original image for a better comparison
    enhance_original: bool
    # Number of optimization iterations per image
    iterations: int
    # Initial prompt for image editing
    initial_prompt: str
    # The prompt for enhancing the original image (if enabled)
    enhance_prompt: str
    # Image editing model configuration
    image_editing_model: ImageEditingModel
    # Evaluator model configuration
    evaluator_model: EvaluatorModel
    # Loss model configuration
    loss_model: LossModel
    # Optimizer model configuration
    optimizer_model: OptimizerModel

#######################
# Evaluation Pipeline
#######################

@dataclass
class Evaluate:
    # Hydra target for evaluation pipeline class
    _target_: str
    # Number of images to evaluate (set to -1 to evaluate all)
    num_images: int
    # Whether to evaluate from a customer perspective (true) or an agent's perspective (false)
    use_customer_perspective: bool
    # Whether to enhance the original image before comparison
    enhance_original: bool
    # The prompt for enhancing the original image (if enabled)
    enhance_prompt: str
    # The image editing model used for enhancing the original image (if enabled)
    image_editing_model: ImageEditingModel
    # The prompt for the image comparison task when evaluating from a customer perspective
    customer_prompt: str
    # The prompt for the image comparison task when evaluating from the agent's perspective
    agent_prompt: str
    # Evaluator model configuration
    evaluator_model: EvaluatorModel

#######################
# Provider Settings
#######################

@dataclass
class Provider:
    # Provider name
    name: str
    # Path to API key file
    key: str
    # Environment variable name for API key
    key_name: str

#######################
# General Settings
#######################

@dataclass
class General:
    # Enable the nudging pipeline
    enable_nudging: bool
    # Enable prompt optimization pipeline (disable for zero-shot testing)
    enable_optimization: bool
    # Enable previous image context (the last edited image)
    enable_image_context: bool
    # Total number of iterations to run per image
    iterations: int
    # Whether to enhance the original image for a better comparison
    enhance_original: bool
    # Enable the evaluation pipeline
    enable_evaluation: bool
    # Directory to evaluate (only used when enable_nudging is false)
    eval_dir: str
    # Whether to evaluate from a customer perspective (true) or an agent's perspective (false)
    use_customer_perspective: bool
    # Directory containing input images
    data_dir: str
    # Model for all API calls
    model: str
    # Temperature for all API calls
    temperature: float
    # Max tokens for all API calls
    max_tokens: int
    # Standard delay for API calls
    delay: int

#######################
# Logging Settings
#######################

@dataclass
class Logging:
    # Directory for log files
    log_dir: str
    # Directory for results (images, configs)
    results_dir: str

######################
# Main Config
######################

@dataclass
class Config:
    # Main pipeline configuration
    visual_nudge: VisualNudge
    # Evaluation pipeline configuration
    evaluate: Evaluate
    # API provider configuration
    provider: Provider
    # General experiment settings
    general: General
    # Logging configuration
    logging: Logging