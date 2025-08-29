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
    # Number of optimization iterations per image
    iterations: int
    # Initial prompt for image editing
    initial_prompt: str
    # Image editing model configuration
    image_editing_model: ImageEditingModel
    # Evaluator model configuration
    evaluator_model: EvaluatorModel
    # Loss model configuration
    loss_model: LossModel
    # Optimizer model configuration
    optimizer_model: OptimizerModel

#######################
# Provider Settings
#######################

@dataclass
class Provider:
    # Provider name (e.g., 'openai')
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
    # Directory containing input images
    data_dir: str
    # Global delay for API calls
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
    # API provider configuration
    provider: Provider
    # General experiment settings
    general: General
    # Logging configuration
    logging: Logging