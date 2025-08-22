from dataclasses import dataclass

######################
# Model Settings
######################

@dataclass
class ImageEditingConfig:
    # Hydra target class path
    _target_: str
    # Hugging Face model identifier
    model_id: str
    # Number of diffusion steps for the image editing model
    inference_steps: int
    # How much the edited image should conform to the original image structure
    image_guidance_scale: float

@dataclass
class EvaluatorConfig:
    # Hydra target class path
    _target_: str
    # The VLM used to compare the original and edited images
    model: str
    # Max tokens for the evaluator VLM response
    max_tokens: int
    # Sampling temperature for the model's output
    temperature: float
    # Delay in seconds before making an API call
    delay: int
    # The system prompt for the image comparison task
    system_prompt: str

@dataclass
class LossConfig:
    # Hydra target class path
    _target_: str
    # The LLM used to generate the critique (loss)
    model: str
    # Max tokens for the critique response
    max_tokens: int
    # Sampling temperature for the model's output
    temperature: float
    # Delay in seconds before making an API call
    delay: int
    # The system prompt for the loss generation task
    system_prompt: str

@dataclass
class OptimizerConfig:
    # Hydra target class path
    _target_: str
    # The LLM used to generate the new, optimized prompt
    model: str
    # Max tokens for the new prompt
    max_tokens: int
    # Sampling temperature for the model's output
    temperature: float
    # Delay in seconds before making an API call
    delay: int
    # The system prompt for the prompt optimization task
    system_prompt: str

######################
# Misc. Objects
######################

@dataclass
class Provider:
    # Name of the API provider (e.g., 'openai')
    name: str
    # Path to the file containing the API key
    key: str
    # The environment variable name to set for the API key
    key_name: str

@dataclass
class VisualNudgeConfig:
    # Hydra target class path for the main pipeline orchestrator
    _target_: str
    # Total number of optimization iterations to run per image
    iterations: int
    # The initial prompt for the image editing task
    initial_prompt: str
    # Configuration object for the image editing model
    image_editing_model: ImageEditingConfig
    # Configuration object for the evaluator model
    evaluator_model: EvaluatorConfig
    # Configuration object for the loss model
    loss_model: LossConfig
    # Configuration object for the optimizer model
    optimizer_model: OptimizerConfig

######################
# General Settings
######################

@dataclass
class General:
    # Directory containing the images to be tested
    data_dir: str
    # Global delay in seconds before API calls to avoid rate limits
    delay: int
    # Global number of optimization iterations to run per image
    iterations: int

######################
# Logging Settings
######################

@dataclass
class Logging:
    # Base directory for writing log files
    log_dir: str
    # Base directory for writing results (images, configs)
    results_dir: str

######################
# The Config
######################

@dataclass
class Config:
    # API provider configuration
    provider: Provider
    # Main pipeline configuration object
    visual_nudge: VisualNudgeConfig
    # General experiment settings
    general: General
    # Logging path settings
    logging: Logging