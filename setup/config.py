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
    # Sampling temperature for responses
    temperature: float
    # Maximum tokens in API response
    max_tokens: int
    # Delay before API calls to avoid rate limits
    delay: int

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

@dataclass
class ImageEnhancer:
    # Hydra target for image enhancement class
    _target_: str
    # Model to be used for image enhancement
    enhancement_model: ImageModel
    # The prompt for enhancing the original images
    enhancement_prompt: str

@dataclass
class BackgroundProcessor:
    # Hydra target for background processing class
    _target_: str
    # Model to be used for background processing
    image_editing_model: ImageModel
    # Maximum number of images to preview from the with-background subset (set to -1 to preview all)
    num_previews_with_background: int
    # Maximum number of images to preview from the without-background subset (set to -1 to preview all)
    num_previews_without_background: int
    # The prompt for the background removal task
    background_removal_prompt: str
    # Threshold for SSIM value to consider an image as having a plain background
    ssim_threshold: float
    # Enable background normalization
    enable_background_normalization: bool
    # The prompt for the background normalization task
    background_normalization_prompt: str

#######################
# Strategy Settings
#######################

@dataclass
class RandomSampling:
    # Hydra target for random sampling strategy
    _target_: str

@dataclass
class VLMFiltering:
    # Hydra target for VLM filtering strategy
    _target_: str
    # Evaluator model configuration
    evaluator_model: LanguageModel

#######################
# Dataset Settings
#######################

@dataclass
class Dataset:
    # Name of the dataset to be processed
    name: str
    # Number of image folders to process (set to -1 to process all)
    num_folders: int
    # Number of images to evaluate from each folder if there is filtering involved (set to -1 to evaluate all)
    num_evaluate_per_folder: int
    # Number of images to finally select for processing from each folder (set to -1 to choose all)
    num_process_per_folder: int

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
    # Directory containing the folders of images to process
    src_dir: str
    # Destination directory for saving the processed images
    dst_dir: str
    # Enhance image quality before further processing
    enhance_image_quality: bool
    # Enable splitting of dataset by background type
    split_by_background: bool
    # Model to be used for processing images
    model: str
    # Temperature for all API calls
    temperature: float
    # Max tokens for all API calls
    max_tokens: int
    # Standard delay in seconds before making an API call
    delay: int

######################
# Main Config
######################

@dataclass
class Config:
    # Dataset configuration
    dataset: Dataset
    # API provider configuration
    provider: Provider
    # API provider configuration for image models
    provider_image: Provider
    # Strategy configuration (RandomSampling or VLMFiltering)
    strategy: RandomSampling  # Or VLMFiltering
    # General experiment settings
    general: General
    # Image enhancement configuration
    image_enhancer: ImageEnhancer
    # Background processing configuration
    background_processor: BackgroundProcessor