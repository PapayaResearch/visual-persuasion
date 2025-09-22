from dataclasses import dataclass
from typing import Any

#######################
# Strategy Settings
#######################

@dataclass
class Strategy:
    # Hydra target for strategy class
    _target_: str
    # Additional strategy-specific parameters
    # These will be filled in by the strategy-specific YAML files

#######################
# Dataset Settings
#######################

@dataclass
class Dataset:
    # Name of the dataset to be processed
    name: str
    # Subfolder within the source directory containing the image folders
    subfolder: str
    # Number of image folders to process (set to -1 to process all)
    num_folders: int
    # Number of images to evaluate from each folder (set to -1 to evaluate all)
    num_evaluate_per_folder: int
    # Number of images to finally select for processing from each folder (set to -1 to choose all)
    num_process_per_folder: int

#######################
# Provider Settings
#######################

@dataclass
class Provider:
    # Provider name (e.g., 'openai', 'anthropic')
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
    # Strategy configuration
    strategy: Strategy
    # General experiment settings
    general: General