from dataclasses import dataclass

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
class ImageEnhancer:
    # Hydra target for image enhancement class
    _target_: str
    # Model to be used for image enhancement
    enhancement_model: ImageModel
    # The prompt for enhancing the original images
    enhancement_prompt: str

#######################
# Strategy Settings
#######################

@dataclass
class SamplingStrategy:
    # Hydra target for sampling strategy class
    _target_: str
    # Additional strategy-specific parameters (from strategy configs)
    # These will be filled in by the strategy-specific YAML files

#######################
# Dataset Settings
#######################

@dataclass
class Dataset:
    # Name of the dataset (used for creating destination subfolder)
    name: str
    # Number of image folders to process (set to -1 to process all)
    num_folders: int
    # Number of images to select for processing from each folder (set to -1 to choose all)
    num_process_per_folder: int

#######################
# General Settings
#######################

@dataclass
class General:
    # Directory containing the folders of images to process
    src_dir: str
    # Destination directory for saving the processed images
    dst_dir: str
    # Maximum number of parallel workers for processing
    max_workers: int

######################
# Main Config
######################

@dataclass
class Config:
    # Image editor model configuration
    editor: ImageModel
    # Strategy configuration (instantiated from strategy YAML files)
    strategy: SamplingStrategy
    # General experiment settings
    general: General
    # Dataset configuration
    dataset: Dataset
    # Image enhancement configuration
    image_enhancer: ImageEnhancer
