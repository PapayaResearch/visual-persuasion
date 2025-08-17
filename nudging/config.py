from dataclasses import dataclass

# API provider settings
@dataclass
class Provider:
    name: str
    key: str
    key_name: str

# Image editing model settings
@dataclass
class ImageEditing:
    model_id: str
    initial_prompt: str
    inference_steps: int
    image_guidance_scale: float

# VLM evaluator settings
@dataclass
class Evaluator:
    evaluator_model: str
    max_tokens: int
    evaluator_prompt: str

# TextGrad optimizer settings
@dataclass
class Optimizer:
    engine: str
    iterations: int
    loss_prompt: str

# General experiment settings
@dataclass
class General:
    data_dir: str

# Logging directory settings
@dataclass
class Logging:
    log_dir: str
    results_dir: str

# Main configuration class that holds all other configs
@dataclass
class Config:
    provider: Provider
    image_editing: ImageEditing
    evaluator: Evaluator
    optimizer: Optimizer
    general: General
    logging: Logging