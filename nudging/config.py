from dataclasses import dataclass
from typing import Any

#######################
# Misc. Objects
#######################

@dataclass
class Nudge:
    # Nudge name
    name: str
    # Nudge object
    nudge: Any

@dataclass
class Provider:
    # Provider name
    name: str
    # Path to key txt file
    key: str
    # Key name for this provider
    key_name: str
    # Use integers or strings in tool calls
    supports_integers: bool

#######################
# General Settings
#######################

@dataclass
class General:
    # Model from those available in the provider
    model: str
    # Allows parallel tool calls
    parallel_tool_calls: bool
    # Force model to use one or more tools (none, auto, required)
    tool_choice: Any
    # Maximum number of tokens in the response from the API
    max_tokens: int
    # Number of participants from the original data
    participants: int
    # Offset the number of participants to be able to dynamically collect more
    offset: int
    # Temperature for the model (OpenAI's default is 1.0)
    temperature: float
    # Number of examples the model sees before playing, or null
    fewshot: Any
    # Include chain-of-Thought prompt with the experiments
    cot: bool
    # Include practice in the context window or not
    include_practice: bool
    # Random seed for reproducibility
    seed: int
    # Drop params specific params that the model doesn't support
    additional_drop_params: list
    # Delay before API calls to avoid rate limits
    delay: int
    # API wrapper
    api_call: Any

#######################
# Logging Settings
#######################

@dataclass
class Logging:
    # Directory for writing logs (for Tensorboard)
    log_dir: str
    # Directory for writing results (for audio and parameters)
    results_dir: str

######################
# The Config
######################

@dataclass
class Config:
    # Nudge settings
    nudge: Nudge
    # Provider settings
    provider: Provider
    # General settings
    general: General
    # Logging settings
    logging: Logging