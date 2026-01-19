import litellm
import time
from typing import Callable

# Suppress litellm info messages
litellm.suppress_debug_info = True

def create_api_call(model: str, delay: float, params: dict) -> Callable:
    """
    Factory for creating the api_call function.
    """
    def api_call(messages, tools):
        time.sleep(delay)
        return litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            **params
        )
    return api_call
