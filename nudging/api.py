import litellm
import time
import logging
import os
from typing import Callable

def create_text_api_call(
        model: str,
        temperature: float,
        max_tokens: int,
        delay: float
) -> Callable:
    """
    Factory for creating the api_call function for text generation tasks.
    """
    def api_call(messages):
        """
        Calls the LLM API with the provided messages.
        """
        time.sleep(delay) # Delay before each call
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ).choices[0].message.content
        except Exception as e:
            logging.error(f"Litellm API call failed: {e}\n")
            return None
    
    return api_call

def create_image_api_call(
        key_name: str,
        model: str,
        delay: float
) -> Callable:
    """
    Factory for creating the api_call function for image generation tasks.
    """
    def api_call(messages):
        """
        Calls the LLM API with the provided messages.
        """
        time.sleep(delay) # Delay before each call
        try:
            return litellm.completion(
                api_key=os.environ[key_name],
                model=model,
                messages=messages
            ).choices[0].message
        except Exception as e:
            logging.error(f"Litellm API call failed: {e}\n")
            return None
    
    return api_call