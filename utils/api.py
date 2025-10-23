import litellm
import time
import os
from typing import Callable

def create_api_call(
        model: str,
        delay: float,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        additional_drop_params: list,
        return_message_only: bool
) -> Callable:
    """
    Factory for creating the api_call function.
    """
    def api_call(messages, tools=None, response_format=None):
        time.sleep(delay)

        result = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            additional_drop_params=list(additional_drop_params)
        )

        return result.choices[0].message if return_message_only else result

    return api_call
