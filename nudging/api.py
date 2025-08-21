import litellm
import time

def create_api_call(
        model,
        temperature,
        max_tokens,
        delay
):
    """Factory for creating the api_call function."""
    return lambda messages: (
        time.sleep(delay),  # Delay before each call
        litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )).choices[0].message.content