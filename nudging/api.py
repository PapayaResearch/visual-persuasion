import litellm
import time
import logging

def create_text_api_call(
        model,
        temperature,
        max_tokens,
        delay
):
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
        key,
        model,
        delay
):
    """
    Factory for creating the api_call function for image generation tasks.
    """
    # Set up provider API key
    try:
        with open(key) as infile:
            api_key = infile.read().strip()
    except FileNotFoundError:
        raise Exception(f"API key file not found at: {key}")

    def api_call(messages):
        """
        Calls the LLM API with the provided messages.
        """
        time.sleep(delay) # Delay before each call
        try:
            return litellm.completion(
                api_key=api_key,
                model=model,
                messages=messages
            ).choices[0].message
        except Exception as e:
            logging.error(f"Litellm API call failed: {e}\n")
            return None
    
    return api_call