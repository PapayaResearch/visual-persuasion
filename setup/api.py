import time
import logging
import litellm

def create_api_call(
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