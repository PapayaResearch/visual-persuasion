import time
import logging
import litellm
import base64
from typing import List

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

class EvaluatorModel:
    """
    A wrapper for the VLM that evaluates the images and chooses the best one.
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.system_prompt = system_prompt
        self.api_call = api_call

    def evaluate(self, images: List[bytes]) -> int:
        """
        Compares the two images and returns the evaluation.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(images[i]).decode('utf-8')}"}} for i in range(len(images))],
                    {"type": "text", "text": "Evaluate the images and return the index of the best one."}
                ]
            }
        ]
        response = self.api_call(messages)
        # Parse response to get the index of the best image
        try:
            best_index = int(response.strip()) - 1
            if best_index in range(len(images)):
                return best_index
            else:
                logging.error(f"Evaluator returned invalid index: {response}\n")
                return 0  # Default to first image on error
        except ValueError:
            logging.error(f"Evaluator response parsing failed: {response}\n")
            return 0  # Default to first image on error