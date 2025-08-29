from abc import ABC, abstractmethod
import base64

class ImageEditingModel(ABC):
    """
    Abstract base class for all image editing models.
    """
    @abstractmethod
    def edit(self, prompt: str, image_bytes: bytes):
        """
        Applies the editing prompt to an image.
            
        Returns:
            tuple: (edited_image, edited_image_bytes)
        """
        pass

class EvaluatorModel:
    """
    A wrapper for the VLM that evaluates the edited image.
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.system_prompt = system_prompt
        self.api_call = api_call

    def evaluate(self, original_bytes: bytes, edited_bytes: bytes) -> str:
        """
        Compares the original and edited images and returns the evaluation.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(original_bytes).decode('utf-8')}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(edited_bytes).decode('utf-8')}"}},
                    {"type": "text", "text": "Here are the original and edited images. Which one is more appealing according to the criteria?"}
                ],
            }
        ]
        try:
            return self.api_call(messages)
        except Exception as e:
            return f"CHOICE: original. ANALYSIS: Evaluation failed due to an API error: {e}"

class LossModel:
    """
    A wrapper for the LLM that generates a critique (the "loss").
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.system_prompt = system_prompt
        self.api_call = api_call

    def get_critique(self, context: str) -> str:
        """
        Generates a critique based on the current prompt and the VLM's evaluation.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
        return self.api_call(messages)

class OptimizerModel:
    """
    A wrapper for the LLM that updates the prompt.
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.system_prompt = system_prompt
        self.api_call = api_call

    def update_prompt(self, context: str) -> str:
        """
        Generates a new prompt based on the old prompt and the critique.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
        return self.api_call(messages)