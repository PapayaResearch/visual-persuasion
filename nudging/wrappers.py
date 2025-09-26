from abc import ABC, abstractmethod
import base64
from typing import Tuple
from PIL import Image

class ImageEditingModel(ABC):
    """
    Abstract base class for all image editing models.
    """
    @abstractmethod
    def edit(self, prompt: str, image_bytes: bytes) -> Tuple[Image.Image, bytes]:
        """
        Applies the editing prompt to an image.
            
        Returns:
            tuple: (edited_image, edited_image_bytes)
        """
        pass

    @abstractmethod
    def edit_with_context(self, prompt: str, image_bytes: bytes, context_image_bytes: bytes) -> Tuple[Image.Image, bytes]:
        """
        Applies the editing prompt to an image with additional context from another image.
            
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

    def evaluate(self, prompt: str, image1_bytes: bytes, image2_bytes: bytes) -> str:
        """
        Compares the two images and returns the evaluation.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image1_bytes).decode('utf-8')}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image2_bytes).decode('utf-8')}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        return self.api_call(messages)

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

    def update_prompt(self, current_prompt: str, critique: str) -> str:
        """
        Generates a new prompt based on the old prompt and the critique.
        """
        context = (
            "ORIGINAL PROMPT:\n"
            f"{current_prompt}\n\n"
            "CRITIQUE:\n"
            f"{critique}\n"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
        return self.api_call(messages)