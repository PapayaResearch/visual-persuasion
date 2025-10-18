from abc import ABC, abstractmethod
import base64
import logging
from typing import List, Tuple
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
                logging.error(f"Evaluator returned invalid index: {response}\nDefaulting to first image.")
                return 0  # Default to first image on error
        except ValueError:
            logging.error(f"Evaluator response parsing failed: {response}\nDefaulting to first image.")
            return 0  # Default to first image on error