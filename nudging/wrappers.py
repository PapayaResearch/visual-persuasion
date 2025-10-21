from abc import ABC, abstractmethod
import base64
from typing import Tuple, List, Union, Type
from pydantic import BaseModel, Field
from PIL import Image
import io
import litellm
import logging
import json

class ImageModel(ABC):
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

class LanguageModel:
    """
    A generalized wrapper for LLM calls using structured outputs.
    
    Uses Pydantic models for automatic validation and parsing.
    Supports multiple inputs (text or images) with type-safe outputs.
    """
    
    def __init__(
        self,
        system_prompt: str,
        api_call: callable,
        output_model: Type[BaseModel],
        enable_json_schema_validation: bool = True
    ):
        self.system_prompt = system_prompt
        self.api_call = api_call
        self.output_model = output_model
        
        # Enable JSON schema validation for models that don't natively support it
        if enable_json_schema_validation:
            litellm.enable_json_schema_validation = True
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """
        Encodes image bytes to base64 data URL.
        """
        # Detect image format
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_format = img.format.lower() if img.format else 'jpeg'
        except Exception as e:
            logging.warning(f"Could not detect image format: {e}, defaulting to jpeg\n")
            img_format = 'jpeg'
        
        # Encode to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/{img_format};base64,{base64_image}"
    
    def _format_input(self, input_item: Union[str, bytes]) -> dict:
        """
        Formats a single input item into the appropriate message content format.
        """
        if isinstance(input_item, bytes):
            # Image input
            return {
                "type": "image_url",
                "image_url": {
                    "url": self._encode_image(input_item)
                }
            }
        else:
            # Text input
            return {
                "type": "text",
                "text": str(input_item)
            }
    
    def _build_messages(self, inputs: List[Union[str, bytes]]) -> List[dict]:
        """
        Constructs the message list for the API call.
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [self._format_input(inp) for inp in inputs]
            }
        ]
        return messages
    
    def get_response(self, inputs: List[Union[str, bytes]]) -> BaseModel:
        """
        Main method to call the LLM with structured output.
        """
        messages = self._build_messages(inputs)
        
        # Call the API with response_format set to the Pydantic model
        response = self.api_call(
            messages=messages,
            response_format=self.output_model
        )

        if response is None:
            return None
        
        # Parse the response
        content = response.choices[0].message.content
        
        parsed_json = json.loads(content)
        result = self.output_model(**parsed_json)
        
        return result

# Define output schema
class EvaluationOutput(BaseModel):
    choice: str = Field(
        description="Which image is better: 'original' or 'edited'"
    )
    reason: str = Field(
        description="Detailed explanation of why the chosen image is more appealing, focusing on specific visual qualities"
    )

class CritiqueOutput(BaseModel):
    issue: str = Field(
        description="The main reason why the edited image was not better than the original, based on the evaluator's feedback"
    )
    suggestions: str = Field(
        description="Concrete, actionable suggestions for improving the next iteration, formatted as a bulleted list"
    )

class OptimizedPromptOutput(BaseModel):
    prompt: str = Field(
        description="The refined image editing instruction, incorporating feedback while keeping successful elements. Must be under 100 words and clearly state what changes to make."
    )

class EvaluatorModel:
    """
    A wrapper for the VLM that evaluates the edited image.
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.wrapper = LanguageModel(
            system_prompt=system_prompt,
            api_call=api_call,
            output_model=EvaluationOutput
        )

    def evaluate(self, task: str, original_bytes: bytes, edited_bytes: bytes) -> EvaluationOutput:
        """
        Compare original and edited images.
        """
        return self.wrapper.get_response([task, original_bytes, edited_bytes])

class LossModel:
    """
    A wrapper for the LLM that generates a critique (the "loss").
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.wrapper = LanguageModel(
            system_prompt=system_prompt,
            api_call=api_call,
            output_model=CritiqueOutput
        )

    def get_critique(self, context: str) -> CritiqueOutput:
        """
        Generates a critique based on the current prompt and the VLM's evaluation.
        """
        return self.wrapper.get_response([context])

class OptimizerModel:
    """
    A wrapper for the LLM that updates the prompt.
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.wrapper = LanguageModel(
            system_prompt=system_prompt,
            api_call=api_call,
            output_model=OptimizedPromptOutput
        )

    def update_prompt(self, current_prompt: str, critique: str) -> OptimizedPromptOutput:
        """
        Generates a new prompt based on the old prompt and the critique.
        """
        context = f"Current prompt: {current_prompt}\n\nFeedback: {critique}"
        return self.wrapper.get_response([context])