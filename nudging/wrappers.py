from abc import ABC, abstractmethod
import base64
from typing import Tuple, List, Type
from PIL import Image
import io
import litellm
import logging
import json
from schema import IOSchema, EvaluatorInput, EvaluationOutput, LossInput, CritiqueOutput, OptimizerInput, OptimizedPromptOutput


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
        input_schema: Type[IOSchema],
        output_schema: Type[IOSchema],
        enable_json_schema_validation: bool = True
    ):
        self.system_prompt = system_prompt
        self.api_call = api_call
        self.input_schema = input_schema
        self.output_schema = output_schema
        
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
    
    def _build_messages(self, inputs: IOSchema) -> List[dict]:
        """
        Constructs the message list for the API call.
        
        Images are sent as separate user messages.
        Text fields are combined into a single formatted user message.
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
        # Add image messages first (one message per image)
        for field_name, field_value in inputs.model_dump().items():
            if isinstance(field_value, list):
                # Assume list fields contain image bytes
                for img_bytes in field_value:
                    if isinstance(img_bytes, bytes):
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": self._encode_image(img_bytes)
                                    }
                                }
                            ]
                        })
        
        # Add text fields as a single formatted message
        text_content = inputs.to_formatted_string()
        if text_content:
            messages.append({
                "role": "user",
                "content": text_content
            })
        
        return messages
    
    def get_response(self, **kwargs) -> IOSchema:
        """
        Main method to call the LLM with structured output.
        """
        # Validate inputs using input_schema
        validated_inputs = self.input_schema(**kwargs)
        
        # Build messages from validated inputs
        messages = self._build_messages(validated_inputs)
        
        # Call the API with response_format set to the output schema
        response = self.api_call(
            messages=messages,
            response_format=self.output_schema
        )

        if response is None:
            return None
        
        # Parse the response
        content = response.choices[0].message.content
        
        parsed_json = json.loads(content)
        result = self.output_schema(**parsed_json)
        
        return result


######################
# Model Wrappers
######################

class EvaluatorModel:
    """
    A wrapper for the VLM that evaluates the edited image.
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.wrapper = LanguageModel(
            system_prompt=system_prompt,
            api_call=api_call,
            input_schema=EvaluatorInput,
            output_schema=EvaluationOutput
        )

    def evaluate(self, task: str, images: List[bytes]) -> EvaluationOutput:
        """
        Compare original and edited images.
        """
        return self.wrapper.get_response(
            task=task,
            images=images
        )


class LossModel:
    """
    A wrapper for the LLM that generates a critique (the "loss").
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.wrapper = LanguageModel(
            system_prompt=system_prompt,
            api_call=api_call,
            input_schema=LossInput,
            output_schema=CritiqueOutput
        )

    def get_critique(self, choice: str, reason: str) -> CritiqueOutput:
        """
        Generates a critique based on the evaluator's choice and reason.
        """
        return self.wrapper.get_response(
            choice=choice,
            reason=reason
        )


class OptimizerModel:
    """
    A wrapper for the LLM that updates the prompt.
    """
    def __init__(self, system_prompt: str, api_call: callable):
        self.wrapper = LanguageModel(
            system_prompt=system_prompt,
            api_call=api_call,
            input_schema=OptimizerInput,
            output_schema=OptimizedPromptOutput
        )

    def update_prompt(self, current_prompt: str, suggestions: str) -> OptimizedPromptOutput:
        """
        Generates a new prompt based on the current prompt and feedback.
        """
        return self.wrapper.get_response(
            current_prompt=current_prompt,
            suggestions=suggestions
        )