import base64
import io
import litellm
import logging
import json
from typing import Tuple, List, Type
from pydantic import BaseModel
from PIL import Image
from abc import ABC, abstractmethod

class IOSchema(BaseModel):
    """
    Base class for all input and output schemas.
    Provides automatic formatting for logging and message construction.
    """

    @classmethod
    def to_description_string(cls) -> str:
        """
        Generates a description string listing all fields and their descriptions.
        """
        parts = []
        for field_name, field_info in cls.model_fields.items():
            description = field_info.description or "No description provided"

            # Format: FIELD_NAME:\ndescription\n\n
            formatted_field = f"{field_name.upper()}:\n{description.strip()}\n\n"
            parts.append(formatted_field)

        return "".join(parts).strip()

    def to_formatted_string(self) -> str:
        """
        Converts the schema to a formatted string for logging or message construction.
        Only includes string fields, skips lists (used for images).
        """
        parts = []
        for field_name, field_value in self.model_dump().items():
            # Skip images
            if isinstance(field_value, (list, bytes)):
                continue

            # Format: FIELD_NAME:\nvalue\n\n
            formatted_field = f"{field_name.upper()}:\n{str(field_value).strip()}\n\n"
            parts.append(formatted_field)

        return "".join(parts).strip()

    def __str__(self) -> str:
        """String representation using formatted output."""
        return self.to_formatted_string()


class ImageModel(ABC):
    """
    Abstract base class for all image editing models.
    """
    @abstractmethod
    def edit(
            self,
            prompt: str,
            image_bytes: bytes,
            context_image_bytes: bytes = None
    ) -> Tuple[Image.Image, bytes]:
        """
        Applies the editing prompt to an image, optionally with context.

        Args:
            prompt: The editing instruction
            image_bytes: The image to edit
            context_image_bytes: Optional context image for reference

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
        input_schema: Type[IOSchema],
        output_schema: Type[IOSchema],
        api_call: callable,
        enable_json_schema_validation: bool = True
    ):
        # Build full system prompt with input schema description
        input_description = input_schema.to_description_string()
        full_system_prompt = f"{system_prompt}\n\nExpect the following inputs:\n\n{input_description}"
        
        self.system_prompt = full_system_prompt
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.api_call = api_call
        self.return_usage_data = False

        # Enable JSON schema validation for models that don't natively support it
        if enable_json_schema_validation:
            litellm.enable_json_schema_validation = True

    def _encode_image(self, image_bytes: bytes, dim: int = 128) -> str:
        """
        Encodes image bytes to base64 data URL.
        """
        # Detect image format
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((dim, dim))
        img_format = img.format.lower() if img.format else 'jpeg'

        # Encode to base64
        buffered = io.BytesIO()
        img.save(buffered, format=img_format)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
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
            elif isinstance(field_value, bytes):
                # Single image bytes field
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._encode_image(field_value)
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

    def get_response(self, **kwargs):
        """
        Main method to call the LLM with structured output.
        Returns result, or (result, usage) if return_usage_data is True.
        """
        # Validate inputs using input_schema
        validated_inputs = self.input_schema(**kwargs)

        # Build messages from validated inputs
        messages = self._build_messages(validated_inputs)

        # Call the API with response_format set to the output schema
        try:
            response = self.api_call(
                messages=messages,
                response_format=self.output_schema
            )
        except Exception as e:
            logging.error(f"Language Model API call failed: {e}\n")
            if self.return_usage_data:
                return None, None
            return None

        # Parse the response
        content = response.choices[0].message.content

        parsed_json = json.loads(content)
        result = self.output_schema(**parsed_json)

        if self.return_usage_data:
            return result, response.usage
        return result
