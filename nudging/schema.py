from pydantic import BaseModel, Field
from typing import List


class IOSchema(BaseModel):
    """
    Base class for all input and output schemas.
    Provides automatic formatting for logging and message construction.
    """
    
    def to_formatted_string(self) -> str:
        """
        Converts the schema to a formatted string for logging or message construction.
        Only includes string fields, skips lists (used for images).
        """
        parts = []
        for field_name, field_value in self.model_dump().items():
            # Skip None values and list fields (images)
            if field_value is None or isinstance(field_value, list):
                continue
            
            # Format: FIELD_NAME:\nvalue\n\n
            formatted_field = f"{field_name.upper()}:\n{field_value}\n\n"
            parts.append(formatted_field)
        
        return "".join(parts).strip()
    
    def __str__(self) -> str:
        """String representation using formatted output."""
        return self.to_formatted_string()


class EvaluatorInput(IOSchema):
    """Input schema for evaluator model."""
    task: str = Field(description="The image editing task description")
    images: List[bytes] = Field(description="List of image bytes: [original, edited]")


class LossInput(IOSchema):
    """Input schema for loss model."""
    choice: str = Field(description="Which image was chosen as better")
    reason: str = Field(description="Reason for the choice")


class OptimizerInput(IOSchema):
    """Input schema for optimizer model."""
    current_prompt: str = Field(description="The current image editing prompt")
    suggestions: str = Field(description="Suggestions for improvement")


class EvaluationOutput(IOSchema):
    """Output schema for evaluator model."""
    choice: str = Field(
        description="Which image is better: 'original' or 'edited'"
    )
    reason: str = Field(
        description="Detailed explanation of why the chosen image is more appealing"
    )


class CritiqueOutput(IOSchema):
    """Output schema for loss model."""
    suggestions: str = Field(
        description="Concrete, actionable suggestions for improving the next iteration, formatted as a bulleted list"
    )


class OptimizedPromptOutput(IOSchema):
    """Output schema for optimizer model."""
    new_prompt: str = Field(
        description="The refined image editing instruction, incorporating feedback while keeping successful elements"
    )