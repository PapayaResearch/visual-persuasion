from pydantic import BaseModel, Field
from typing import List, Type


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
            # Skip None values, list fields (images), and bytes fields (images)
            if field_value is None or isinstance(field_value, (list, bytes)):
                continue
            
            # Format: FIELD_NAME:\nvalue\n\n
            formatted_field = f"{field_name.upper()}:\n{field_value}\n\n"
            parts.append(formatted_field)
        
        return "".join(parts).strip()
    
    def __str__(self) -> str:
        """String representation using formatted output."""
        return self.to_formatted_string()


def create_evaluator_input_schema(task_description: str, images_description: str) -> Type[IOSchema]:
    """
    Creates an EvaluatorInput schema class with configurable field descriptions.
    """
    class EvaluatorInput(IOSchema):
        """Input schema for evaluator model."""
        task: str = Field(description=task_description)
        images: List[bytes] = Field(description=images_description)
    
    return EvaluatorInput


def create_evaluator_output_schema(choice_description: str, reason_description: str) -> Type[IOSchema]:
    """
    Creates an EvaluatorOutput schema class with configurable field descriptions.
    """
    class EvaluatorOutput(IOSchema):
        """Output schema for evaluator model."""
        choice: str = Field(description=choice_description)
        reason: str = Field(description=reason_description)
    
    return EvaluatorOutput


def create_loss_input_schema(choice_description: str, reason_description: str) -> Type[IOSchema]:
    """
    Creates a LossInput schema class with configurable field descriptions.
    """
    class LossInput(IOSchema):
        """Input schema for loss model."""
        choice: str = Field(description=choice_description)
        reason: str = Field(description=reason_description)
    
    return LossInput


def create_loss_output_schema(suggestions_description: str) -> Type[IOSchema]:
    """
    Creates a CritiqueOutput schema class with configurable field descriptions.
    """
    class LossOutput(IOSchema):
        """Output schema for loss model."""
        suggestions: str = Field(description=suggestions_description)
    
    return LossOutput


def create_optimizer_input_schema(current_prompt_description: str, suggestions_description: str) -> Type[IOSchema]:
    """
    Creates an OptimizerInput schema class with configurable field descriptions.
    """
    class OptimizerInput(IOSchema):
        """Input schema for optimizer model."""
        current_prompt: str = Field(description=current_prompt_description)
        suggestions: str = Field(description=suggestions_description)
    
    return OptimizerInput


def create_optimizer_output_schema(new_prompt_description: str) -> Type[IOSchema]:
    """
    Creates an OptimizedPromptOutput schema class with configurable field descriptions.
    """
    class OptimizedPromptOutput(IOSchema):
        """Output schema for optimizer model."""
        new_prompt: str = Field(description=new_prompt_description)
    
    return OptimizedPromptOutput
