from pydantic import Field
from typing import List, Type
from utils.wrappers import IOSchema


def create_evaluator_input_schema(task_description: str, images_description: str) -> Type[IOSchema]:
    """
    Creates an EvaluatorInput schema class with configurable field descriptions.
    """
    class EvaluatorInput(IOSchema):
        """Input schema for evaluator model."""
        task: str = Field(description=task_description)
        images: List[bytes] = Field(description=images_description)

    return EvaluatorInput


def create_evaluator_output_schema(choice_description: str) -> Type[IOSchema]:
    """
    Creates an EvaluatorOutput schema class with configurable field descriptions.
    """
    class EvaluatorOutput(IOSchema):
        choice: int = Field(description=choice_description) # type: ignore

    return EvaluatorOutput
