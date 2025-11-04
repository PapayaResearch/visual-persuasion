from pydantic import Field
from typing import List, Type, Literal
from utils.wrappers import IOSchema


def create_evaluator_input_schema(images_description: str) -> Type[IOSchema]:
    """
    Creates an EvaluatorInput schema class with configurable field descriptions.
    """
    class EvaluatorInput(IOSchema):
        """Input schema for evaluator model."""
        images: List[bytes] = Field(description=images_description)

    return EvaluatorInput


def create_evaluator_output_schema(choice_description: str, choice_options: List[str], reason_description: str) -> Type[IOSchema]:
    """
    Creates an EvaluatorOutput schema class with configurable field descriptions.
    """
    class EvaluatorOutput(IOSchema):
        choice: Literal[*choice_options] = Field(description=choice_description) # type: ignore
        reason: str = Field(description=reason_description)

    return EvaluatorOutput


def create_loss_input_schema(choice_description: str, choice_options: List[str], reason_description: str, to_improve_description: str) -> Type[IOSchema]:
    """
    Creates a LossInput schema class with configurable field descriptions.
    """
    class LossInput(IOSchema):
        """Input schema for loss model."""
        choice: Literal[*choice_options] = Field(description=choice_description) # type: ignore
        reason: str = Field(description=reason_description)
        to_improve: Literal[*choice_options] = Field(description=to_improve_description) # type: ignore

    return LossInput


def create_loss_output_schema(suggestions_description: str) -> Type[IOSchema]:
    """
    Creates a CritiqueOutput schema class with configurable field descriptions.
    """
    class LossOutput(IOSchema):
        """Output schema for loss model."""
        suggestions: str = Field(description=suggestions_description)

    return LossOutput


def create_optimizer_input_schema(
    current_prompt_description: str,
    reason_description: str,
    history_of_prompts_description: str = "",
    current_iteration_description: str = "",
    total_iterations_description: str = ""
) -> Type[IOSchema]:
    """
    Creates an OptimizerInput schema class with configurable field descriptions.
    """
    class OptimizerInput(IOSchema):
        """Input schema for optimizer model."""
        current_prompt: str = Field(description=current_prompt_description)
        reason: str = Field(description=reason_description)
        history_of_prompts: str = Field(description=history_of_prompts_description)
        current_iteration: int = Field(description=current_iteration_description)
        total_iterations: int = Field(description=total_iterations_description)


    return OptimizerInput


def create_optimizer_output_schema(new_prompt_description: str) -> Type[IOSchema]:
    """
    Creates an OptimizedPromptOutput schema class with configurable field descriptions.
    """
    class OptimizedPromptOutput(IOSchema):
        """Output schema for optimizer model."""
        new_prompt: str = Field(description=new_prompt_description)

    return OptimizedPromptOutput


def create_proposer_input_schema(
    current_prompt_description: str = "",
    history_of_prompts_description: str = "",
    current_iteration_description: str = "",
    judge_feedback_description: str = "",
    total_iterations_description: str = "",
    num_proposals_description: str = "",
    metadata_description: str = ""
) -> Type[IOSchema]:
    """
    Creates a ProposerInput schema class with configurable field descriptions.
    """
    class ProposerInput(IOSchema):
        """Input schema for proposer model."""
        current_prompt: str = Field(description=current_prompt_description)
        history_of_prompts: str = Field(description=history_of_prompts_description)
        current_iteration: int = Field(description=current_iteration_description)
        judge_feedback: str = Field(description=judge_feedback_description)
        total_iterations: int = Field(description=total_iterations_description)
        num_proposals: int = Field(description=num_proposals_description)
        metadata: str = Field(description=metadata_description)

    return ProposerInput


def create_proposer_output_schema(candidate_prompts_description: str) -> Type[IOSchema]:
    """
    Creates a ProposerOutput schema class with configurable field descriptions.
    """
    class ProposerOutput(IOSchema):
        """Output schema for proposer model."""
        candidate_prompts: List[str] = Field(description=candidate_prompts_description)

    return ProposerOutput


def create_selector_input_schema(
    image_bytes_list_description: str,
    candidate_descriptions_description: str,
    num_candidates_description: str,
    judge_feedback_description: str
) -> Type[IOSchema]:
    """
    Creates a SelectorInput schema class with configurable field descriptions.
    """
    class SelectorInput(IOSchema):
        """Input schema for selector model."""
        images: List[bytes] = Field(description=image_bytes_list_description)
        candidate_descriptions: str = Field(description=candidate_descriptions_description)
        num_candidates: int = Field(description=num_candidates_description)
        judge_feedback: str = Field(description=judge_feedback_description)

    return SelectorInput


def create_selector_output_schema(choice_description: str) -> Type[IOSchema]:
    """
    Creates a SelectorOutput schema class with configurable field descriptions.
    """
    class SelectorOutput(IOSchema):
        """Output schema for selector model."""
        choice: str = Field(description=choice_description)

    return SelectorOutput
