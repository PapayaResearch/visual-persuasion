from pydantic import Field
from typing import List, Type, Literal
from visualpersuasion.utils.wrappers import IOSchema


def create_difference_detector_input_schema(
    images_description: str,
    instruction_description: str
) -> Type[IOSchema]:
    """
    Creates a DifferenceDetectorInput schema class with configurable field descriptions.
    Used for autointerp where we only need images, no metadata.
    """
    class DifferenceDetectorInput(IOSchema):
        """Input schema for difference detector model."""
        instruction: str = Field(description=instruction_description)
        images: List[bytes] = Field(description=images_description)

    return DifferenceDetectorInput


def create_evaluator_without_prompt_input_schema(
    images_description: str,
    metadata_description: str = ""
) -> Type[IOSchema]:
    """
    Creates an EvaluatorWithoutPromptInput schema class with configurable field descriptions.
    """
    class EvaluatorWithoutPromptInput(IOSchema):
        """Input schema for evaluator model without judge prompt."""
        images: List[bytes] = Field(description=images_description)
        metadata: str = Field(description=metadata_description)

    return EvaluatorWithoutPromptInput

def create_evaluator_input_schema(
    images_description: str,
    judge_prompt_description: str,
    metadata_description: str = ""
) -> Type[IOSchema]:
    """
    Creates an EvaluatorInput schema class with configurable field descriptions.
    """
    class EvaluatorInput(IOSchema):
        """Input schema for evaluator model."""
        images: List[bytes] = Field(description=images_description)
        judge_prompt: str = Field(description=judge_prompt_description)
        metadata: str = Field(description=metadata_description)

    return EvaluatorInput


def create_evaluator_output_schema(choice_description: str, choice_options: List[str], reason_description: str) -> Type[IOSchema]:
    """
    Creates an EvaluatorOutput schema class with configurable field descriptions.
    """
    class EvaluatorOutput(IOSchema):
        choice: Literal[*choice_options] = Field(description=choice_description) # type: ignore
        reason: str = Field(description=reason_description)

    return EvaluatorOutput


def create_optimizer_input_schema(
    current_prompt_description: str = "",
    current_image_description: str = "",
    history_of_prompts_description: str = "",
    current_iteration_description: str = "",
    judge_feedback_description: str = "",
    metadata_description: str = ""
) -> Type[IOSchema]:
    """
    Creates an OptimizerInput schema class tailored for single-step improvements.
    """
    class OptimizerInput(IOSchema):
        """Input schema for optimizer model."""
        current_prompt: str = Field(description=current_prompt_description)
        current_image: bytes = Field(description=current_image_description)
        history_of_prompts: str = Field(description=history_of_prompts_description)
        current_iteration: int = Field(description=current_iteration_description)
        judge_feedback: str = Field(description=judge_feedback_description)
        metadata: str = Field(description=metadata_description)

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
    num_candidates_description: str = "",
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
        num_candidates: int = Field(description=num_candidates_description)
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

def create_difference_detector_output_schema(differences_description: str) -> Type[IOSchema]:
    """
    Creates a DifferenceDetectorOutput schema class with configurable field descriptions.
    """
    class DifferenceDetectorOutput(IOSchema):
        """Output schema for difference detector model."""
        differences: str = Field(description=differences_description)

    return DifferenceDetectorOutput


def create_theme_summarizer_input_schema(differences_description: str) -> Type[IOSchema]:
    """
    Creates a ThemeSummarizerInput schema class with configurable field descriptions.
    """
    class ThemeSummarizerInput(IOSchema):
        """Input schema for theme summarizer model."""
        differences: str = Field(description=differences_description)

    return ThemeSummarizerInput


def create_theme_summarizer_output_schema(themes_description: str) -> Type[IOSchema]:
    """
    Creates a ThemeSummarizerOutput schema class with configurable field descriptions.
    """
    class ThemeSummarizerOutput(IOSchema):
        """Output schema for theme summarizer model."""
        themes: str = Field(description=themes_description)

    return ThemeSummarizerOutput


def create_feedback_descent_proposer_input_schema(
    current_prompt_description: str = "",
    feedback_history_description: str = "",
    metadata_description: str = ""
) -> Type[IOSchema]:
    """
    Creates a FeedbackDescentProposerInput schema class with configurable field descriptions.
    """
    class FeedbackDescentProposerInput(IOSchema):
        """Input schema for Feedback Descent proposer model."""
        current_prompt: str = Field(description=current_prompt_description)
        feedback_history: str = Field(description=feedback_history_description)
        metadata: str = Field(description=metadata_description)

    return FeedbackDescentProposerInput


def create_feedback_descent_proposer_output_schema(new_prompt_description: str) -> Type[IOSchema]:
    """
    Creates a FeedbackDescentProposerOutput schema class with configurable field descriptions.
    """
    class FeedbackDescentProposerOutput(IOSchema):
        """Output schema for Feedback Descent proposer model."""
        new_prompt: str = Field(description=new_prompt_description)

    return FeedbackDescentProposerOutput


def create_context_removal_input_schema(
    images_description: str,
    metadata_description: str = ""
) -> Type[IOSchema]:
    """
    Creates a ContextRemovalInput schema class with configurable field descriptions.
    """
    class ContextRemovalInput(IOSchema):
        """Input schema for context removal model."""
        images: List[bytes] = Field(description=images_description)
        metadata: str = Field(description=metadata_description)

    return ContextRemovalInput


def create_context_removal_output_schema(
    editing_instruction_1_description: str,
    editing_instruction_2_description: str
) -> Type[IOSchema]:
    """
    Creates a ContextRemovalOutput schema class with configurable field descriptions.
    """
    class ContextRemovalOutput(IOSchema):
        """Output schema for context removal model."""
        editing_instruction_1: str = Field(description=editing_instruction_1_description)
        editing_instruction_2: str = Field(description=editing_instruction_2_description)

    return ContextRemovalOutput
