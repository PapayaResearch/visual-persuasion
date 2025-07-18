import os
from PIL import Image, ImageEnhance
import litellm
from textgrad import Variable
from textgrad.engine import get_engine
from textgrad.optimizer import TextualGradientDescent
from textgrad.loss import MultiFieldEvaluation
import io
import base64

# Engine to use for optimization
ENGINE = get_engine("gpt-4o-mini")
# Path to the test image
IMG_PATH = os.path.join("nudging", "data", "xxxxxx.jpg")

# Load API key
with open(os.path.join("nudging", "keys", "oai.txt"), "r") as key_file:
    os.environ["OPENAI_API_KEY"] = key_file.read().strip()

# Wrapper for image editing model
def image_editing_model(prompt: Variable, original_image_bytes: bytes) -> Variable:
    """
    Takes a text prompt and the bytes of an original image, and returns the bytes of the edited image.
    """
    # Convert bytes to a PIL Image object
    image = Image.open(io.BytesIO(original_image_bytes))

    # MOCK EDIT: Adjust brightness
    brightness_factor = 1.5
    enhancer = ImageEnhance.Brightness(image)
    edited_image = enhancer.enhance(brightness_factor)

    # Convert the edited image back to bytes
    with io.BytesIO() as output:
        edited_image.save(output, format="JPEG")
        edited_image_bytes = output.getvalue()

    return edited_image_bytes

# Wrapper for evaluator model
def evaluator_vlm(prompt: Variable, original_image_bytes: bytes, edited_image_bytes: bytes) -> Variable:
    """
    Returns 'edited' if the edited image seems more appealing, 'original' otherwise.
    """
    # System prompt to guide the evaluation
    system_prompt = """
        You are an evaluator model.
        You will be given two images: original and edited.
        Your task is to decide which one is more appealing.
        Give a detailed justification highlighting the differences that affected your choice.
        Respond in the following format:

        Choice: [original/edited]
        Reason: [Your reason here]
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(original_image_bytes).decode('utf-8')}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(edited_image_bytes).decode('utf-8')}"}},
                {"type": "text", "text": "Here are the original and edited images. Which one is more appealing?"}
            ]
        }
    ]

    # Call LiteLLM to get the VLM's decision
    try:
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=100
        )
        response_content = response.choices[0].message.content
        # Parse the response to extract choice and reason
        choice_line = [line for line in response_content.split('\n') if line.startswith("Choice:")][0]
        reason_line = [line for line in response_content.split('\n') if line.startswith("Reason:")][0]
        choice = choice_line.split(":")[1].strip()
        reason = reason_line.split(":")[1].strip()
    except Exception as e:
        print(f"Error calling LiteLLM for VLM simulation: {e}")
        # Fallback to a default or mock decision if API call fails
        choice = "original"
        reason = "API call failed, defaulting to original."

    print(f"VLM Response:\n{response_content}\n")

    response = f"Choice: {choice}\nReason: {reason}"
    
    return Variable(
        response,
        requires_grad=True,
        role_description="Model Output (choice and reason)",
        predecessors=[prompt]
    )

# Run the test
def run_visual_nudge_test():
    """
    Runs the full pipeline to optimize the image editing prompt.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this example.")
        return
        
    # Initialize the variable to be optimized
    image_editing_prompt = Variable(
        "Make the image of the product more appealing.",
        role_description="prompt for an image editing model",
        requires_grad=True
    )

    # Initialize the TextGrad optimizer
    optimizer = TextualGradientDescent(engine=ENGINE, parameters=[image_editing_prompt])

    # Define the instruction for the loss function
    loss_instruction = Variable(
        """
            Give constructive feedback to make the image more appealing based on the reason given.
            Your job is to improve the edited model.
            Try to work on the highlights if the edited image was chosen,
            and try to work on the shortcomings if the original image was chosen.
        """,
        requires_grad=False,
        role_description="instruction for the loss function"
    )

    # Instantiate the evaluation module
    loss_fn = MultiFieldEvaluation(
        evaluation_instruction=loss_instruction,
        role_descriptions=["Model Output", "Target Output"],
        engine=ENGINE
    )

    # Define our target (model should choose the edited image)
    target_choice = Variable("edited",
                             requires_grad=False,
                             role_description="Target Output")


    # Load the test image
    with open(IMG_PATH, "rb") as f:
        original_image_bytes = f.read()


    print(f"--- Starting Optimization ---")

    # Run the optimization loop
    for i in range(1):
        print(f"\n>> ITERATION {i + 1}\n")
        print(f"Original Prompt:\n{image_editing_prompt.get_value()}\n")

        # Zero out gradients from the previous iteration
        optimizer.zero_grad()

        edited_image_bytes = image_editing_model(image_editing_prompt, original_image_bytes)

        vlm_choice = evaluator_vlm(image_editing_prompt, original_image_bytes, edited_image_bytes)

        loss = loss_fn([vlm_choice, target_choice])
        print(f"Loss:\n{loss.get_value()}\n")

        loss.backward(ENGINE)

        optimizer.step()
        print(f"New Optimized Prompt:\n{image_editing_prompt.get_value()}\n")
        print("-" * 30)

if __name__ == "__main__":
    run_visual_nudge_test()
