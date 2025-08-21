import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import io
import base64
from api import create_api_call

class ImageEditingModel:
    """
    A wrapper for the Hugging Face image editing model.
    """
    def __init__(self, model_id: str, inference_steps: int, image_guidance_scale: float):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        self.pipe.enable_attention_slicing()
        self.inference_steps = inference_steps
        self.image_guidance_scale = image_guidance_scale

    def edit(self, prompt: str, image_bytes: bytes):
        """
        Applies the editing prompt to an image.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        edited_image = self.pipe(
            prompt, 
            image=image, 
            num_inference_steps=self.inference_steps, 
            image_guidance_scale=self.image_guidance_scale
        ).images[0]
        
        with io.BytesIO() as output:
            edited_image.save(output, format="JPEG")
            edited_image_bytes = output.getvalue()
            
        return edited_image, edited_image_bytes

class EvaluatorModel:
    """
    A wrapper for the VLM that evaluates the edited image.
    """
    def __init__(self, model: str, temperature: float, max_tokens: int, system_prompt: str, delay: int):
        self.system_prompt = system_prompt
        self.api_call = create_api_call(model, temperature, max_tokens, delay)

    def evaluate(self, original_bytes: bytes, edited_bytes: bytes) -> str:
        """
        Compares the original and edited images and returns the evaluation.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(original_bytes).decode('utf-8')}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(edited_bytes).decode('utf-8')}"}},
                    {"type": "text", "text": "Here are the original and edited images. Which one is more appealing according to the criteria?"}
                ],
            }
        ]
        try:
            return self.api_call(messages)
        except Exception as e:
            return f"CHOICE: original. ANALYSIS: Evaluation failed due to an API error: {e}"

class LossModel:
    """
    A wrapper for the LLM that generates a critique (the "loss").
    """
    def __init__(self, model: str, temperature: float, max_tokens: int, system_prompt: str, delay: int):
        self.system_prompt = system_prompt
        self.api_call = create_api_call(model, temperature, max_tokens, delay)

    def get_critique(self, context: str) -> str:
        """
        Generates a critique based on the current prompt and the VLM's evaluation.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
        return self.api_call(messages)

class OptimizerModel:
    """
    A wrapper for the LLM that updates the prompt.
    """
    def __init__(self, model: str, temperature: float, max_tokens: int, system_prompt: str, delay: int):
        self.system_prompt = system_prompt
        self.api_call = create_api_call(model, temperature, max_tokens, delay)

    def update_prompt(self, context: str) -> str:
        """
        Generates a new prompt based on the old prompt and the critique.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
        return self.api_call(messages)