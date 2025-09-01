from wrappers import ImageEditingModel
import logging
import torch
import io
from PIL import Image
import base64
from diffusers import StableDiffusionInstructPix2PixPipeline
from google import genai

class InstructPix2Pix(ImageEditingModel):
    """
    Implementation of the InstructPix2Pix model for image editing.
    """
    def __init__(self, model_id: str, inference_steps: int, image_guidance_scale: float):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            # This helps prevent memory errors
            low_cpu_mem_usage=True
        )
        # Use efficient memory management techniques
        self.pipe.enable_model_cpu_offload()
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

class NanoBanana(ImageEditingModel):
    """
    Implementation of Google's Nano Banana (Gemini 2.5 Flash Image) model for image editing.
    """
    def __init__(self, key: str, model: str):        
        self.model = model

        # Set up provider API key
        try:
            with open(key) as infile:
                self.api_key = infile.read().strip()
            logging.info(f"Set Nano Banana API key from {key}")
        except FileNotFoundError:
            logging.error(f"Nano Banana API key file not found at: {key}")
        
        self.client = genai.Client(api_key=self.api_key)
        
    def edit(self, prompt: str, image_bytes: bytes):            
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # API call
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image],
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                logging.info(f"\nNano Banana Response:\n{part.text}\n")
            elif part.inline_data is not None:
                edited_image_bytes = part.inline_data.data
                edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")

        return edited_image, edited_image_bytes