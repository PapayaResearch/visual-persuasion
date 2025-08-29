from wrappers import ImageEditingModel
import torch
import io
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

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