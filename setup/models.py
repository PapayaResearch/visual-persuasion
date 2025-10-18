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
    
    def edit_with_context(self, prompt: str, image_bytes: bytes, context_image_bytes: bytes):
        raise NotImplementedError("InstructPix2Pix does not support context-based editing.")

class Gemini(ImageEditingModel):
    """
    Implementation of Gemini-based image models for image editing.
    """
    def __init__(self, key: str, model: str):        
        self.model = model

        # Set up provider API key
        try:
            with open(key) as infile:
                self.api_key = infile.read().strip()
            logging.info(f"Set Gemini API key from {key}\n")
        except FileNotFoundError:
            logging.error(f"Gemini API key file not found at: {key}\n")
        
        self.client = genai.Client(api_key=self.api_key)
        
    def edit(self, prompt: str, image_bytes: bytes):            
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # API call
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image],
            )
        except Exception as e:
            logging.error(f"Gemini API call failed: {e}\n")
            return None, None

        if response is None or not hasattr(response, 'candidates') or not response.candidates:
            logging.error("Gemini API call failed: No response or candidates\n")
            return None, None

        edited_image = None
        edited_image_bytes = None

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                logging.info(f"Image Model Response:\n{part.text}\n")
            elif part.inline_data is not None:
                edited_image_bytes = part.inline_data.data
                edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")
        
        return edited_image, edited_image_bytes

    def edit_with_context(self, prompt: str, image_bytes: bytes, context_image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        context_image = Image.open(io.BytesIO(context_image_bytes)).convert("RGB")

        # API call
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image, context_image],
            )
        except Exception as e:
            logging.error(f"Gemini API call failed: {e}\n")
            return None, None

        if response is None or not hasattr(response, 'candidates') or not response.candidates:
            logging.error("Gemini API call failed: No response or candidates\n")
            return None, None
        
        edited_image = None
        edited_image_bytes = None

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                logging.info(f"Image Model Response:\n{part.text}\n")
            elif part.inline_data is not None:
                edited_image_bytes = part.inline_data.data
                edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")
        
        return edited_image, edited_image_bytes

class LiteLLM(ImageEditingModel):
    """
    Implementation of LiteLLM-based image models for image editing.
    """
    def __init__(self, api_call: callable):
        self.api_call = api_call
        
    def edit(self, prompt: str, image_bytes: bytes):            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # API call
        response = self.api_call(messages)

        if response is None:
            logging.error("LiteLLM API call failed: No response\n")
            return None, None

        if not hasattr(response, 'image') or response.image is None:
            logging.error("LiteLLM API call failed: No image in response\n")
            return None, None
        
        image_string = response.image["url"].split(',', 1)[1]

        if not hasattr(response, 'content') or response.content is None:
            text_response = ""
        else:   
            text_response = response.content
        
        logging.info(f"Image Model Response:\n{text_response}\n")
        edited_image_bytes = base64.b64decode(image_string)
        edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")

        return edited_image, edited_image_bytes
    
    def edit_with_context(self, prompt: str, image_bytes: bytes, context_image_bytes: bytes):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(context_image_bytes).decode('utf-8')}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # API call
        response = self.api_call(messages)

        if response is None:
            logging.error("LiteLLM API call failed: No response\n")
            return None, None

        if not hasattr(response, 'image') or response.image is None:
            logging.error("LiteLLM API call failed: No image in response\n")
            return None, None
        
        image_string = response.image["url"].split(',', 1)[1]

        if not hasattr(response, 'content') or response.content is None:
            text_response = ""
        else:   
            text_response = response.content
        
        logging.info(f"Image Model Response:\n{text_response}\n")
        edited_image_bytes = base64.b64decode(image_string)
        edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")

        return edited_image, edited_image_bytes