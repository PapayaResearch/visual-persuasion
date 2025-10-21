from wrappers import ImageModel
import logging
import os
import io
from PIL import Image
import base64
from google import genai

class Gemini(ImageModel):
    """
    Implementation of Gemini-based image models for image editing.
    """
    def __init__(self, key_name: str, model: str):        
        self.model = model
        self.client = genai.Client(api_key=os.environ[key_name])
        
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

class LiteLLM(ImageModel):
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