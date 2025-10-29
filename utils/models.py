import logging
import io
import base64
from PIL import Image
from google.genai import Client, types
from utils.wrappers import ImageModel

class Gemini(ImageModel):
    """
    Implementation of Gemini-based image models for image editing.
    """
    def __init__(self, model: str, max_retries: int, aspect_ratio: str):
        self.model = model
        self.max_retries = max_retries
        self.aspect_ratio = aspect_ratio
        self.client = Client()
        # Suppress regular logs from Gemini SDK
        logging.getLogger('google_genai.models').setLevel(logging.WARNING)

    def edit(self, prompt: str, image_bytes: bytes, context_image_bytes: bytes = None):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt += "\nReturn a single image as output, do not give any options."

        # Build contents list conditionally
        contents = [prompt, image]
        if context_image_bytes:
            context_image = Image.open(io.BytesIO(context_image_bytes)).convert("RGB")
            contents.append(context_image)

        # API call
        for attempt in range(self.max_retries):
            if attempt == self.max_retries:
                logging.error("Gemini API call failed: maximum retries exceeded.\n")
                return None, None
            # Try to get a response
            response = None
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['Image'],
                        image_config=types.ImageConfig(
                            aspect_ratio=self.aspect_ratio
                        )
                    )
                )
                if response and response.candidates and response.candidates[0].content:
                    break
            except Exception as e:
                logging.error(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}\n")
                continue

        edited_image = None
        edited_image_bytes = None

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                edited_image_bytes = part.inline_data.data
                edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")

        return edited_image, edited_image_bytes

class LiteLLM(ImageModel):
    """
    Implementation of LiteLLM-based image models for image editing.
    """
    def __init__(self, api_call: callable):
        self.api_call = api_call

    def edit(self, prompt: str, image_bytes: bytes, context_image_bytes: bytes = None):
        content = [
            {
                "type": "image_url", "image_url":
                {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                }
            },
        ]

        if context_image_bytes:
            content.append(
                {
                    "type": "image_url", "image_url":
                    {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(context_image_bytes).decode('utf-8')}"
                    }
                }
            )

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # API call
        try:
            response = self.api_call(messages)
        except Exception as e:
            logging.error(f"Image Model API call failed: {e}\n")
            return None, None

        image_string = response.choices[0].message.images[0]["image_url"]["url"].split(',', 1)[1]

        edited_image_bytes = base64.b64decode(image_string)
        edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")

        return edited_image, edited_image_bytes
