import os
import io
import base64
import logging
import torch
import litellm
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from textgrad import Variable, TextualGradientDescent, TextLoss, get_engine, set_backward_engine

from config import Config

class VisualNudge:
    # Initialize class with configuration
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.getLogger("textgrad").setLevel(logging.INFO)
        self._setup_models()
        self._setup_textgrad()

    # Load the image editing model from Hugging Face
    def _setup_models(self):
        logging.info(f"Loading Image Editing model: {self.cfg.image_editing.model_id}")
        self.image_editing_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.cfg.image_editing.model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        self.image_editing_pipe.enable_attention_slicing()
        logging.info("Image Editing model loaded")

    # Set up TextGrad engine, variables, optimizer, and loss
    def _setup_textgrad(self):
        self.engine = get_engine(self.cfg.optimizer.engine)
        set_backward_engine(self.engine, override=True)
        
        self.image_editing_prompt = Variable(
            self.cfg.image_editing.initial_prompt,
            role_description="an instruction for an image editing model",
            requires_grad=True
        )
        
        self.optimizer = TextualGradientDescent(parameters=[self.image_editing_prompt])

        loss_instruction = Variable(
            self.cfg.optimizer.loss_prompt,
            requires_grad=False,
            role_description="instruction for the loss function"
        )
        self.loss_fn = TextLoss(eval_system_prompt=loss_instruction, engine=self.engine)

    # Apply the image editing model to an image
    def image_editing_model(self, prompt: Variable, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        edited_image = self.image_editing_pipe(
            prompt.value, 
            image=image, 
            num_inference_steps=self.cfg.image_editing.inference_steps, 
            image_guidance_scale=self.cfg.image_editing.image_guidance_scale
        ).images[0]
        
        with io.BytesIO() as output:
            edited_image.save(output, format="JPEG")
            edited_image_bytes = output.getvalue()
            
        return edited_image, edited_image_bytes

    # Use a VLM to evaluate the edited image
    def evaluator_vlm(self, prompt_placeholder: Variable, original_bytes: bytes, edited_bytes: bytes) -> Variable:
        messages = [
            {"role": "system", "content": self.cfg.evaluator.evaluator_prompt},
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
            response = litellm.completion(model=self.cfg.evaluator.evaluator_model, messages=messages, max_tokens=self.cfg.evaluator.max_tokens)
            response_content = response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error during VLM evaluation: {e}")
            response_content = "CHOICE: original. ANALYSIS: Evaluation failed due to an API error"

        return Variable(
            response_content,
            requires_grad=True,
            role_description="VLM's decision (choice and reason)",
            predecessors=[prompt_placeholder]
        )

    # Main optimization loop for a list of images
    def run(self, image_paths: list):
        for img_idx, image_path in enumerate(image_paths):
            base_filename, _ = os.path.splitext(os.path.basename(image_path))
            print(f"\n\n===== Processing Image {img_idx + 1}/{len(image_paths)}: {base_filename} =====\n")
            # logging.info(f"===== Processing Image {img_idx + 1}/{len(image_paths)}: {base_filename} =====")
            
            with open(image_path, "rb") as f:
                original_image_bytes = f.read()

            # Save the original image for reference
            original_image = Image.open(io.BytesIO(original_image_bytes))
            original_save_path = os.path.join(self.cfg.logging.results_dir, f"{base_filename}_original.jpg")
            original_image.save(original_save_path)
            print(f"Saved original image to: {original_save_path}")
            
            print("\n--- Starting Optimization ---")
            # logging.info("--- Starting Optimization ---")

            for i in range(self.cfg.optimizer.iterations):
                # Print and log current iteration and prompt
                iteration_info = f"\n>> ITERATION {i + 1}/{self.cfg.optimizer.iterations} <<"
                print(iteration_info)
                # logging.info(iteration_info)
                prompt_info = f"Current Prompt:\n{self.image_editing_prompt.value}"
                print(prompt_info)
                # logging.info(prompt_info)

                # Zero gradients and run the forward pass
                self.optimizer.zero_grad()
                edited_image, edited_image_bytes = self.image_editing_model(self.image_editing_prompt, original_image_bytes)
                
                # Save the edited image for this iteration
                edited_image_save_path = os.path.join(self.cfg.logging.results_dir, f"{base_filename}_iter_{i+1}.jpg")
                edited_image.save(edited_image_save_path)
                print(f"Saved edited image to: {edited_image_save_path}")

                # Create placeholder to link the computation graph
                image_output_placeholder = Variable(
                    "The edited image produced by the prompt",
                    role_description="A placeholder representing the output of the image editing model",
                    predecessors=[self.image_editing_prompt]
                )

                # Get VLM evaluation and calculate loss
                vlm_choice = self.evaluator_vlm(image_output_placeholder, original_image_bytes, edited_image_bytes)
                vlm_response_info = f"VLM Response:\n{vlm_choice.value}"
                print(vlm_response_info)
                # logging.info(vlm_response_info)

                loss = self.loss_fn(vlm_choice)
                loss_info = f"Loss (Critique):\n{loss.value}"
                print(loss_info)
                # logging.info(loss_info)

                # Run backward pass and update the prompt
                loss.backward()
                gradient_info = f"Prompt Gradients:\n{self.image_editing_prompt.gradients}"
                print(gradient_info)
                # logging.info(gradient_info)

                self.optimizer.step()
                new_prompt_info = f"New Optimized Prompt:\n{self.image_editing_prompt.value}"
                print(new_prompt_info)
                # logging.info(new_prompt_info)
                print("-" * 30)