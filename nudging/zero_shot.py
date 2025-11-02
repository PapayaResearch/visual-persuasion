import os
import io
import logging
import dataclasses
from string import Template
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel

@dataclasses.dataclass
class ZeroShot:
    """
    Zero-shot visual nudging: applies editing prompts to images without optimization.
    Supports two modes:
    1. With priors: substitutes each prior into base_prompt using template variable
    2. Without priors: uses base_prompt with template variable replaced by empty string
    """
    base_prompt: str
    image_editing_model: ImageModel
    prior_variable: str = "prior"
    priors: List[str] = dataclasses.field(default_factory=list)

    def _process_single_image(
        self,
        image_path: str,
        img_idx: int,
        total_images: int,
        results_dir: str,
        pbar: tqdm
    ) -> None:
        """
        Processes a single image by applying editing prompt(s).
        """
        base_filename, _ = os.path.splitext(os.path.basename(image_path))
        logging.info(f"\n===== Processing Image {img_idx + 1}/{total_images}: {base_filename} =====\n")

        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

        original_image = Image.open(io.BytesIO(original_image_bytes))
        original_save_path = os.path.join(results_dir, f"{base_filename}_original.jpg")
        original_image.save(original_save_path)
        logging.info(f"Saved original image to: {original_save_path}\n")

        # Determine which prompts to apply
        template = Template(self.base_prompt)
        if self.priors:
            prompts_to_apply = [(template.substitute(**{self.prior_variable: prior}), f"prior-{i}")
                                for i, prior in enumerate(self.priors)]
        else:
            prompts_to_apply = [(template.substitute(**{self.prior_variable: ""}), "edit")]

        for prompt, label in prompts_to_apply:
            logging.info(f"\n>> Applying {label.upper()} <<\n")
            logging.info(f"PROMPT:\n{prompt}\n")

            # Edit image
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                prompt,
                original_image_bytes
            )

            edited_image_save_path = os.path.join(results_dir, f"{base_filename}_{label}.jpg")
            edited_image.save(edited_image_save_path)
            logging.info(f"Saved edited image to: {edited_image_save_path}\n")

            # Save prompt used
            with open(edited_image_save_path.replace(".jpg", ".txt"), "w") as f:
                f.write(prompt)

            pbar.update(1)

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs zero-shot editing for each image, optionally in parallel.
        """
        num_edits_per_image = len(self.priors) if self.priors else 1
        total_edits = len(image_paths) * num_edits_per_image

        with tqdm(
                total=total_edits,
                desc="Total progress",
                unit="edit", position=0
        ) as pbar_total:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(
                    self._process_single_image,
                    image_path,
                    idx,
                    len(image_paths),
                    results_dir,
                    pbar_total
                ): image_path
                          for idx, image_path in enumerate(image_paths)}

                for future in tqdm(
                        as_completed(futures),
                        total=len(image_paths),
                        desc="Images completed",
                        unit="image",
                        position=1,
                        leave=False
                ):
                    future.result()
