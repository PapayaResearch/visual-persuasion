import os
import logging
import dataclasses
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel


@dataclasses.dataclass
class DistillationEdit:
    """
    Applies a distillation prompt to each image in a single zero-shot edit.
    """
    name: str
    distillation_prompt: str
    image_editing_model: ImageModel

    def _apply_distillation(
        self,
        image_path: str,
        results_dir: str
    ) -> dict:
        """
        Apply the distillation prompt to a single image and save the result.
        """
        base_img = os.path.splitext(os.path.basename(image_path))[0]

        logging.info(f"\n{'='*80}\n")
        logging.info(f"Applying distillation to: {base_img}\n")
        logging.info(f"{'='*80}\n")

        # Load original image
        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

        # Save original image copy
        original_copy_path = os.path.join(results_dir, f"{base_img}_original.jpg")
        with open(original_copy_path, "wb") as f:
            f.write(original_image_bytes)

        # Apply distillation edit
        logging.info(f"Distillation prompt: {self.distillation_prompt}\n")
        edited_image, edited_image_bytes = self.image_editing_model.edit(
            self.distillation_prompt,
            original_image_bytes
        )

        # Save result
        output_path = os.path.join(results_dir, f"{base_img}_distillation.jpg")
        edited_image.save(output_path)
        logging.info(f"Saved distillation edit to: {output_path}\n")

        return {
            "image_name": base_img,
            "output_path": output_path
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Apply the distillation prompt to all images.
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"🧪 Starting Distillation Edit")
        logging.info(f"   Total images: {len(image_paths)}")
        logging.info(f"   Distillation: {self.distillation_prompt}")
        logging.info(f"   Max workers: {max_workers}")
        logging.info(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        results = []

        with tqdm(total=len(image_paths), desc="Images completed", unit="image") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._apply_distillation,
                        image_path,
                        results_dir
                    ): image_path
                    for image_path in image_paths
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Generate summary
        summary_path = os.path.join(results_dir, "global_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Distillation Edit - Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total images processed: {len(image_paths)}\n")
            f.write(f"Distillation prompt:\n{self.distillation_prompt}\n\n")
            f.write(f"Successful edits:\n")
            for r in results:
                f.write(f"  {r['image_name']}\n")

        logging.info(f"\n{'='*80}")
        logging.info(f"✅ Distillation Edit Complete!")
        logging.info(f"   Images processed: {len(image_paths)}")
        logging.info(f"   Summary saved to: {summary_path}")
        logging.info(f"{'='*80}\n")

        return results
