import os
import logging
import dataclasses
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel


@dataclasses.dataclass
class FormulaEdit:
    """
    Applies a learned formula prompt to each image in a single zero-shot edit.
    """
    name: str
    formula_prompt: str
    image_editing_model: ImageModel

    def _apply_formula(
        self,
        image_path: str,
        results_dir: str
    ) -> dict:
        """
        Apply the formula prompt to a single image and save the result.
        """
        base_img = os.path.splitext(os.path.basename(image_path))[0]

        logging.info(f"\n{'='*80}\n")
        logging.info(f"Applying formula to: {base_img}\n")
        logging.info(f"{'='*80}\n")

        # Load original image
        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

        # Save original image copy
        original_copy_path = os.path.join(results_dir, f"{base_img}_original.jpg")
        with open(original_copy_path, "wb") as f:
            f.write(original_image_bytes)

        # Apply formula edit
        logging.info(f"Formula prompt: {self.formula_prompt}\n")
        edited_image, edited_image_bytes = self.image_editing_model.edit(
            self.formula_prompt,
            original_image_bytes
        )

        # Save result
        output_path = os.path.join(results_dir, f"{base_img}_formula.jpg")
        edited_image.save(output_path)
        logging.info(f"Saved formula edit to: {output_path}\n")

        return {
            "image_name": base_img,
            "output_path": output_path
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Apply the formula prompt to all images.
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"🧪 Starting Formula Edit")
        logging.info(f"   Total images: {len(image_paths)}")
        logging.info(f"   Formula: {self.formula_prompt}")
        logging.info(f"   Max workers: {max_workers}")
        logging.info(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        results = []

        with tqdm(total=len(image_paths), desc="Images completed", unit="image") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._apply_formula,
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
            f.write(f"Formula Edit - Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total images processed: {len(image_paths)}\n")
            f.write(f"Formula prompt:\n{self.formula_prompt}\n\n")
            f.write(f"Successful edits:\n")
            for r in results:
                f.write(f"  {r['image_name']}\n")

        logging.info(f"\n{'='*80}")
        logging.info(f"✅ Formula Edit Complete!")
        logging.info(f"   Images processed: {len(image_paths)}")
        logging.info(f"   Summary saved to: {summary_path}")
        logging.info(f"{'='*80}\n")

        return results
