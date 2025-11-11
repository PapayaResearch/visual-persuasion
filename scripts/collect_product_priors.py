"""Analyzes product images using an LLM to extract perceived product attributes."""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pydantic import Field
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.wrappers import IOSchema, LanguageModel

load_dotenv()


class ProductImageInput(IOSchema):
    image: bytes = Field(description="Product image to analyze")

    def to_formatted_string(self) -> str:
        return """Please analyze this product image and provide your assessment of the following attributes:
1. Price: Estimated price in US dollars (numeric value only)
2. Rating: Perceived consumer rating from 0-100
3. Usability: How versatile/useful the product appears (0-100)
4. Appealing: Aesthetic appeal and attractiveness (0-100)
5. Quality: Perceived build quality and craftsmanship (0-100)

Base your assessment purely on what you observe in the image."""


class ProductAnalysis(IOSchema):
    price: float = Field(description="Estimated price in US dollars")
    rating: int = Field(ge=0, le=100, description="Consumer rating from 0-100")
    usability: int = Field(ge=0, le=100, description="Versatility/usefulness score 0-100")
    appealing: int = Field(ge=0, le=100, description="Aesthetic appeal score 0-100")
    quality: int = Field(ge=0, le=100, description="Perceived quality score 0-100")


def setup_analyzer(llm_config_name: str) -> LanguageModel:
    llm_config_path = Path(__file__).parent.parent / "nudging" / "conf" / "llm" / f"{llm_config_name}.yaml"
    llm_config = OmegaConf.load(llm_config_path)
    api_call = hydra.utils.instantiate(llm_config)

    system_prompt = """You are an expert product analyst. Your task is to analyze product images and provide objective assessments of their attributes based solely on visual information. Be realistic and consistent in your evaluations."""

    return LanguageModel(
        system_prompt=system_prompt,
        input_schema=ProductImageInput,
        output_schema=ProductAnalysis,
        api_call=api_call,
        enable_json_schema_validation=True
    )


def analyze_image(llm, image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = llm.get_response(image=image_bytes)
    result_dict = result.model_dump()
    result_dict["filename"] = image_path.name

    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Analyze product images using an LLM")
    parser.add_argument("--data", required=True, help="Directory containing .jpg product images")
    parser.add_argument("--output", "-o", default="product_analysis.csv", help="Output CSV filename")
    parser.add_argument("--llm", "-l", default="gpt-5", help="LLM config name from nudging/conf/llm/")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    analyzer = setup_analyzer(args.llm)

    image_dir = Path(args.data)
    image_files = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) +
        list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG"))
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(analyze_image, analyzer, img): img for img in image_files}

        for future in tqdm(as_completed(futures), total=len(image_files), desc="Analyzing images"):
            results.append(future.result())

    results.sort(key=lambda x: x["filename"])

    # Create output directory structure: result_priors/{model_name}/{data_dir_name}/
    data_dir_name = image_dir.name
    output_dir = Path("result_priors") / args.llm / data_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    print(f"Saving results to: {output_path}")

    fieldnames = ["filename", "price", "rating", "usability", "appealing", "quality"]
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
