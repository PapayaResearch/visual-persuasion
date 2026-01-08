"""Analyzes product images using paired comparisons to determine relative attributes."""

import sys
import csv
sys.path.append(str(Path(__file__).parent.parent))
import argparse
import hydra
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from itertools import combinations
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pydantic import Field
from utils.wrappers import IOSchema, LanguageModel
from tqdm import tqdm


load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Analyze product images using paired comparisons")
    parser.add_argument("--data", required=True, help="Directory containing .jpg product images")
    parser.add_argument("--output", "-o", default="paired_product_analysis.csv", help="Output CSV filename")
    parser.add_argument("--llm", "-l", default="gpt-5", help="LLM config name from nudging/conf/llm/")
    parser.add_argument("--max-workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--category-prefix", action="store_true", help="Only compare images within same category (assumes format: category_itemname.jpg)")
    args = parser.parse_args()

    analyzer = setup_analyzer(args.llm)

    image_dir = Path(args.data)
    image_files = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) +
        list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG"))
    )

    # Generate pairs
    if args.category_prefix:
        # Group by category prefix (before first underscore)
        image_categories = {}
        for image_path in image_files:
            base_name = image_path.stem
            category = base_name.split("_")[0]
            image_categories.setdefault(category, [])
            image_categories[category].append(image_path)

        # Generate pairs within each category
        pairs = []
        for category, paths in image_categories.items():
            category_pairs = list(combinations(paths, 2))
            pairs.extend(category_pairs)

        print(f"Found {len(image_categories)} categories with {len(pairs)} total pairs")
    else:
        # Compare all images against each other
        pairs = list(combinations(image_files, 2))
        print(f"Comparing all {len(image_files)} images: {len(pairs)} total pairs")

    # Analyze pairs
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(compare_image_pair, analyzer, pair[0], pair[1]): pair
            for pair in pairs
        }

        for future in tqdm(as_completed(futures), total=len(pairs), desc="Comparing pairs"):
            results.append(future.result())

    # Sort results by image names for consistency
    results.sort(key=lambda x: (x["image_a"], x["image_b"]))

    # Create output directory structure: result_priors/{model_name}/{data_dir_name}/
    data_dir_name = image_dir.name
    output_dir = Path("result_priors") / args.llm / data_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    print(f"Saving results to: {output_path}")

    # Write results to CSV
    fieldnames = [
        "image_a", "image_b",
        "price_winner", "price_reasoning",
        "rating_winner", "rating_reasoning",
        "usability_winner", "usability_reasoning",
        "appealing_winner", "appealing_reasoning",
        "quality_winner", "quality_reasoning"
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Completed {len(results)} paired comparisons")


class PairedProductInput(IOSchema):
    image_a: bytes = Field(description="First product image to compare")
    image_b: bytes = Field(description="Second product image to compare")
    attribute: str = Field(description="The attribute being judged")

    def to_formatted_string(self) -> str:
        attribute_prompts = {
            "price": "Which product appears more expensive? Consider visual cues like materials, finish, design complexity, and overall presentation.",
            "rating": "Which product would likely receive higher consumer ratings? Consider factors that typically drive positive reviews.",
            "usability": "Which product appears more versatile and useful? Consider the range of potential uses and practical functionality.",
            "appealing": "Which product is more aesthetically appealing? Consider visual design, attractiveness, and overall appeal.",
            "quality": "Which product appears to have better build quality and craftsmanship? Consider materials, construction, and attention to detail."
        }

        prompt = attribute_prompts.get(self.attribute, "Which product is superior?")
        return f"""Compare these two product images based on {self.attribute.upper()}.

{prompt}

Indicate which image (first or second) is superior and explain your reasoning in detail.
Base your assessment purely on what you observe in the images."""


class AttributeComparison(IOSchema):
    winner: str = Field(description="Either 'first' or 'second' indicating which image is superior for this attribute")
    reasoning: str = Field(description="Detailed explanation for why this image is superior")


def setup_analyzer(llm_config_name: str) -> LanguageModel:
    llm_config_path = Path(__file__).parent.parent / "nudging" / "conf" / "llm" / f"{llm_config_name}.yaml"
    llm_config = OmegaConf.load(llm_config_path)
    api_call = hydra.utils.instantiate(llm_config)

    system_prompt = """You are an expert product analyst. Your task is to compare pairs of product images on a specific attribute and determine which is superior based solely on visual information. Be objective and provide clear, detailed reasoning for your judgment."""

    return LanguageModel(
        system_prompt=system_prompt,
        input_schema=PairedProductInput,
        output_schema=AttributeComparison,
        api_call=api_call,
        enable_json_schema_validation=True
    )


def compare_image_pair(llm, image_a_path: Path, image_b_path: Path) -> dict:
    """Compare a pair of images across all attributes with independent judgments.

    For each attribute, we test both orders (A-B and B-A) to check for consistency.
    If inconsistent, we retry once. If still inconsistent, we discard that attribute judgment.
    """
    with open(image_a_path, "rb") as f:
        image_a_bytes = f.read()
    with open(image_b_path, "rb") as f:
        image_b_bytes = f.read()

    attributes = ["price", "rating", "usability", "appealing", "quality"]

    result_dict = {
        "image_a": image_a_path.name,
        "image_b": image_b_path.name,
    }

    # Make separate judgment for each attribute
    for attribute in attributes:
        # Try up to 2 times to get consistent results
        consistent = False
        for attempt in range(2):
            # First order: A is first, B is second
            result_ab = llm.get_response(
                image_a=image_a_bytes,
                image_b=image_b_bytes,
                attribute=attribute
            )

            # Second order: B is first, A is second (reversed)
            result_ba = llm.get_response(
                image_a=image_b_bytes,  # B is now first
                image_b=image_a_bytes,  # A is now second
                attribute=attribute
            )

            # Check consistency: if A won in first order, B should win in second order
            # (because the positions are swapped)
            if result_ab.winner == "first" and result_ba.winner == "second":
                # Consistent: A is superior in both
                consistent = True
                result_dict[f"{attribute}_winner"] = image_a_path.name
                result_dict[f"{attribute}_reasoning"] = result_ab.reasoning
                break
            elif result_ab.winner == "second" and result_ba.winner == "first":
                # Consistent: B is superior in both
                consistent = True
                result_dict[f"{attribute}_winner"] = image_b_path.name
                result_dict[f"{attribute}_reasoning"] = result_ab.reasoning
                break
            elif attempt == 0:
                # Inconsistent on first attempt, retry
                continue
            else:
                # Still inconsistent after retry, discard
                consistent = False
                break

        if not consistent:
            # Discard inconsistent judgment
            result_dict[f"{attribute}_winner"] = "INCONSISTENT"
            result_dict[f"{attribute}_reasoning"] = "Judgment was inconsistent across order reversals"

    return result_dict


if __name__ == "__main__":
    main()
