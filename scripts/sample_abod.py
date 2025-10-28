import argparse
import gzip
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm

# TODO: Take as input argument
TARGET_CATEGORIES = [
    "CELLULAR_PHONE_CASE",
    "SHOES",
    "CHAIR",
    "FINERING",
    "SOFA",
    "RUG",
    "HANDBAG",
    "HAT",
    "WALL_ART",
    "LAMP",
    "SUITCASE",
    "PILLOW",
    "PLANTER",
    "BACKPACK",
    "BED",
    "BATTERY",
    "DESK",
    "WALLET",
    "SAUTE_FRY_PAN",
    "BREAD"
]


def load_json_metadata(metadata_dir: Path) -> pd.DataFrame:
    """Load and combine all .json.gz files from the metadata directory."""
    json_files = sorted(metadata_dir.glob("*.json.gz"))

    all_records = []
    for json_file in tqdm(json_files, desc="Reading JSON files"):
        with gzip.open(json_file, "rt", encoding="utf-8") as f:
            for line in f:
                all_records.append(json.loads(line))

    return pd.DataFrame(all_records)


def filter_by_categories(product_df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """Filter products by their product_type."""
    print(f"Filtering by categories: {categories}")
    product_df["product_type_str"] = product_df["product_type"].apply(lambda x: x[0]["value"] if x else None)
    mask = product_df["product_type_str"].str.lower().isin([c.lower() for c in categories])
    return product_df[mask].copy()


def sample_images_per_category(df: pd.DataFrame, n_samples: int, random_seed: int = 42) -> pd.DataFrame:
    """Sample n images per category."""
    sampled_dfs = [
        df[df["product_type_str"] == category].sample(n=n_samples, random_state=random_seed)
        for category in df["product_type_str"].unique()
    ]
    return pd.concat(sampled_dfs, ignore_index=True)


def copy_images_to_output(df: pd.DataFrame, dataset_dir: Path, output_dir: Path) -> None:
    """Copy sampled images to output directory organized by category."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        category_dir = output_dir / row["product_type_str"]
        category_dir.mkdir(parents=True, exist_ok=True)
        source_path = dataset_dir / row["path"]
        dest_path = category_dir / source_path.name
        shutil.copy2(source_path, dest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Sample images from a dataset by product categories"
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory (small dataset)"
    )

    parser.add_argument(
        "--image-metadata",
        type=Path,
        required=True,
        help="Path to the image metadata CSV file (.csv.gz)"
    )

    parser.add_argument(
        "--product-metadata-dir",
        type=Path,
        required=True,
        help="Path to directory containing product metadata JSON files (.json.gz)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory for sampled images"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of images to sample per category"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )

    args = parser.parse_args()

    product_df = load_json_metadata(args.product_metadata_dir)
    image_df = pd.read_csv(args.image_metadata, compression="gzip")
    filtered_df = filter_by_categories(product_df, TARGET_CATEGORIES)
    merged_df = filtered_df.merge(image_df, left_on="main_image_id", right_on="image_id", how="inner")
    sampled_df = sample_images_per_category(merged_df, args.n_samples, args.seed)
    copy_images_to_output(sampled_df, args.dataset_dir, args.output_dir)

if __name__ == "__main__":
    main()
