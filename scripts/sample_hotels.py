import argparse
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Sample images equally from category subfolders"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to input directory with category subfolders containing .jpg images"
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
        help="Total number of images to sample (divided equally among categories)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )

    args = parser.parse_args()

    # Expand paths and create output directory
    args.input_dir = args.input_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(args.seed)

    # Find all category subfolders
    categories = [d for d in args.input_dir.iterdir() if d.is_dir()]

    # Calculate samples per category
    samples_per_category = args.n_samples // len(categories)

    print(f"Found {len(categories)} categories")
    print(f"Sampling {samples_per_category} images per category")

    # Sample from each category
    for category_dir in tqdm(sorted(categories), desc="Sampling categories"):
        # Get all .jpg files in this category
        jpg_files = list(category_dir.glob("*.jpg"))

        # Sample the appropriate number (or all if fewer available)
        n_to_sample = min(samples_per_category, len(jpg_files))
        sampled_files = random.sample(jpg_files, n_to_sample)

        # Create output category directory
        output_category_dir = args.output_dir / category_dir.name
        output_category_dir.mkdir(parents=True, exist_ok=True)

        # Copy sampled files
        for img_path in sampled_files:
            dest_path = output_category_dir / img_path.name
            shutil.copy2(img_path, dest_path)

if __name__ == "__main__":
    main()
