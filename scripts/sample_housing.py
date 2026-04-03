import os
import argparse
import shutil
import pandas as pd
from tqdm.auto import tqdm


def copy_images_to_output(df: pd.DataFrame, dataset_dir: str, output_dir: str) -> None:
    """Copy sampled images to output directory organized by category."""
    output_dir = os.path.join(output_dir, "sampled_houses")
    os.makedirs(output_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        source_path = os.path.join(dataset_dir, "%d_frontal.jpg" % row["id"])
        shutil.copy2(source_path, os.path.join(output_dir, os.path.basename(source_path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the dataset directory (small dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory for sampled images"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of images to sample per category after similarity-based selection"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    args = parser.parse_args()

    df = pd.read_csv(
        os.path.join(args.dataset_dir, "HousesInfo.txt"),
        sep=" ",
        header=None
    )
    df.columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "id"}, inplace=True)
    df["id"] = df["id"] + 1  # IDs should be 1-indexed

    prices = df.price.describe()
    print("Price statistics:")
    print(prices)

    areas = df.area.describe()
    print("Area statistics:")
    print(areas)

    final_sampled_df = df[
        (df.price >= prices["mean"] - (prices["std"] / 2)) &
        (df.price <= prices["mean"] + (prices["std"] / 2))
    ]

    print(f"Sampled {len(final_sampled_df)} houses within 1/2 standard deviation of the mean price.")

    final_sampled_df = final_sampled_df[
        (final_sampled_df.area >= areas["mean"] - (areas["std"] / 2)) &
        (final_sampled_df.area <= areas["mean"] + (areas["std"] / 2))
    ]

    print(f"Sampled {len(final_sampled_df)} houses within 1/2 standard deviation of the mean area.")

    if len(final_sampled_df) > args.max_samples:
        final_sampled_df = final_sampled_df.sample(
            n=args.max_samples,
            random_state=args.seed
        )
        print(f"Reduced to {args.max_samples} samples.")

    copy_images_to_output(final_sampled_df, args.dataset_dir, args.output_dir)

if __name__ == "__main__":
    main()
