import argparse
import gzip
import json
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm
from fastembed import ImageEmbedding
from sklearn.cluster import MiniBatchKMeans

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
    "WALLET",
    "SAUTE_FRY_PAN",
    "BREAD",
    "CLOCK",
    "VASE",
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


def calculate_embeddings(df: pd.DataFrame, dataset_dir: Path) -> np.ndarray:
    """Calculate embeddings for all images in the dataframe."""
    model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision", threads=8)

    image_paths = [str(dataset_dir / path) for path in df["path"].tolist()]

    embeddings = list(tqdm(
        model.embed(images=image_paths, batch_size=32, parallel=None),
        total=len(df),
        desc="Calculating embeddings"
    ))

    return np.vstack(embeddings)

def subsample_by_similarity_clustering(df: pd.DataFrame, embeddings: np.ndarray, n_samples: int) -> pd.DataFrame:
    """Subsample images using density-based clustering.

    This method clusters the embeddings and samples proportionally from each cluster
    based on density, favoring images from denser regions (more similar images).
    """
    sampled_dfs = []

    for category in tqdm(df["product_type_str"].unique(), desc="Subsampling by similarity (clustering)"):
        df_cat = df[df["product_type_str"] == category].copy()
        cat_indices = df_cat.index.tolist()
        cat_features = embeddings[cat_indices, :]

        # Over-cluster to better estimate density
        n_clusters = n_samples * 1
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
        labels = kmeans.fit_predict(cat_features)

        # Count points per cluster (density estimate)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_weights = counts / counts.sum()

        # Sample proportionally from each cluster
        selected_indices = []
        samples_per_cluster = np.round(cluster_weights * n_samples).astype(int)

        for cluster_id, n_samples_cluster in enumerate(samples_per_cluster):
            if n_samples_cluster > 0:
                cluster_points = np.where(labels == cluster_id)[0]
                sampled = np.random.choice(
                    cluster_points,
                    size=min(n_samples_cluster, len(cluster_points)),
                    replace=False
                )
                selected_indices.extend(sampled)

        # Adjust if we're slightly over/under
        selected_indices = selected_indices[:n_samples]

        # Get the actual dataframe rows
        selected_rows = df_cat.iloc[selected_indices]
        sampled_dfs.append(selected_rows)

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
        help="Number of images to sample per category after similarity-based selection"
    )

    parser.add_argument(
        "--initial-samples",
        type=int,
        default=100,
        help="Number of images to initially sample randomly per category (e.g., 100)"
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
    initially_sampled_df = sample_images_per_category(merged_df, args.initial_samples, args.seed)

    embeddings = calculate_embeddings(initially_sampled_df, args.dataset_dir)

    final_sampled_df = subsample_by_similarity_clustering(
        initially_sampled_df,
        embeddings,
        args.n_samples
    )

    copy_images_to_output(final_sampled_df, args.dataset_dir, args.output_dir)

if __name__ == "__main__":
    main()
