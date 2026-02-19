import os
import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Iterable


def write_long_csv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)


def list_all_s3_object_keys(bucket_name: str, prefix: str = "") -> list:
    """
    Lists all keys of objects in an S3 bucket.

    Kept identical in behavior to the per-script implementations it replaces.
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    all_keys: list[str] = []
    for page in tqdm(pages, desc="Listing all S3 objects", leave=False):
        if "Contents" in page:
            for obj in page["Contents"]:
                all_keys.append(obj["Key"])
    return all_keys


def drop_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def parse_cache_or_data_image_path(path: str, data_prefix: str) -> dict[str, Any]:
    """
    Parse an image S3 key into metadata columns, matching the scripts' original logic.
    """
    if path.startswith(data_prefix):
        parts = path.split("/")
        filename = parts[-1]

        map_datasets_from_dirs = {
            "abod_enhanced": "products",
            "hotels_enhanced": "hotels",
            "houses_enhanced": "houses",
            "shhq_enhanced": "people",
        }

        current_id = os.path.splitext(filename)[0].split("_")[-1]
        return {
            "dataset": map_datasets_from_dirs.get(parts[2], parts[2]),
            "image_class": filename.split("_")[0],
            "current_id": current_id,
            "other_id": None,
            "current_base_id": current_id,
            "other_base_id": None,
            "debiased_iter": None,
        }

    parts = path.split("/")
    filename = parts[-1]
    dataset = parts[2]
    image_class = filename.split("_")[0]
    filename_parts_no_ext = os.path.splitext(filename)[0].split("_")
    image_ids = [
        "%s_%s" % (filename_parts_no_ext[1], filename_parts_no_ext[2]),
        "%s_%s" % (filename_parts_no_ext[4], filename_parts_no_ext[5]),
    ]

    id_idx = int(filename_parts_no_ext[-1])
    assert len(image_ids) == 2, f"Unexpected filename format: {filename}"
    assert id_idx in [1, 2], f"Unexpected image ID index: {id_idx} in filename: {filename}"
    current_id = image_ids[id_idx - 1]
    other_id = image_ids[1 - (id_idx - 1)]
    current_base_id = current_id.split("_")[0]
    other_base_id = other_id.split("_")[0]
    debiased_iter = int(filename_parts_no_ext[6].replace("iter", ""))
    return {
        "dataset": dataset,
        "image_class": image_class,
        "current_id": current_id,
        "other_id": other_id,
        "current_base_id": current_base_id,
        "other_base_id": other_base_id,
        "debiased_iter": debiased_iter,
    }


def load_or_build_image_metadata(
    *,
    out_dir: str,
    cached_metadata: bool,
    cache_bucket: str,
    cache_prefix: str,
    data_bucket: str,
    data_prefix: str,
    img_exts: Iterable[str] = (".jpg", ".jpeg", ".png"),
) -> pd.DataFrame:
    """
    Load `image_metadata.csv` from `out_dir` if `cached_metadata` is set; otherwise:
    - list cache + data S3 keys (image extensions only),
    - parse metadata from key names,
    - write `image_metadata.csv`.
    """
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "image_metadata.csv")
    if cached_metadata:
        df = pd.read_csv(metadata_path)
        return drop_duplicated_columns(df)

    files = [f for f in list_all_s3_object_keys(cache_bucket, cache_prefix) if f.endswith(tuple(img_exts))]
    print(f"Found {len(files)} files in cache directory")
    files_data = [f for f in list_all_s3_object_keys(data_bucket, data_prefix) if f.endswith(tuple(img_exts))]
    print(f"Found {len(files_data)} files in data directory")

    img_files = files + files_data
    print(f"Found {len(img_files)} images in cache + data directory")

    df = pd.DataFrame(img_files, columns=["image_path"])
    metadata = pd.json_normalize(df["image_path"].apply(lambda p: parse_cache_or_data_image_path(p, data_prefix)))
    df = pd.concat([df, metadata], axis=1)
    df.to_csv(metadata_path, index=False)

    return drop_duplicated_columns(df)


def make_pair_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the `pair_id` column with the exact same semantics as the scripts.
    """
    df = df.copy()
    df["pair_id"] = df.apply(
        lambda row: "_".join(sorted([row.current_id, row.other_id, str(row.debiased_iter)]))
        if pd.notnull(row.other_id)
        else None,
        axis=1,
    )
    return df


def filter_valid_pair_rows(df: pd.DataFrame, *, min_per_pair: int = 2) -> pd.DataFrame:
    pair_id_counts = df["pair_id"].value_counts()
    valid_pair_ids = pair_id_counts[pair_id_counts >= min_per_pair].index
    return df[df["pair_id"].isin(valid_pair_ids)].copy()


def make_95_confidence_interval(row: pd.Series, *, metric: str) -> pd.Series:
    return pd.Series(
        {
            "mean": row[(metric, "mean")],
            "std": row[(metric, "std")],
            "count": row[(metric, "count")],
            "ci95_low": row[(metric, "mean")] - 1.96 * (row[(metric, "std")] / np.sqrt(row[(metric, "count")])),
            "ci95_high": row[(metric, "mean")] + 1.96 * (row[(metric, "std")] / np.sqrt(row[(metric, "count")])),
        }
    )


def plot_joint_metric(
    *,
    plt,
    stats_between_pairs: pd.DataFrame,
    stats_from_originals: pd.DataFrame,
    out_path: str,
    y_label: str,
    between_label: str = "Between pairs",
    from_label: str = "From originals",
    final_from_originals_label: str = "Final from originals",
    original_from_originals_label: str = "Original from originals",
) -> None:
    """
    Shared plotting helper for the scripts.

    Supports either:
    - 2-source plots: Between pairs + From originals
    - 3-source plots: Between pairs + Final from originals + Original from originals (requires `is_final`)
    """
    plot_between = stats_between_pairs.reset_index().copy()
    plot_between["source"] = between_label
    plot_from = stats_from_originals.reset_index().copy()

    has_is_final = "is_final" in plot_from.columns
    if has_is_final:
        plot_from["source"] = np.where(plot_from["is_final"], final_from_originals_label, original_from_originals_label)
    else:
        plot_from["source"] = from_label

    plot_df = pd.concat([plot_between, plot_from], axis=0, ignore_index=True)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if has_is_final:
        colors = {
            between_label: "#2a6f97",
            final_from_originals_label: "#d1495b",
            original_from_originals_label: "#edb458",
        }
        offsets = {
            between_label: -0.12,
            final_from_originals_label: 0.0,
            original_from_originals_label: 0.12,
        }
    else:
        colors = {between_label: "#2a6f97", from_label: "#d1495b"}
        offsets = {between_label: -0.08, from_label: 0.08}

    for source, group in plot_df.groupby("source", sort=False):
        x = group["debiased_iter"].astype(float).to_numpy() + offsets[source]
        y = group["mean"].to_numpy()
        yerr = np.vstack([y - group["ci95_low"].to_numpy(), group["ci95_high"].to_numpy() - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=6,
            capsize=5,
            elinewidth=2,
            color=colors[source],
            label=source,
        )
        ax.plot(x, y, linewidth=2, color=colors[source], alpha=0.85)

    ax.set_xlabel("Mitigation Passes", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xticks(sorted(plot_df["debiased_iter"].unique().tolist()))
    ax.legend(frameon=True, facecolor="white", edgecolor="#d0d0d0")
    ax.set_axisbelow(True)
    fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
