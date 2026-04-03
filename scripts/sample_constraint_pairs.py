#!/usr/bin/env python3
"""
Sample matched original/final pairs for constraint satisfaction validation.

For each task subdirectory, assigns pairs to participants such that:
  - Each participant sees n_samples pairs per task
  - All participants combined cover the full pool (n_participants × n_samples)
  - For tasks with small categories (e.g. products: 5 each), whole categories are
    assigned per participant. For tasks with few/large categories, pairs are divided
    equally across participants within each category.

Output:
  output_dir/images/  — flat directory with all original and final images
  output_dir/constraint_comparisons.csv  — participant assignments with image URLs
"""

import argparse
import csv
import random
import re
import shutil
import yaml
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def parse_filename(filename):
    """
    Parse image filename to extract category, image_id, and status.
    Expected format: CATEGORY_ID_(original|final).jpg
    Returns: (category, image_id, status) or None if no match.
    """
    match = re.match(r"^([A-Za-z_]+)_([a-z0-9]+)_(original|final)\.jpg$", filename)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def copy_and_record(all_rows, copied_images, images_dir, base_url,
                    participant_id, task, category, image_id, orig_path, final_path):
    for path in [orig_path, final_path]:
        if path.name not in copied_images:
            shutil.copy2(path, images_dir / path.name)
            copied_images.add(path.name)
    all_rows.append({
        "participant_id": participant_id,
        "task": task,
        "category": category,
        "image_id": image_id,
        "original_url": f"{base_url}/{orig_path.name}",
        "final_url": f"{base_url}/{final_path.name}",
    })


def main():
    parser = argparse.ArgumentParser(
        description="Sample matched original/final pairs for constraint satisfaction validation"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory whose subdirectories each contain *_original.jpg and *_final.jpg files"
    )
    parser.add_argument(
        "--base-url", required=True,
        help="Base URL for images (prepended to image filenames in the CSV)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("constraint_samples"),
        help="Output directory (default: constraint_samples)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of pairs per participant per task (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url.rstrip("/")
    all_rows = []
    copied_images = set()

    for task_dir in sorted(args.input_dir.iterdir()):
        with open(task_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        task = config["task"]["name"]
        print(f"\nProcessing task: {task}")

        originals = {}
        finals = {}
        for img_path in task_dir.glob("*.jpg"):
            parsed = parse_filename(img_path.name)
            if not parsed:
                continue
            category, image_id, status = parsed
            if status == "original":
                originals[(category, image_id)] = img_path
            else:
                finals[(category, image_id)] = img_path

        pairs_by_category = defaultdict(list)
        for (category, image_id), orig_path in originals.items():
            pairs_by_category[category].append((image_id, orig_path, finals[(category, image_id)]))

        categories = list(pairs_by_category.keys())
        min_cat_size = min(len(p) for p in pairs_by_category.values())
        n_cats = len(categories)
        total_pairs = sum(len(p) for p in pairs_by_category.values())
        n_participants = total_pairs // args.n_samples

        if min_cat_size < args.n_samples and min_cat_size * n_cats > args.n_samples:
            # Small categories (e.g. products: 5 each) — assign whole categories per participant
            cats_per_participant = args.n_samples // min_cat_size
            random.shuffle(categories)
            selected_cats = categories[:n_participants * cats_per_participant]

            for p in tqdm(range(n_participants), desc=f"  {task} (small categories)"):
                participant_cats = selected_cats[p * cats_per_participant:(p + 1) * cats_per_participant]
                for category in participant_cats:
                    for image_id, orig_path, final_path in pairs_by_category[category]:
                        copy_and_record(all_rows, copied_images, images_dir, base_url,
                                        p + 1, task, category, image_id, orig_path, final_path)
        else:
            # Few/large categories (e.g. hotels, houses) — divide pairs equally per category
            per_cat_per_participant = args.n_samples // n_cats
            total_per_cat = n_participants * per_cat_per_participant

            cat_pairs = {}
            for category in categories:
                pairs = list(pairs_by_category[category])
                random.shuffle(pairs)
                cat_pairs[category] = pairs[:total_per_cat]

            for p in tqdm(range(n_participants), desc=f"  {task}"):
                for category in categories:
                    chunk = cat_pairs[category][p * per_cat_per_participant:(p + 1) * per_cat_per_participant]
                    for image_id, orig_path, final_path in chunk:
                        copy_and_record(all_rows, copied_images, images_dir, base_url,
                                        p + 1, task, category, image_id, orig_path, final_path)

    csv_path = args.output_dir / "constraint_comparisons.csv"
    fieldnames = ["participant_id", "task", "category", "image_id", "original_url", "final_url"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_rows)} rows")
    print(f"Images saved to: {images_dir}")
    print(f"CSV saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
