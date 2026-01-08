#!/usr/bin/env python3
"""
Generate Prolific survey data by pairing zero-shot and final images.
"""

import argparse
import csv
import os
import random
import re
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def parse_filename(filename):
    """
    Parse image filename to extract category, product ID, and status.
    Expected format: pair-X_CAT_ID_vs_CAT_ID_CAT_ID_STATUS.jpg
    Returns: (category, product_id, status) or None if no match
    """
    # Match the pattern at the end: CATEGORY_ID_STATUS.jpg
    match = re.search(r'_([A-Z]+)_([a-f0-9]+)_(zero-shot|final)\.jpg$', filename)
    if not match:
        return None
    category = match.group(1)
    product_id = match.group(2)
    status = match.group(3)
    return (category, product_id, status)


def collect_images(data_dir):
    """
    Collect all zero-shot and final images from the data directory.
    Returns: dict with structure {category: {product_id: {status: filename}}}
    """
    images = defaultdict(lambda: defaultdict(dict))
    data_path = Path(data_dir)

    # Find all images ending in _zero-shot.jpg or _final.jpg
    for img_file in data_path.glob("*.jpg"):
        filename = img_file.name
        if filename.endswith('_zero-shot.jpg') or filename.endswith('_final.jpg'):
            parsed = parse_filename(filename)
            if parsed:
                category, product_id, status = parsed
                images[category][product_id][status] = filename

    return images


def create_pairs(images):
    """
    Create pairs of (zero-shot from product A, final from product B) for each category.
    Returns: tuple of two lists (pairs_set1, pairs_set2)
    where each is a list of tuples (category, img1_data, img2_data)
    and img_data is (filename, product_id, status)
    """
    pairs_set1 = []
    pairs_set2 = []

    for category, products in images.items():
        product_ids = list(products.keys())
        prod1_id, prod2_id = product_ids

        # Get zero-shot and final for each product
        prod1_zero = products[prod1_id]['zero-shot']
        prod1_final = products[prod1_id]['final']
        prod2_zero = products[prod2_id]['zero-shot']
        prod2_final = products[prod2_id]['final']

        # Set 1: zero-shot from prod1, final from prod2
        pairs_set1.append((
            category,
            (prod1_zero, prod1_id, 'zero-shot'),
            (prod2_final, prod2_id, 'final')
        ))

        # Set 2: zero-shot from prod2, final from prod1
        pairs_set2.append((
            category,
            (prod2_zero, prod2_id, 'zero-shot'),
            (prod1_final, prod1_id, 'final')
        ))

    return pairs_set1, pairs_set2


def main():
    parser = argparse.ArgumentParser(
        description='Generate Prolific survey data from image pairs'
    )
    parser.add_argument(
        '--base-url',
        required=True,
        help='Base URL for the images (will be prepended to image filenames)'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Data directory'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory name (default: prolific_survey)',
        default='prolific_survey'
    )

    args = parser.parse_args()

    # Set up paths
    data_dir = Path(args.data_dir)
    data_dir = data_dir.resolve()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'

    # Create output directories
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading images from: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Collect and pair images
    images = collect_images(data_dir)
    pairs_set1, pairs_set2 = create_pairs(images)

    print(f"Found {len(pairs_set1)} pairs in set 1 and {len(pairs_set2)} pairs in set 2")

    base_url = args.base_url.rstrip('/')

    # Copy all unique images and generate both CSVs
    all_pairs = pairs_set1 + pairs_set2
    csv_data_set1 = []
    csv_data_set2 = []
    copied_images = set()

    for pair_list, csv_data in [(pairs_set1, csv_data_set1), (pairs_set2, csv_data_set2)]:
        for category, (img1_orig, img1_id, img1_status), (img2_orig, img2_id, img2_status) in tqdm(pair_list, desc="Processing pairs"):
            # Rename images to CATEGORY_ID_STATUS.jpg
            img1_new = f"{category}_{img1_id}_{img1_status}.jpg"
            img2_new = f"{category}_{img2_id}_{img2_status}.jpg"

            # Copy images to output directory (only if not already copied)
            if img1_new not in copied_images:
                shutil.copy2(data_dir / img1_orig, images_dir / img1_new)
                copied_images.add(img1_new)
            if img2_new not in copied_images:
                shutil.copy2(data_dir / img2_orig, images_dir / img2_new)
                copied_images.add(img2_new)

            # Create full paths with base URL
            img1_full_path = f"{base_url}/{img1_new}"
            img2_full_path = f"{base_url}/{img2_new}"

            # Randomly order the images
            if random.random() < 0.5:
                csv_data.append({
                    'category': category.lower(),
                    'image1_path': img1_full_path,
                    'image2_path': img2_full_path,
                    'image1_id': img1_id,
                    'image2_id': img2_id
                })
            else:
                csv_data.append({
                    'category': category.lower(),
                    'image1_path': img2_full_path,
                    'image2_path': img1_full_path,
                    'image1_id': img2_id,
                    'image2_id': img1_id
                })

    # Write both CSVs
    fieldnames = ['category', 'image1_path', 'image2_path', 'image1_id', 'image2_id']

    csv_path_set1 = output_dir / 'survey_pairs_set1.csv'
    with open(csv_path_set1, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data_set1)

    csv_path_set2 = output_dir / 'survey_pairs_set2.csv'
    with open(csv_path_set2, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data_set2)

    print(f"Generated {len(csv_data_set1)} pairs in set 1 and {len(csv_data_set2)} pairs in set 2")
    print(f"Images saved to: {images_dir}")
    print(f"CSVs saved to: {csv_path_set1} and {csv_path_set2}")


if __name__ == '__main__':
    main()
