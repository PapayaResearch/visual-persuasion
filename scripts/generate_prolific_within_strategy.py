#!/usr/bin/env python3
"""
Generate Prolific survey for within-strategy evaluation (original vs zero-shot vs final).
"""

import argparse
import csv
import os
import random
import re
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm


# Task-specific questions for Prolific survey
TASK_QUESTIONS = {
    'hotels': 'Which hotel would you book?',
    'houses': 'Which house would you buy?',
    'people': 'Who would you hire for a job?'
}

# Category display names for products task (uppercase -> readable)
CATEGORY_DISPLAY_NAMES = {
    'BACKPACK': 'backpack',
    'BED': 'bed',
    'BREAD': 'bread',
    'CELLULARPHONECASE': 'phone case',
    'CHAIR': 'chair',
    'CLOCK': 'clock',
    'FINERING': 'ring',
    'HANDBAG': 'handbag',
    'HAT': 'hat',
    'LAMP': 'lamp',
    'PILLOW': 'pillow',
    'PLANTER': 'planter',
    'RUG': 'rug',
    'SAUTEFRYPAN': 'pan',
    'SHOES': 'shoes',
    'SOFA': 'sofa',
    'SUITCASE': 'suitcase',
    'VASE': 'vase',
    'WALLART': 'wall art',
    'WALLET': 'wallet',
}


def parse_filename(filename):
    """
    Parse image filename to extract category, image_id, and status.
    Expected format: CATEGORY_ID_STATUS.jpg
    Returns: (category, image_id, status) or None if no match
    """
    match = re.match(r'^([A-Za-z_]+)_([a-z0-9]+)_(original|zero-shot|final)\.jpg$', filename)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def find_config_file(img_path):
    """
    Find config.yaml in the same directory or parent directories.
    Returns path to config.yaml or None.
    """
    current = img_path.parent
    for _ in range(5):  # Search up to 5 levels up
        config_path = current / "config.yaml"
        if config_path.exists():
            return config_path
        current = current.parent
    return None


def collect_images(data_dir):
    """
    Recursively collect all images with original/zero-shot/final status.
    Returns: {strategy: {task: {category: {image_id: {status: filepath}}}}}
    """
    images = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    data_path = Path(data_dir)

    # Find all jpg files recursively
    for img_path in data_path.rglob("*.jpg"):
        parsed = parse_filename(img_path.name)
        if not parsed:
            continue

        category, image_id, status = parsed

        # Find config to get strategy and task
        config_path = find_config_file(img_path)
        if not config_path:
            print(f"Warning: Could not find config.yaml for {img_path}, skipping")
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        strategy = config['strategy']['name']
        task = config['task']['name']

        images[strategy][task][category][image_id][status] = img_path

    return images


def create_participant_set(category_images, seed):
    """
    Create one participant set for a category by randomly assigning one status per ID.

    Args:
        category_images: {image_id: {status: filepath}}
        seed: Random seed

    Returns:
        List of (id1, status1, path1, id2, status2, path2) tuples
    """
    random.seed(seed)

    # For each ID, randomly pick one status
    assigned = {}
    for image_id, statuses in category_images.items():
        available_statuses = list(statuses.keys())
        if len(available_statuses) == 0:
            continue
        chosen_status = random.choice(available_statuses)
        assigned[image_id] = (chosen_status, statuses[chosen_status])

    # Generate all pairs (skip same status comparisons)
    pairs = []
    for (id1, (status1, path1)), (id2, (status2, path2)) in combinations(assigned.items(), 2):
        # Skip if both images have the same status
        if status1 == status2:
            continue
        pairs.append((id1, status1, path1, id2, status2, path2))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Generate Prolific survey for within-strategy evaluation'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing optimization results'
    )
    parser.add_argument(
        '--base-url',
        required=True,
        help='Base URL for the images (will be prepended to image filenames)'
    )
    parser.add_argument(
        '--output-dir',
        default='prolific_within_strategy',
        help='Output directory (default: prolific_within_strategy)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--max-comparisons',
        type=int,
        default=160,
        help='Maximum comparisons per task (default: 160)'
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning images from: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Max comparisons per task: {args.max_comparisons}")

    # Collect all images
    images = collect_images(args.data_dir)

    if not images:
        print("Error: No images found")
        return

    # Generate comparisons for all strategy-task combinations
    all_comparisons = []
    base_url = args.base_url.rstrip('/')
    copied_images = set()

    for strategy in sorted(images.keys()):
        for task in sorted(images[strategy].keys()):
            print(f"\nProcessing {strategy} - {task}...")

            # Generate all pairs grouped by category
            pairs_by_category = {}
            for category in sorted(images[strategy][task].keys()):
                category_images = images[strategy][task][category]
                pairs = create_participant_set(category_images, args.seed)
                pairs_by_category[category] = [(category,) + p for p in pairs]

            # Count total pairs
            total_pairs = sum(len(pairs) for pairs in pairs_by_category.values())
            print(f"  Generated {total_pairs} pairs across {len(pairs_by_category)} categories")

            # Subsample if needed, balanced across categories
            strategy_task_pairs = []
            if total_pairs > args.max_comparisons:
                random.seed(args.seed)
                per_category = args.max_comparisons // len(pairs_by_category)
                for category, pairs in sorted(pairs_by_category.items()):
                    sample_size = min(per_category, len(pairs))
                    sampled = random.sample(pairs, sample_size)
                    strategy_task_pairs.extend(sampled)
                print(f"  Subsampled to {len(strategy_task_pairs)} pairs (~{per_category} per category)")
            else:
                for pairs in pairs_by_category.values():
                    strategy_task_pairs.extend(pairs)

            # Process pairs
            for category, id1, status1, path1, id2, status2, path2 in tqdm(strategy_task_pairs, desc=f"  {strategy}-{task}"):
                # Create new filenames (need strategy prefix to avoid overwrites)
                img1_new = f"{strategy}_{category}_{id1}_{status1}.jpg"
                img2_new = f"{strategy}_{category}_{id2}_{status2}.jpg"

                # Copy images if not already copied
                if img1_new not in copied_images:
                    shutil.copy2(path1, images_dir / img1_new)
                    copied_images.add(img1_new)
                if img2_new not in copied_images:
                    shutil.copy2(path2, images_dir / img2_new)
                    copied_images.add(img2_new)

                # Create URLs
                img1_url = f"{base_url}/{img1_new}"
                img2_url = f"{base_url}/{img2_new}"

                # Get question for this task
                if task == 'products':
                    category_display = CATEGORY_DISPLAY_NAMES.get(category, category.lower())
                    question = f"Which {category_display} would you buy?"
                else:
                    question = TASK_QUESTIONS.get(task, 'Which one do you prefer?')

                # Randomly order the images to avoid position bias
                if random.random() < 0.5:
                    all_comparisons.append({
                        'strategy': strategy,
                        'task': task,
                        'category': category,
                        'question': question,
                        'image1_id': id1,
                        'image1_status': status1,
                        'image1_url': img1_url,
                        'image2_id': id2,
                        'image2_status': status2,
                        'image2_url': img2_url
                    })
                else:
                    all_comparisons.append({
                        'strategy': strategy,
                        'task': task,
                        'category': category,
                        'question': question,
                        'image1_id': id2,
                        'image1_status': status2,
                        'image1_url': img2_url,
                        'image2_id': id1,
                        'image2_status': status1,
                        'image2_url': img1_url
                    })

    # Write CSV
    csv_path = output_dir / 'within_strategy_comparisons.csv'
    fieldnames = ['strategy', 'task', 'category', 'question', 'image1_id', 'image1_status', 'image1_url',
                  'image2_id', 'image2_status', 'image2_url']

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comparisons)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_comparisons)} total comparisons")
    print(f"Images saved to: {images_dir}")
    print(f"CSV saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
