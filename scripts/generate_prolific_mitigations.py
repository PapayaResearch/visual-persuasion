#!/usr/bin/env python3
"""
Generate Prolific survey for mitigations evaluation.
"""

import argparse
import csv
import random
import re
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image


TASK_QUESTIONS = {
    'hotels': 'Which hotel would you book?',
    'houses': 'Which house would you buy?',
    'people': 'Who would you hire for a job?'
}

CATEGORY_DISPLAY_NAMES = {
    'BACKPACK': 'backpack',
    'BED': 'bed',
    'BREAD': 'bread',
    'CELLULAR_PHONE_CASE': 'phone case',
    'CHAIR': 'chair',
    'CLOCK': 'clock',
    'FINERING': 'ring',
    'HANDBAG': 'handbag',
    'HAT': 'hat',
    'LAMP': 'lamp',
    'PILLOW': 'pillow',
    'PLANTER': 'planter',
    'RUG': 'rug',
    'SAUTE_FRY_PAN': 'pan',
    'SHOES': 'shoes',
    'SOFA': 'sofa',
    'SUITCASE': 'suitcase',
    'VASE': 'vase',
    'WALL_ART': 'wall art',
    'WALLET': 'wallet',
}


def parse_cache_filename(filename):
    """
    Parse: CATEGORY_IMAGEID1_STATUS1_vs_IMAGEID2_STATUS2_iterN_debiased_M.jpg
    Returns: (category, imageid1, status1, imageid2, status2, iteration, debiased_num)
    """
    pattern = r'^([A-Za-z_]+)_([a-z0-9]+)_(original|final|zero-shot)_vs_([a-z0-9]+)_(original|final|zero-shot)_iter(\d+)_debiased_([12])\.jpg$'
    match = re.match(pattern, filename)
    if not match:
        return None
    return (
        match.group(1),
        match.group(2),
        match.group(3),
        match.group(4),
        match.group(5),
        int(match.group(6)),
        int(match.group(7))
    )


def collect_cache_pairs(cache_dir, target_iteration=3):
    """
    Collect all comparison pairs from cache for specified iteration.
    Returns: {task: [(category, imageid1, status1, imageid2, status2, path1, path2)]}
    """
    pairs = defaultdict(list)
    cache_path = Path(cache_dir)

    for task in ['products', 'hotels', 'people', 'houses']:
        task_dir = cache_path / task
        if not task_dir.exists():
            continue

        comparison_files = defaultdict(dict)

        for img_path in task_dir.glob(f"*_iter{target_iteration}_debiased_*.jpg"):
            parsed = parse_cache_filename(img_path.name)
            if not parsed:
                continue

            category, imageid1, status1, imageid2, status2, iteration, debiased_num = parsed
            key = (category, imageid1, status1, imageid2, status2)
            comparison_files[key][debiased_num] = img_path

        for key, files in comparison_files.items():
            if 1 in files and 2 in files:
                try:
                    Image.open(files[1])
                    Image.open(files[2])
                    category, imageid1, status1, imageid2, status2 = key
                    pairs[task].append((category, imageid1, status1, imageid2, status2, files[1], files[2]))
                except Exception:
                    continue

    return pairs


def partition_into_sets(all_task_pairs, pairs_per_task_per_set, seed):
    """
    Partition pairs across all tasks into sets where each set has pairs_per_task_per_set
    from each task.

    Args:
        all_task_pairs: dict of {task: [pairs]}
        pairs_per_task_per_set: number of pairs per task per set (e.g., 10)
        seed: random seed

    Returns: list of (task, pair, set_number) tuples
    """
    random.seed(seed)

    # Shuffle and partition each task's pairs into chunks
    task_partitions = {}
    for task, pairs in all_task_pairs.items():
        shuffled = pairs.copy()
        random.shuffle(shuffled)

        # Split into chunks of pairs_per_task_per_set
        chunks = [shuffled[i:i + pairs_per_task_per_set]
                  for i in range(0, len(shuffled), pairs_per_task_per_set)]
        task_partitions[task] = chunks

    # Combine chunks across tasks into sets
    assigned_pairs = []
    max_sets = max(len(chunks) for chunks in task_partitions.values())

    for set_number in range(max_sets):
        for task in sorted(task_partitions.keys()):
            if set_number < len(task_partitions[task]):
                for pair in task_partitions[task][set_number]:
                    assigned_pairs.append((task, pair, set_number))

    return assigned_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output-dir', default='prolific_mitigations')
    parser.add_argument('--iteration', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pairs-per-task', type=int, default=10)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning cache: {args.cache_dir}")
    print(f"Iteration: {args.iteration}")
    print(f"Seed: {args.seed}")
    print(f"Pairs per task per set: {args.pairs_per_task}")

    pairs = collect_cache_pairs(args.cache_dir, args.iteration)

    for task in sorted(pairs.keys()):
        print(f"\n{task}: {len(pairs[task])} pairs")

    print("\nPartitioning into sets...")
    pairs_with_sets = partition_into_sets(pairs, args.pairs_per_task, args.seed)
    num_sets = max(set_num for _, _, set_num in pairs_with_sets) + 1
    print(f"Created {num_sets} sets")

    all_comparisons = []
    base_url = args.base_url.rstrip('/')
    copied_images = set()

    for task, pair, set_number in tqdm(pairs_with_sets, desc="Processing pairs"):
        category, imageid1, status1, imageid2, status2, path1, path2 = pair

        img1_new = f"{task}_{category}_{imageid1}_{status1}_vs_{imageid2}_{status2}_iter{args.iteration}_debiased_1.jpg"
        img2_new = f"{task}_{category}_{imageid1}_{status1}_vs_{imageid2}_{status2}_iter{args.iteration}_debiased_2.jpg"

        if img1_new not in copied_images:
            shutil.copy2(path1, images_dir / img1_new)
            copied_images.add(img1_new)
        if img2_new not in copied_images:
            shutil.copy2(path2, images_dir / img2_new)
            copied_images.add(img2_new)

        img1_url = f"{base_url}/{img1_new}"
        img2_url = f"{base_url}/{img2_new}"

        if task == 'products':
            category_display = CATEGORY_DISPLAY_NAMES.get(category, category.lower())
            question = f"Which {category_display} would you buy?"
        else:
            question = TASK_QUESTIONS.get(task, 'Which one do you prefer?')

        if random.random() < 0.5:
            all_comparisons.append({
                'task': task,
                'set': set_number,
                'category': category,
                'question': question,
                'image1_id': imageid1,
                'image1_status': status1,
                'image1_url': img1_url,
                'image2_id': imageid2,
                'image2_status': status2,
                'image2_url': img2_url
            })
        else:
            all_comparisons.append({
                'task': task,
                'set': set_number,
                'category': category,
                'question': question,
                'image1_id': imageid2,
                'image1_status': status2,
                'image1_url': img2_url,
                'image2_id': imageid1,
                'image2_status': status1,
                'image2_url': img1_url
            })

    csv_path = output_dir / 'mitigations_comparisons.csv'
    fieldnames = ['task', 'set', 'category', 'question', 'image1_id', 'image1_status', 'image1_url',
                  'image2_id', 'image2_status', 'image2_url']

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comparisons)

    # Calculate distribution stats
    set_task_counts = defaultdict(lambda: defaultdict(int))
    for comp in all_comparisons:
        set_task_counts[comp['set']][comp['task']] += 1

    print(f"\n{'='*60}")
    print(f"Generated {len(all_comparisons)} total comparisons across {num_sets} sets")
    print(f"\nPairs per set per task:")
    for set_num in sorted(set_task_counts.keys()):
        counts = set_task_counts[set_num]
        total = sum(counts.values())
        print(f"  Set {set_num}: {total} pairs - products: {counts.get('products', 0)}, hotels: {counts.get('hotels', 0)}, people: {counts.get('people', 0)}, houses: {counts.get('houses', 0)}")
    print(f"\nImages saved to: {images_dir}")
    print(f"CSV saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
