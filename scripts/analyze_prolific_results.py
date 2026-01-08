#!/usr/bin/env python3
"""
Analyze Prolific survey results to determine how often participants
chose zero-shot vs final images.
"""

import csv
import re
from pathlib import Path
from collections import defaultdict


def parse_image_status(image_url):
    """
    Extract status (zero-shot or final) from image URL.
    Expected format: .../CATEGORY_ID_STATUS.jpg
    """
    match = re.search(r'_([a-f0-9]+)_(zero-shot|final)\.jpg$', image_url)
    if match:
        return match.group(2)
    return None


def load_survey_pairs(csv_path):
    """
    Load survey pairs from CSV.
    Returns: dict mapping row index to (category, image1_status, image2_status)
    """
    pairs = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            category = row['category']
            img1_status = parse_image_status(row['image1_path'])
            img2_status = parse_image_status(row['image2_path'])
            pairs[idx] = (category, img1_status, img2_status)
    return pairs


def analyze_results(results_path, set1_pairs_path, set2_pairs_path):
    """
    Analyze survey results and count zero-shot vs final selections.
    """
    # Load both sets of pairs
    set1_pairs = load_survey_pairs(set1_pairs_path)
    set2_pairs = load_survey_pairs(set2_pairs_path)

    # Counters
    total_zero_shot = 0
    total_final = 0
    by_category = defaultdict(lambda: {'zero-shot': 0, 'final': 0})

    # Read results - need to handle duplicate column names manually
    with open(results_path, 'r') as f:
        lines = list(csv.reader(f))

        # Get header row (first row)
        header = lines[0]

        # Skip the second header row (descriptions) and third row (import IDs)
        # Data starts at row 4 (index 3)
        data_rows = lines[3:]

        # Find column indices for each question set
        # Set 1: 2_Q7 through 18_Q7 (first occurrence)
        # Set 2: 1_Q7 through 17_Q7 (appears after 18_Q7)

        set1_cols = {}  # Maps question number to column index
        set2_cols = {}  # Maps question number to column index

        # Find first occurrence of each numbered Q7 column for set 1
        for i, col_name in enumerate(header):
            if col_name.endswith('_Q7'):
                q_num = col_name.split('_')[0]
                if q_num.isdigit():
                    q_num = int(q_num)
                    if q_num >= 2 and q_num <= 18 and q_num not in set1_cols:
                        set1_cols[q_num] = i

        # Find second occurrence of numbered Q7 columns for set 2
        # These start after column index with 18_Q7
        max_set1_idx = max(set1_cols.values()) if set1_cols else 0
        for i, col_name in enumerate(header):
            if i > max_set1_idx and col_name.endswith('_Q7'):
                q_num = col_name.split('_')[0]
                if q_num.isdigit():
                    q_num = int(q_num)
                    if q_num >= 1 and q_num <= 17:
                        set2_cols[q_num] = i

        # Process each participant
        for row in data_rows:
            if len(row) < len(header):
                continue

            # Check which set this participant answered
            # Set 1: has answers in 2_Q7 through 18_Q7 columns
            # Set 2: has answers in the second group of 1_Q7 through 17_Q7 columns

            has_set1 = any(row[set1_cols[q]].strip() for q in set1_cols if row[set1_cols[q]].strip() in ['Top', 'Bottom'])
            has_set2 = any(row[set2_cols[q]].strip() for q in set2_cols if row[set2_cols[q]].strip() in ['Top', 'Bottom'])

            if has_set1:
                # Process set 1 responses
                pairs = set1_pairs
                for q_num in range(2, 19):  # 2_Q7 through 18_Q7
                    if q_num not in set1_cols:
                        continue

                    col_idx = set1_cols[q_num]
                    answer = row[col_idx].strip() if col_idx < len(row) else ''

                    pair_idx = q_num - 1  # Pair index starts at 1 for question 2_Q7
                    if answer and pair_idx in pairs:
                        category, img1_status, img2_status = pairs[pair_idx]

                        # Determine which image was selected
                        if answer == 'Top':
                            selected_status = img1_status
                        elif answer == 'Bottom':
                            selected_status = img2_status
                        else:
                            continue

                        # Count the selection
                        if selected_status == 'zero-shot':
                            total_zero_shot += 1
                            by_category[category]['zero-shot'] += 1
                        elif selected_status == 'final':
                            total_final += 1
                            by_category[category]['final'] += 1

            if has_set2:
                # Process set 2 responses
                pairs = set2_pairs
                for q_num in range(1, 18):  # 1_Q7 through 17_Q7
                    if q_num not in set2_cols:
                        continue

                    col_idx = set2_cols[q_num]
                    answer = row[col_idx].strip() if col_idx < len(row) else ''

                    if answer and q_num in pairs:
                        category, img1_status, img2_status = pairs[q_num]

                        # Determine which image was selected
                        if answer == 'Top':
                            selected_status = img1_status
                        elif answer == 'Bottom':
                            selected_status = img2_status
                        else:
                            continue

                        # Count the selection
                        if selected_status == 'zero-shot':
                            total_zero_shot += 1
                            by_category[category]['zero-shot'] += 1
                        elif selected_status == 'final':
                            total_final += 1
                            by_category[category]['final'] += 1

    return total_zero_shot, total_final, dict(by_category)


def main():
    script_dir = Path(__file__).parent
    survey_dir = script_dir / 'prolific_survey'

    results_path = survey_dir / 'results.csv'
    set1_path = survey_dir / 'survey_pairs_set1.csv'
    set2_path = survey_dir / 'survey_pairs_set2.csv'

    # Check files exist
    for path in [results_path, set1_path, set2_path]:
        if not path.exists():
            print(f"Error: {path} not found")
            return

    # Analyze results
    total_zero_shot, total_final, by_category = analyze_results(
        results_path, set1_path, set2_path
    )

    # Print summary
    print("=" * 60)
    print("PROLIFIC SURVEY RESULTS SUMMARY")
    print("=" * 60)
    print()

    total = total_zero_shot + total_final
    if total > 0:
        print(f"Total responses: {total}")
        print(f"Zero-shot selected: {total_zero_shot} ({100*total_zero_shot/total:.1f}%)")
        print(f"Final selected: {total_final} ({100*total_final/total:.1f}%)")
        print()

        print("=" * 60)
        print("BY CATEGORY")
        print("=" * 60)
        print()

        # Sort categories alphabetically
        for category in sorted(by_category.keys()):
            counts = by_category[category]
            cat_total = counts['zero-shot'] + counts['final']
            if cat_total > 0:
                zero_pct = 100 * counts['zero-shot'] / cat_total
                final_pct = 100 * counts['final'] / cat_total
                print(f"{category:20s}: zero-shot={counts['zero-shot']:2d} ({zero_pct:4.1f}%)  "
                      f"final={counts['final']:2d} ({final_pct:4.1f}%)")
    else:
        print("No valid responses found.")


if __name__ == '__main__':
    main()
