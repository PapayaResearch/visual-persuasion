#!/usr/bin/env python3
"""
Analyze evaluation results to see which edit types are chosen most often.
"""

import sys
import pandas as pd
from collections import Counter

def extract_edit_type(base_name):
    """Extract edit type from base name like 'ID_original' or 'ID_prior-5'."""
    if base_name == "inconsistent":
        return "inconsistent"
    return base_name.split('_', 1)[1]

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_choices.py <path_to_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    # Filter out comparisons where both are "original"
    def is_original_vs_original(row):
        edit_type_1 = extract_edit_type(row['base1'])
        edit_type_2 = extract_edit_type(row['base2'])
        return edit_type_1 == 'original' and edit_type_2 == 'original'

    df = df[~df.apply(is_original_vs_original, axis=1)]

    # Extract edit type of chosen image
    chosen_edit_types = []
    for _, row in df.iterrows():
        chosen_base = row['choice']
        edit_type = extract_edit_type(chosen_base)
        if edit_type == "inconsistent":
            continue  # Skip inconsistent choices
        chosen_edit_types.append(edit_type)

    # Count choices
    counts = Counter(chosen_edit_types)
    total = len(chosen_edit_types)

    # Report results
    print(f"\nTotal comparisons: {total}\n")
    print("Edit type choices:")
    print("-" * 40)
    for edit_type, count in sorted(counts.items()):
        percentage = (count / total) * 100
        print(f"{edit_type:20s} {count:5d} ({percentage:5.1f}%)")


    # Print win-counts and win-rates for each edit_type pair
    pair_counts = Counter()
    for _, row in df.iterrows():
        edit_type_1 = extract_edit_type(row['base1'])
        edit_type_2 = extract_edit_type(row['base2'])
        pair = sorted([edit_type_1, edit_type_2])
        if row['choice'] == 'inconsistent':
            continue  # Skip inconsistent choices
        winner = row['choice'].split('_', 1)[1]  # Get the edit type of the chosen image
        if winner == row['base1']:
            pair_counts[(pair[0], pair[1], pair[0])] += 1  # edit_type_1 wins
        elif winner == row['base2']:
            pair_counts[(pair[0], pair[1], pair[1])] += 1  # edit_type_2 wins
        pair_win = (pair[0], pair[1], winner)
        pair_counts[pair_win] += 1

    print("\nPairwise win counts:")
    print("-" * 40)
    for (edit_type_1, edit_type_2, winner), count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        total_pair = pair_counts[(edit_type_1, edit_type_2, edit_type_1)] + pair_counts[(edit_type_1, edit_type_2, edit_type_2)]
        if total_pair > 0:
            win_rate = (count / total_pair) * 100
            if winner == edit_type_1:
                print(f"{edit_type_1:20s} beats {edit_type_2:20s} {count:5d} ({win_rate:5.1f}%)")
            else:
                print(f"{edit_type_2:20s} beats {edit_type_1:20s} {count:5d} ({win_rate:5.1f}%)")


    # Check win counts/rates for each base
    pairs = {}
    for _, row in df.iterrows():
        if row['choice'] == 'inconsistent':
            continue  # Skip inconsistent choices
        pairs.setdefault(row['base1'].split("_", 1)[0] + ' vs ' + row['base2'].split("_", 1)[0], []).append(row['choice'].split("_", 1)[0])

    # Print win counts and rates for each base product within each pair
    print("\nWin counts and rates for each base product within each pair:")
    print("-" * 40)
    for pair, choices in pairs.items():
        count_base1 = sum(1 for choice in choices if choice == pair.split(' vs ')[0])
        count_base2 = sum(1 for choice in choices if choice == pair.split(' vs ')[1])
        total = count_base1 + count_base2
        if total > 0:
            win_rate_base1 = (count_base1 / total) * 100
            win_rate_base2 = (count_base2 / total) * 100
            print(f"{pair:40s} {count_base1:5d} ({win_rate_base1:5.1f}%) vs {count_base2:5d} ({win_rate_base2:5.1f}%)")
            print("-" * 40)

if __name__ == "__main__":
    main()
