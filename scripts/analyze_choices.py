#!/usr/bin/env python3
"""
Analyze evaluation results to see which edit types are chosen most often.
"""

import sys
import pandas as pd
from collections import Counter

def extract_edit_type(base_name):
    """Extract edit type from base name like 'ID_original' or 'ID_prior-5'."""
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
        chosen_edit_types.append(edit_type)

    # Count choices
    counts = Counter(chosen_edit_types)
    total = len(chosen_edit_types)

    # Count position bias
    chose_first = sum(df['choice'] == df['first'])
    chose_second = sum(df['choice'] == df['second'])

    # Report results
    print(f"\nTotal comparisons: {total}\n")
    print("Edit type choices:")
    print("-" * 40)
    for edit_type, count in sorted(counts.items()):
        percentage = (count / total) * 100
        print(f"{edit_type:20s} {count:5d} ({percentage:5.1f}%)")

    print(f"\n\nPosition bias:")
    print("-" * 40)
    print(f"Chose first image:  {chose_first:5d} ({(chose_first/total)*100:5.1f}%)")
    print(f"Chose second image: {chose_second:5d} ({(chose_second/total)*100:5.1f}%)")

if __name__ == "__main__":
    main()
