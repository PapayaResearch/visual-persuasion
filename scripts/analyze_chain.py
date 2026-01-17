#!/usr/bin/env python3
"""
Analyze chain evaluation results to see progression patterns.
"""

import sys
import pandas as pd
from collections import Counter

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_chain.py <path_to_chain_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    total = len(df)
    consistent = df[df['choice'] != 'inconsistent']
    inconsistent_count = total - len(consistent)

    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total comparisons: {total}")
    print(f"Consistent: {len(consistent)} ({len(consistent)/total*100:.1f}%)")
    print(f"Inconsistent: {inconsistent_count} ({inconsistent_count/total*100:.1f}%)")

    # Analyze by comparison type
    print(f"\n{'='*80}")
    print(f"BY COMPARISON TYPE")
    print(f"{'='*80}")

    for _, group in consistent.groupby(['first', 'second']):
        first_status = group['first'].iloc[0]
        second_status = group['second'].iloc[0]
        comparison_type = f"{first_status} vs {second_status}"

        first_wins = len(group[group['choice'] == 'first'])
        second_wins = len(group[group['choice'] == 'second'])
        total_comp = len(group)

        print(f"\n{comparison_type}:")
        print(f"  {first_status:15s} wins: {first_wins:4d} ({first_wins/total_comp*100:.1f}%)")
        print(f"  {second_status:15s} wins: {second_wins:4d} ({second_wins/total_comp*100:.1f}%)")

    # Per-category analysis
    print(f"\n{'='*80}")
    print(f"PER-CATEGORY SUMMARIES")
    print(f"{'='*80}")

    for category in sorted(df['image_class'].unique()):
        category_df = df[df['image_class'] == category]
        category_consistent = category_df[category_df['choice'] != 'inconsistent']

        if len(category_consistent) == 0:
            continue

        print(f"\nCategory: {category}")
        print(f"  Total: {len(category_df)}, Consistent: {len(category_consistent)}")
        print("-" * 40)

        for _, group in category_consistent.groupby(['first', 'second']):
            first_status = group['first'].iloc[0]
            second_status = group['second'].iloc[0]

            first_wins = len(group[group['choice'] == 'first'])
            second_wins = len(group[group['choice'] == 'second'])
            total_comp = len(group)

            if total_comp > 0:
                print(f"  {first_status:15s} vs {second_status:15s}: " +
                      f"{first_status} {first_wins}/{total_comp} ({first_wins/total_comp*100:.1f}%), " +
                      f"{second_status} {second_wins}/{total_comp} ({second_wins/total_comp*100:.1f}%)")

if __name__ == "__main__":
    main()
