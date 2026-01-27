#!/usr/bin/env python3
"""
Analyze methods comparison results from evaluation directory
Usage: python scripts/analyze_methods.py <path_to_evaluation_dir>
"""

import os
import sys
import pandas as pd


def analyze_csv(csv_path, model_name):
    """Analyze a single results_methods.csv file"""
    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Total comparisons: {len(df)}\n")

    # Count wins per strategy
    wins = {}
    for _, row in df.iterrows():
        if row['choice'] == 'first':
            winner = row['strategy1']
        elif row['choice'] == 'second':
            winner = row['strategy2']
        else:  # inconsistent
            continue

        wins[winner] = wins.get(winner, 0) + 1

    # Calculate totals
    total_valid = sum(wins.values())
    inconsistent = (df['choice'] == 'inconsistent').sum()

    # Print win percentages
    print("Win percentages (sorted by wins):")
    for strategy, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_valid) * 100
        print(f"  {strategy:20s}: {count:4d}/{total_valid} ({percentage:5.1f}%)")

    print(f"\nInconsistent judgments: {inconsistent}/{len(df)} ({inconsistent/len(df)*100:.1f}%)")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_methods.py <path_to_evaluation_dir>")
        sys.exit(1)

    eval_dir = sys.argv[1]

    if not os.path.isdir(eval_dir):
        print(f"Error: {eval_dir} is not a directory")
        sys.exit(1)

    # Find all results_methods.csv files recursively
    found_files = []
    for root, dirs, files in os.walk(eval_dir):
        if 'results_methods.csv' in files:
            csv_path = os.path.join(root, 'results_methods.csv')
            # Get model name from relative path
            rel_path = os.path.relpath(root, eval_dir)
            model_name = rel_path.replace(os.sep, '/')
            found_files.append((csv_path, model_name))

    if not found_files:
        print(f"No results_methods.csv files found in {eval_dir}")
        sys.exit(1)

    # Sort by model name and analyze each
    for csv_path, model_name in sorted(found_files, key=lambda x: x[1]):
        analyze_csv(csv_path, model_name)


if __name__ == "__main__":
    main()
