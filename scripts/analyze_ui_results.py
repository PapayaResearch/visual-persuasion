#!/usr/bin/env python3
"""
Summarize UI evaluation results by reporting how often each variant type wins.
"""

import argparse
import os
import sys
import pandas as pd
from collections import Counter


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"variant_a", "variant_b", "choice"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def extract_variant_type(label: str) -> str:
    """
    Parse labels like 'ui-1 (final) [filename]' and return the trailing bucket (final/baseline).
    """
    if "(final)" in label:
        return "final"
    if "(baseline)" in label:
        return "baseline"
    return "unknown"


def summarize_wins(df: pd.DataFrame):
    filtered = df[df["choice"] != "inconsistent"].copy()
    if filtered.empty:
        print("No consistent UI decisions to summarize.")
        return

    win_types = []
    for _, row in filtered.iterrows():
        choice = row["choice"]
        if choice == row["variant_a"]:
            win_types.append(extract_variant_type(row["variant_a"]))
        elif choice == row["variant_b"]:
            win_types.append(extract_variant_type(row["variant_b"]))
        else:
            win_types.append("unknown")

    counts = Counter(win_types)
    total = sum(counts.values())

    print(f"Total consistent comparisons: {total}\n")
    print("UI variant wins:")
    print("-" * 30)
    for variant_type, count in sorted(counts.items()):
        pct = (count / total) * 100 if total else 0
        print(f"{variant_type:10s} {count:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Summarize UI evaluation wins by variant type.")
    parser.add_argument("csv_path", help="Path to results_ui.csv")
    args = parser.parse_args()

    try:
        df = load_results(args.csv_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    summarize_wins(df)


if __name__ == "__main__":
    main()
