"""Analyzes correlations between product attributes from LLM assessments."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def print_correlation_matrix(df: pd.DataFrame):
    """Print a nicely formatted correlation matrix."""
    attributes = ['price', 'rating', 'usability', 'appealing', 'quality']

    # Filter to only include attributes that exist in the dataframe
    available_attrs = [attr for attr in attributes if attr in df.columns]

    if not available_attrs:
        print("ERROR: No attribute columns found in CSV")
        return

    # Compute correlation matrix
    corr_matrix = df[available_attrs].corr()

    print("=" * 80)
    print("PRODUCT ATTRIBUTE CORRELATION MATRIX")
    print("=" * 80)
    print(f"\nBased on {len(df)} products\n")

    # Print header
    print("              ", end="")
    for attr in available_attrs:
        print(f"{attr[:10]:>12}", end="")
    print()
    print("-" * 80)

    # Print rows
    for i, attr1 in enumerate(available_attrs):
        print(f"{attr1:<12}  ", end="")
        for j, attr2 in enumerate(available_attrs):
            corr_val = corr_matrix.loc[attr1, attr2]
            if i == j:
                # Diagonal (correlation with self = 1.0)
                print(f"     1.000   ", end="")
            else:
                # Format with color indicators
                if corr_val > 0.7:
                    indicator = "+++"
                elif corr_val > 0.4:
                    indicator = "++"
                elif corr_val > 0.2:
                    indicator = "+"
                elif corr_val < -0.7:
                    indicator = "---"
                elif corr_val < -0.4:
                    indicator = "--"
                elif corr_val < -0.2:
                    indicator = "-"
                else:
                    indicator = "   "

                print(f"{corr_val:>+6.3f} {indicator}", end="")
        print()

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("+++  : Strong positive correlation (> 0.7)")
    print("++   : Moderate positive correlation (> 0.4)")
    print("+    : Weak positive correlation (> 0.2)")
    print("     : No significant correlation")
    print("-    : Weak negative correlation (< -0.2)")
    print("--   : Moderate negative correlation (< -0.4)")
    print("---  : Strong negative correlation (< -0.7)")
    print("\n" + "=" * 80)

    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 80)
    print(f"{'Attribute':<12}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 80)
    for attr in available_attrs:
        stats = df[attr].describe()
        print(f"{attr:<12}  {stats['mean']:>8.2f}  {stats['std']:>8.2f}  "
              f"{stats['min']:>8.2f}  {stats['max']:>8.2f}")
    print("=" * 80)


def print_strongest_correlations(df: pd.DataFrame, top_n: int = 5):
    """Print the strongest positive and negative correlations."""
    attributes = ['price', 'rating', 'usability', 'appealing', 'quality']
    available_attrs = [attr for attr in attributes if attr in df.columns]

    if len(available_attrs) < 2:
        return

    # Compute correlation matrix
    corr_matrix = df[available_attrs].corr()

    # Get all unique pairs (excluding diagonal)
    correlations = []
    for i, attr1 in enumerate(available_attrs):
        for j, attr2 in enumerate(available_attrs):
            if i < j:  # Only upper triangle to avoid duplicates
                correlations.append({
                    'attr1': attr1,
                    'attr2': attr2,
                    'correlation': corr_matrix.loc[attr1, attr2]
                })

    correlations_df = pd.DataFrame(correlations)

    print("\nSTRONGEST CORRELATIONS")
    print("=" * 80)

    # Sort by absolute correlation value
    sorted_corr = correlations_df.sort_values('correlation', ascending=False, key=abs)

    print("\nTop positive correlations:")
    print("-" * 80)
    positive = sorted_corr[sorted_corr['correlation'] > 0].head(top_n)
    for _, row in positive.iterrows():
        print(f"  {row['attr1']:<12} ↔ {row['attr2']:<12} : {row['correlation']:>+.3f}")

    print("\nTop negative correlations:")
    print("-" * 80)
    negative = sorted_corr[sorted_corr['correlation'] < 0].tail(top_n)
    for _, row in negative.iterrows():
        print(f"  {row['attr1']:<12} ↔ {row['attr2']:<12} : {row['correlation']:>+.3f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations between product attributes"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to product_analysis.csv from collect_product_priors.py"
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=5,
        help="Number of top correlations to display (default: 5)"
    )
    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    if len(df) == 0:
        print("ERROR: CSV file is empty")
        return

    print(f"Loaded {len(df)} products from {input_path}\n")

    # Print correlation matrix
    print_correlation_matrix(df)

    # Print strongest correlations
    print_strongest_correlations(df, args.top)


if __name__ == "__main__":
    main()
