"""Analyzes correlations between product attributes from LLM assessments."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def print_correlation_matrix(df: pd.DataFrame):
    """Print a nicely formatted correlation matrix."""
    attributes = ['price', 'rating', 'usability', 'appealing', 'quality']
    corr_matrix = df[attributes].corr()

    print("=" * 80)
    print("PRODUCT ATTRIBUTE CORRELATION MATRIX")
    print("=" * 80)
    print(f"\nBased on {len(df)} products\n")

    print("              ", end="")
    for attr in attributes:
        print(f"{attr[:10]:>12}", end="")
    print()
    print("-" * 80)

    for i, attr1 in enumerate(attributes):
        print(f"{attr1:<12}  ", end="")
        for j, attr2 in enumerate(attributes):
            corr_val = corr_matrix.loc[attr1, attr2]
            if i == j:
                print(f"     1.000   ", end="")
            else:
                print(f"{corr_val:>+6.3f}     ", end="")
        print()

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"{'Attribute':<12}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 80)
    for attr in attributes:
        stats = df[attr].describe()
        print(f"{attr:<12}  {stats['mean']:>8.2f}  {stats['std']:>8.2f}  "
              f"{stats['min']:>8.2f}  {stats['max']:>8.2f}")
    print("=" * 80)

    return corr_matrix


def plot_correlation_matrix(corr_matrix: pd.DataFrame, output_path: Path):
    """Save correlation matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    attributes = corr_matrix.columns.tolist()
    ax.set_xticks(np.arange(len(attributes)))
    ax.set_yticks(np.arange(len(attributes)))
    ax.set_xticklabels(attributes)
    ax.set_yticklabels(attributes)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(attributes)):
        for j in range(len(attributes)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)

    ax.set_title('Product Attribute Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCorrelation matrix plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to product_analysis.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    corr_matrix = print_correlation_matrix(df)

    csv_path = Path(args.csv)
    output_path = csv_path.parent / "correlation_matrix.png"
    plot_correlation_matrix(corr_matrix, output_path)


if __name__ == "__main__":
    main()
