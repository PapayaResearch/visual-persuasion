"""Analyzes priors results against product attributes to understand model biases."""

import argparse
import pandas as pd
from pathlib import Path


def load_and_merge_data(priors_csv: Path, attributes_csv: Path) -> pd.DataFrame:
    """Load priors results and product attributes, merge them together."""
    # Load the CSVs
    priors = pd.read_csv(priors_csv)
    attributes = pd.read_csv(attributes_csv)

    # Create a lookup dictionary for attributes by filename (without extension)
    # Handle both with and without .jpg extension
    attr_dict = {}
    for _, row in attributes.iterrows():
        filename = row['filename']
        # Store with and without extension
        base = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        attr_dict[base] = row.to_dict()
        attr_dict[filename] = row.to_dict()

    # Merge attributes for both images in each comparison
    merged_data = []
    for _, comparison in priors.iterrows():
        img1_id = comparison['image_id1']
        img2_id = comparison['image_id2']

        if img1_id not in attr_dict or img2_id not in attr_dict:
            continue

        img1_attrs = attr_dict[img1_id]
        img2_attrs = attr_dict[img2_id]

        merged_data.append({
            'category': comparison['category'],
            'image_id1': img1_id,
            'image_id2': img2_id,
            'winner': comparison['winner'],
            'winner_score': comparison['winner_score'],
            'consistent_judges': comparison['consistent_judges'],
            'price1': img1_attrs['price'],
            'price2': img2_attrs['price'],
            'rating1': img1_attrs['rating'],
            'rating2': img2_attrs['rating'],
            'usability1': img1_attrs['usability'],
            'usability2': img2_attrs['usability'],
            'appealing1': img1_attrs['appealing'],
            'appealing2': img2_attrs['appealing'],
            'quality1': img1_attrs['quality'],
            'quality2': img2_attrs['quality'],
        })

    return pd.DataFrame(merged_data)


def analyze_attribute_preferences(df: pd.DataFrame) -> dict:
    """Analyze how often the model prefers products with higher values of each attribute."""
    attributes = ['price', 'rating', 'usability', 'appealing', 'quality']
    results = {}

    for attr in attributes:
        attr1 = f"{attr}1"
        attr2 = f"{attr}2"

        # Count how often winner had higher attribute value
        winner_had_higher = 0
        winner_had_lower = 0
        winner_had_equal = 0
        total_comparisons = 0

        differences_when_winner = []
        differences_when_loser = []

        for _, row in df.iterrows():
            winner_is_1 = (row['winner'] == row['image_id1'])

            val1 = row[attr1]
            val2 = row[attr2]

            if pd.isna(val1) or pd.isna(val2):
                continue

            total_comparisons += 1

            # Determine winner's attribute value
            winner_val = val1 if winner_is_1 else val2
            loser_val = val2 if winner_is_1 else val1

            diff = winner_val - loser_val

            if winner_val > loser_val:
                winner_had_higher += 1
                differences_when_winner.append(diff)
            elif winner_val < loser_val:
                winner_had_lower += 1
                differences_when_loser.append(diff)
            else:
                winner_had_equal += 1

        if total_comparisons > 0:
            results[attr] = {
                'total': total_comparisons,
                'winner_higher': winner_had_higher,
                'winner_lower': winner_had_lower,
                'winner_equal': winner_had_equal,
                'prop_winner_higher': winner_had_higher / total_comparisons,
                'prop_winner_lower': winner_had_lower / total_comparisons,
                'prop_winner_equal': winner_had_equal / total_comparisons,
                'avg_diff_when_winner_higher': sum(differences_when_winner) / len(differences_when_winner) if differences_when_winner else 0,
                'avg_diff_when_winner_lower': sum(differences_when_loser) / len(differences_when_loser) if differences_when_loser else 0,
            }

    return results


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute correlations between winning and attribute differences."""
    attributes = ['price', 'rating', 'usability', 'appealing', 'quality']
    correlations = {}

    # Create binary winner column (1 if image1 won, 0 if image2 won)
    df['winner_is_1'] = (df['winner'] == df['image_id1']).astype(int)

    for attr in attributes:
        # Compute difference (image1 - image2)
        diff_col = f"{attr}_diff"
        df[diff_col] = df[f"{attr}1"] - df[f"{attr}2"]

        # Correlation between winning (being image1) and having higher attribute value
        if diff_col in df.columns and not df[diff_col].isna().all():
            corr = df['winner_is_1'].corr(df[diff_col])
            correlations[attr] = corr

    return correlations


def print_summary(results: dict, correlations: dict, total_comparisons: int, total_categories: int):
    """Print a nicely formatted summary of the analysis."""
    print("=" * 80)
    print("PRIORS ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal comparisons analyzed: {total_comparisons}")
    print(f"Total categories: {total_categories}")
    print("\n" + "=" * 80)
    print("ATTRIBUTE PREFERENCE ANALYSIS")
    print("=" * 80)
    print("\nHow often does the model prefer products with HIGHER values of each attribute?")
    print("-" * 80)

    for attr in ['price', 'rating', 'usability', 'appealing', 'quality']:
        if attr not in results:
            continue

        r = results[attr]
        print(f"\n{attr.upper()}:")
        print(f"  Winner had HIGHER {attr}: {r['winner_higher']:>4} ({r['prop_winner_higher']:>6.1%})")
        print(f"  Winner had LOWER {attr}:  {r['winner_lower']:>4} ({r['prop_winner_lower']:>6.1%})")
        print(f"  Winner had EQUAL {attr}:  {r['winner_equal']:>4} ({r['prop_winner_equal']:>6.1%})")

        if r['avg_diff_when_winner_higher'] != 0:
            print(f"  Avg difference when winner higher: {r['avg_diff_when_winner_higher']:>+.2f}")
        if r['avg_diff_when_winner_lower'] != 0:
            print(f"  Avg difference when winner lower:  {r['avg_diff_when_winner_lower']:>+.2f}")

    print("\n" + "=" * 80)
    print("CORRELATIONS")
    print("=" * 80)
    print("\nCorrelation between winning and having higher attribute value:")
    print("(Positive = prefers higher, Negative = prefers lower)")
    print("-" * 80)

    for attr in ['price', 'rating', 'usability', 'appealing', 'quality']:
        if attr in correlations and not pd.isna(correlations[attr]):
            corr = correlations[attr]
            direction = "↑" if corr > 0 else "↓"
            print(f"  {attr.capitalize():<12}: {corr:>+.3f} {direction}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze priors results against product attributes"
    )
    parser.add_argument(
        "--priors",
        "-p",
        required=True,
        help="Path to priors_results.csv from run_priors.py"
    )
    parser.add_argument(
        "--attributes",
        "-a",
        required=True,
        help="Path to product_analysis.csv from analyze_product_images.py"
    )
    args = parser.parse_args()

    # Load and merge data
    print("Loading data...")
    df = load_and_merge_data(Path(args.priors), Path(args.attributes))

    if len(df) == 0:
        print("ERROR: No matching data found between priors and attributes files.")
        print("Check that image filenames match between the two CSVs.")
        return

    # Analyze
    print(f"Analyzing {len(df)} comparisons...")
    results = analyze_attribute_preferences(df)
    correlations = compute_correlations(df)

    # Print summary
    total_categories = df['category'].nunique()
    print_summary(results, correlations, len(df), total_categories)


if __name__ == "__main__":
    main()
