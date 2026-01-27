#!/usr/bin/env python3
"""
Analyze processed Prolific results to show head-to-head statistics.
"""

import sys
import pandas as pd
from collections import Counter


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_prolific_results.py <processed_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    print(f"Total responses: {len(df)}")
    print(f"Total participants: {df['participant_id'].nunique()}\n")

    # Detect mode based on columns
    if 'image1_status' in df.columns:
        mode = 'within'
        variant_col1 = 'image1_status'
        variant_col2 = 'image2_status'
        print(f"Mode: within-strategy (comparing statuses)\n")
    else:
        mode = 'across'
        variant_col1 = 'image1_strategy'
        variant_col2 = 'image2_strategy'
        print(f"Mode: across-strategies (comparing strategies)\n")

    if mode == 'within':
        # Analyze per strategy
        for strategy in sorted(df['strategy'].unique()):
            strategy_df = df[df['strategy'] == strategy]

            print(f"\n{'='*80}")
            print(f"STRATEGY: {strategy.upper()}")
            print(f"{'='*80}")
            print(f"Total comparisons: {len(strategy_df)}\n")

            # Determine winner for each comparison
            winners = []
            for _, row in strategy_df.iterrows():
                if row['choice'] == 1:
                    winner = row[variant_col1]
                else:  # choice == 2
                    winner = row[variant_col2]
                winners.append(winner)

            # Count overall wins per status
            status_counts = Counter(winners)
            total = len(winners)

            print("Overall status wins:")
            print("-" * 40)
            for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
                percentage = (count / total) * 100
                print(f"{status:20s} {count:5d} ({percentage:5.1f}%)")

            # Head-to-head statistics
            print("\nHead-to-head matchups:")
            print("-" * 40)

            pair_counts = Counter()
            for _, row in strategy_df.iterrows():
                status1 = row[variant_col1]
                status2 = row[variant_col2]

                # Create sorted pair for consistency
                pair = tuple(sorted([status1, status2]))

                # Determine winner
                if row['choice'] == 1:
                    winner = status1
                else:
                    winner = status2

                # Record the matchup with winner
                pair_counts[(pair[0], pair[1], winner)] += 1

            # Print head-to-head results
            processed_pairs = set()
            for (status1, status2, winner), count in sorted(pair_counts.items()):
                pair = (status1, status2)
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                # Get counts for both directions
                status1_wins = pair_counts.get((status1, status2, status1), 0)
                status2_wins = pair_counts.get((status1, status2, status2), 0)
                total_pair = status1_wins + status2_wins

                if total_pair > 0:
                    status1_rate = (status1_wins / total_pair) * 100
                    status2_rate = (status2_wins / total_pair) * 100

                    print(f"\n{status1} vs {status2} ({total_pair} comparisons):")
                    print(f"  {status1:20s}: {status1_wins:4d} ({status1_rate:5.1f}%)")
                    print(f"  {status2:20s}: {status2_wins:4d} ({status2_rate:5.1f}%)")

            # Per-task breakdown
            print(f"\n{'-'*80}")
            print("PER-TASK BREAKDOWN")
            print(f"{'-'*80}")

            for task in sorted(strategy_df['task'].unique()):
                task_df = strategy_df[strategy_df['task'] == task]

                print(f"\nTask: {task} ({len(task_df)} comparisons)")

                task_pair_counts = Counter()
                for _, row in task_df.iterrows():
                    status1 = row[variant_col1]
                    status2 = row[variant_col2]
                    pair = tuple(sorted([status1, status2]))

                    if row['choice'] == 1:
                        winner = status1
                    else:
                        winner = status2

                    task_pair_counts[(pair[0], pair[1], winner)] += 1

                # Print task head-to-head
                task_processed = set()
                for (status1, status2, winner), count in sorted(task_pair_counts.items()):
                    pair = (status1, status2)
                    if pair in task_processed:
                        continue
                    task_processed.add(pair)

                    status1_wins = task_pair_counts.get((status1, status2, status1), 0)
                    status2_wins = task_pair_counts.get((status1, status2, status2), 0)
                    total_pair = status1_wins + status2_wins

                    if total_pair > 0:
                        status1_rate = (status1_wins / total_pair) * 100
                        print(f"  {status1} vs {status2}: {status1} {status1_wins}/{total_pair} ({status1_rate:.1f}%)")

    else:  # mode == 'across'
        print(f"\n{'='*80}")
        print("ACROSS-STRATEGIES ANALYSIS")
        print(f"{'='*80}")
        print(f"Total comparisons: {len(df)}\n")

        # Determine winner for each comparison
        winners = []
        for _, row in df.iterrows():
            if row['choice'] == 1:
                winner = row[variant_col1]
            else:  # choice == 2
                winner = row[variant_col2]
            winners.append(winner)

        # Count overall wins per strategy
        strategy_counts = Counter(winners)
        total = len(winners)

        print("Overall strategy wins:")
        print("-" * 40)
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            percentage = (count / total) * 100
            print(f"{strategy:20s} {count:5d} ({percentage:5.1f}%)")

        # Head-to-head statistics
        print("\nHead-to-head matchups:")
        print("-" * 40)

        pair_counts = Counter()
        for _, row in df.iterrows():
            strategy1 = row[variant_col1]
            strategy2 = row[variant_col2]

            # Create sorted pair for consistency
            pair = tuple(sorted([strategy1, strategy2]))

            # Determine winner
            if row['choice'] == 1:
                winner = strategy1
            else:
                winner = strategy2

            # Record the matchup with winner
            pair_counts[(pair[0], pair[1], winner)] += 1

        # Print head-to-head results
        processed_pairs = set()
        for (strategy1, strategy2, winner), count in sorted(pair_counts.items()):
            pair = (strategy1, strategy2)
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            # Get counts for both directions
            strategy1_wins = pair_counts.get((strategy1, strategy2, strategy1), 0)
            strategy2_wins = pair_counts.get((strategy1, strategy2, strategy2), 0)
            total_pair = strategy1_wins + strategy2_wins

            if total_pair > 0:
                strategy1_rate = (strategy1_wins / total_pair) * 100
                strategy2_rate = (strategy2_wins / total_pair) * 100

                print(f"\n{strategy1} vs {strategy2} ({total_pair} comparisons):")
                print(f"  {strategy1:20s}: {strategy1_wins:4d} ({strategy1_rate:5.1f}%)")
                print(f"  {strategy2:20s}: {strategy2_wins:4d} ({strategy2_rate:5.1f}%)")

        # Per-task breakdown
        print(f"\n{'-'*80}")
        print("PER-TASK BREAKDOWN")
        print(f"{'-'*80}")

        for task in sorted(df['task'].unique()):
            task_df = df[df['task'] == task]

            print(f"\nTask: {task} ({len(task_df)} comparisons)")

            task_pair_counts = Counter()
            for _, row in task_df.iterrows():
                strategy1 = row[variant_col1]
                strategy2 = row[variant_col2]
                pair = tuple(sorted([strategy1, strategy2]))

                if row['choice'] == 1:
                    winner = strategy1
                else:
                    winner = strategy2

                task_pair_counts[(pair[0], pair[1], winner)] += 1

            # Print task head-to-head
            task_processed = set()
            for (strategy1, strategy2, winner), count in sorted(task_pair_counts.items()):
                pair = (strategy1, strategy2)
                if pair in task_processed:
                    continue
                task_processed.add(pair)

                strategy1_wins = task_pair_counts.get((strategy1, strategy2, strategy1), 0)
                strategy2_wins = task_pair_counts.get((strategy1, strategy2, strategy2), 0)
                total_pair = strategy1_wins + strategy2_wins

                if total_pair > 0:
                    strategy1_rate = (strategy1_wins / total_pair) * 100
                    print(f"  {strategy1} vs {strategy2}: {strategy1} {strategy1_wins}/{total_pair} ({strategy1_rate:.1f}%)")


if __name__ == "__main__":
    main()
