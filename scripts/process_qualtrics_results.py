#!/usr/bin/env python3
"""
Process Qualtrics survey results and match with comparison metadata.
"""

import argparse
import csv
import pandas as pd
from pathlib import Path


def load_comparisons(comparisons_csv):
    """
    Load comparisons CSV and index by row number (1-based).
    Returns: {row_number: comparison_dict}, mode ('within' or 'across')
    """
    comparisons = {}
    mode = None
    with open(comparisons_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Detect mode based on columns
        if 'strategy' in fieldnames:
            mode = 'within'
        else:
            mode = 'across'

        for idx, row in enumerate(reader, start=1):
            comparisons[idx] = row

    return comparisons, mode


def process_results(results_csv, comparisons, mode, skip_rows=[1, 2]):
    """
    Process Qualtrics results CSV and match answers to comparisons.

    Args:
        results_csv: Path to Qualtrics results CSV
        comparisons: Dict of comparison metadata indexed by row number
        skip_rows: List of row indices to skip (default: [1,2] for Qualtrics 2 metadata rows)

    Returns:
        List of dicts with processed results
    """
    # Read the results CSV, skipping metadata rows but keeping first row as header
    df = pd.read_csv(results_csv, skiprows=skip_rows)

    # Find all question columns (pattern: N_Q7)
    question_cols = [col for col in df.columns if '_Q7' in col and col.split('_')[0].isdigit()]

    # Sort question columns numerically (not lexicographically)
    question_cols = sorted(question_cols, key=lambda x: int(x.split('_')[0]))

    print(f"Found {len(question_cols)} question columns")
    print(f"Found {len(df)} participants")

    processed = []

    # Process by comparison number (to match comparisons CSV order)
    for question_col in question_cols:
        comparison_num = int(question_col.split('_')[0])

        # Get comparison metadata
        if comparison_num not in comparisons:
            print(f"Warning: No comparison found for number {comparison_num}")
            continue

        comparison = comparisons[comparison_num]

        # Process all participants' answers to this comparison
        for idx, row in df.iterrows():
            participant_id = row.get('ResponseId', f'participant_{idx}')
            answer = row[question_col]

            # Skip if no answer
            if pd.isna(answer) or answer == '':
                continue

            # Convert answer to choice number (Top=1, Bottom=2)
            if answer == 'Top':
                choice = 1
            elif answer == 'Bottom':
                choice = 2
            else:
                # Skip invalid answers
                continue

            # Create result row based on mode
            if mode == 'within':
                result = {
                    'participant_id': participant_id,
                    'choice': choice,
                    'strategy': comparison['strategy'],
                    'task': comparison['task'],
                    'category': comparison['category'],
                    'image1_id': comparison['image1_id'],
                    'image1_status': comparison['image1_status'],
                    'image2_id': comparison['image2_id'],
                    'image2_status': comparison['image2_status']
                }
            else:  # mode == 'across'
                result = {
                    'participant_id': participant_id,
                    'choice': choice,
                    'task': comparison['task'],
                    'category': comparison['category'],
                    'image1_id': comparison['image1_id'],
                    'image1_strategy': comparison['image1_strategy'],
                    'image2_id': comparison['image2_id'],
                    'image2_strategy': comparison['image2_strategy']
                }

            processed.append(result)

    return processed


def main():
    parser = argparse.ArgumentParser(
        description='Process Qualtrics survey results and match with comparison metadata'
    )
    parser.add_argument(
        '--comparisons',
        required=True,
        help='Path to comparisons CSV'
    )
    parser.add_argument(
        '--results',
        required=True,
        help='Path to Qualtrics results CSV'
    )
    parser.add_argument(
        '--output',
        default='processed_results.csv',
        help='Output CSV file (default: processed_results.csv)'
    )
    parser.add_argument(
        '--skip-rows',
        type=str,
        default='1,2',
        help='Comma-separated list of row indices to skip (default: 1,2)'
    )

    args = parser.parse_args()

    print(f"Loading comparisons from: {args.comparisons}")
    comparisons, mode = load_comparisons(args.comparisons)
    print(f"Loaded {len(comparisons)} comparisons")
    print(f"Mode: {mode} ({'within-strategy' if mode == 'within' else 'across-strategies'})")

    # Parse skip_rows argument
    skip_rows = [int(x) for x in args.skip_rows.split(',')]
    print(f"Skipping rows: {skip_rows}")

    print(f"\nProcessing results from: {args.results}")
    processed = process_results(args.results, comparisons, mode, skip_rows)
    print(f"Processed {len(processed)} total responses")

    # Write to CSV
    if processed:
        if mode == 'within':
            fieldnames = [
                'participant_id', 'choice',
                'strategy', 'task', 'category',
                'image1_id', 'image1_status',
                'image2_id', 'image2_status'
            ]
        else:  # mode == 'across'
            fieldnames = [
                'participant_id', 'choice',
                'task', 'category',
                'image1_id', 'image1_strategy',
                'image2_id', 'image2_strategy'
            ]

        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed)

        print(f"\nProcessed results saved to: {args.output}")

        # Print summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        # Count responses per participant
        participants = {}
        for result in processed:
            pid = result['participant_id']
            participants[pid] = participants.get(pid, 0) + 1

        print(f"Total participants: {len(participants)}")
        print(f"Total responses: {len(processed)}")
        print(f"Average responses per participant: {len(processed) / len(participants):.1f}")

        # Count by strategy-task (only for within mode)
        if mode == 'within':
            strategy_task_counts = {}
            for result in processed:
                key = f"{result['strategy']} - {result['task']}"
                strategy_task_counts[key] = strategy_task_counts.get(key, 0) + 1

            print(f"\nResponses by strategy-task:")
            for key in sorted(strategy_task_counts.keys()):
                print(f"  {key:40s}: {strategy_task_counts[key]:4d}")
        else:
            # For across mode, count by task only
            task_counts = {}
            for result in processed:
                task = result['task']
                task_counts[task] = task_counts.get(task, 0) + 1

            print(f"\nResponses by task:")
            for task in sorted(task_counts.keys()):
                print(f"  {task:40s}: {task_counts[task]:4d}")
    else:
        print("No responses found!")


if __name__ == '__main__':
    main()
