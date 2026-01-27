#!/usr/bin/env python3
"""
Analyze evaluation results to see which edit types are chosen most often.
"""

import sys
import os
import glob
import pandas as pd
import yaml
from datetime import datetime


def main():
    if len(sys.argv) != 2:
        print("Usage: python combine_results.py <path_to_results_folder>")
        sys.exit(1)

    dfs = []
    for filename in glob.glob(os.path.join(sys.argv[1], "**/results_methods.csv"), recursive=True):
        df = pd.read_csv(filename)
        df["source_file"] = filename
        df["model"] = filename.split("/")[-2]

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(
        os.path.join(sys.argv[1], "head2head_results-%s.csv" % datetime.now().strftime("%Y%m%d-%H%M%S")),
        index=False
    )


if __name__ == "__main__":
    main()
