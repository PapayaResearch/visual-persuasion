#!/usr/bin/env python3
"""
Combine results_mitigations.csv files from multiple evaluation runs.
"""

import sys
import os
import glob
import pandas as pd
import yaml
from datetime import datetime


def main():
    if len(sys.argv) != 2:
        print("Usage: python combine_results_mitigations.py <path_to_results_folder>")
        sys.exit(1)

    dfs = []
    for filename in glob.glob(os.path.join(sys.argv[1], "**/results_mitigations.csv"), recursive=True):
        df = pd.read_csv(filename)
        df["source_file"] = filename
        df["model"] = filename.split("/")[-2]

        cfg_path = os.path.join(os.path.dirname(filename), "config_mitigations.yaml")

        with open(cfg_path) as yaml_file:
            cfg = yaml.safe_load(yaml_file)

        cfg_flat = pd.json_normalize(cfg).to_dict(orient="records")[0]

        for key, value in cfg_flat.items():
            if key in ["task.name", "evaluate.strategy_name"]:
                df.loc[:, key] = value

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(
        os.path.join(sys.argv[1], "combined_mitigations-%s.csv" % datetime.now().strftime("%Y%m%d-%H%M%S")),
        index=False
    )


if __name__ == "__main__":
    main()
