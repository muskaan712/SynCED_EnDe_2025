#!/usr/bin/env python3
# labels.py
#
# Purpose:
#   Analyze and filter label distribution in the dataset.
#   Helps inspect how many ERR vs NOT rows exist.
#
# Input : ced_final_dataset.tsv
# Output: label_stats.tsv (counts of each label)

import logging
import pandas as pd

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="labels.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Config ──────────────────────────────────────────────────────────────
DATA_FILE  = "ced_final_dataset.tsv"
STATS_FILE = "label_stats.tsv"

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    try:
        df = pd.read_csv(DATA_FILE, sep="\t")
        logging.info(f"Loaded dataset: {len(df)} rows")
    except FileNotFoundError:
        logging.error(f"Input file not found: {DATA_FILE}")
        raise

    # Compute counts per label
    counts = df["target_err"].value_counts().reset_index()
    counts.columns = ["label", "count"]

    logging.info("Label distribution calculated")
    for _, row in counts.iterrows():
        logging.info(f"Label {row['label']}: {row['count']} rows")

    # Save stats
    counts.to_csv(STATS_FILE, sep="\t", index=False)
    print(f"✅ Label stats written to {STATS_FILE}")
    print(counts)

if __name__ == "__main__":
    main()
