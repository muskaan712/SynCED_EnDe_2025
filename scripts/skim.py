#!/usr/bin/env python3
# skim.py
#
# Purpose:
#   Quickly skim the dataset by showing the first few rows.
#   Used for sanity checks after generation/merging.
#
# Input : ced_final_dataset.tsv
# Output: console preview (head of dataset)

import logging
import pandas as pd

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="skim.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Config ──────────────────────────────────────────────────────────────
DATA_FILE = "ced_final_dataset.tsv"
N_PREVIEW = 10   # how many rows to show

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    try:
        df = pd.read_csv(DATA_FILE, sep="\t")
        logging.info(f"Loaded dataset: {len(df)} rows")
    except FileNotFoundError:
        logging.error(f"Input file not found: {DATA_FILE}")
        raise

    # Preview
    print(f"Showing first {N_PREVIEW} rows from {DATA_FILE}:\n")
    print(df.head(N_PREVIEW))

    logging.info(f"Displayed {N_PREVIEW} preview rows")

if __name__ == "__main__":
    main()
