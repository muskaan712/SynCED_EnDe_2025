#!/usr/bin/env python3
# data_check.py
#
# Purpose:
#   Run basic sanity checks on dataset TSVs:
#   - File loading
#   - Row counts
#   - Label distributions
#   - Missing values
#
# Input : path to dataset TSV(s)
# Output: console summary + logs

import sys
import logging
import pandas as pd

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="data_check.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Functions ──────────────────────────────────────────────────────────
def check_file(path: str):
    """Load TSV and print/log basic dataset statistics."""
    try:
        df = pd.read_csv(path, sep="\t")
        logging.info(f"Loaded {path}: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        print(f"❌ File not found: {path}")
        return
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        print(f"❌ Error reading {path}: {e}")
        return

    print(f"\n✅ File: {path}")
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")

    # Label distribution if present
    if "target_err" in df.columns:
        counts = df["target_err"].value_counts()
        print("Label distribution:")
        print(counts)
        logging.info(f"Label distribution for {path}: {counts.to_dict()}")

    # Missing values
    missing = df.isna().sum()
    if missing.sum() > 0:
        print("Missing values per column:")
        print(missing[missing > 0])
        logging.warning(f"Missing values in {path}: {missing.to_dict()}")

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python extras/data_check.py <file1.tsv> [file2.tsv ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        check_file(path)

if __name__ == "__main__":
    main()
