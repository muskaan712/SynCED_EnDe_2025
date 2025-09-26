#!/usr/bin/env python3
# block_evalrows.py
#
# Purpose:
#   Remove evaluation rows from dataset based on provided blocklist of IDs.
#
# Input :
#   ced_final_dataset.tsv (rid, src_en, mt_de, target_err)
#   eval_rids.txt         (list of rids to drop, one per line)
# Output:
#   ced_trainonly_dataset.tsv (same format, without blocked rids)

import logging
import pandas as pd

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="block_evalrows.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Config ──────────────────────────────────────────────────────────────
DATA_FILE  = "ced_final_dataset.tsv"
BLOCK_FILE = "eval_rids.txt"
OUT_FILE   = "ced_trainonly_dataset.tsv"

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    try:
        df = pd.read_csv(DATA_FILE, sep="\t")
        logging.info(f"Loaded dataset: {len(df)} rows")
    except FileNotFoundError:
        logging.error(f"Input file not found: {DATA_FILE}")
        raise

    try:
        with open(BLOCK_FILE, "r", encoding="utf-8") as f:
            blocklist = {line.strip() for line in f if line.strip()}
        logging.info(f"Loaded {len(blocklist)} blocked rids")
    except FileNotFoundError:
        logging.error(f"Blocklist not found: {BLOCK_FILE}")
        raise

    # Filter
    filtered = df[~df["rid"].astype(str).isin(blocklist)]
    logging.info(f"Filtered dataset has {len(filtered)} rows (kept only non-blocked)")

    # Save
    filtered.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"✅ Wrote filtered dataset to {OUT_FILE}")

if __name__ == "__main__":
    main()
