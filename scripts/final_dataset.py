#!/usr/bin/env python3
# final.py
#
# Purpose:
#   Merge injected ERR rows with NOT rows (untouched MT outputs)
#   into a single final dataset for Critical Error Detection (CED).
#
# Input :
#   reinjected_err.tsv   (rid, src_en, mt_de, target_err, mt_de_injected)
#   not_rows.tsv         (rid, src_en, mt_de, target_err)
# Output:
#   ced_final_dataset.tsv

import pandas as pd

# ─── Config ──────────────────────────────────────────────────────────────
ERR_FILE  = "reinjected_err.tsv"
NOT_FILE  = "not_rows.tsv"
OUT_FILE  = "ced_final_dataset.tsv"

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    # Load injected ERR rows
    err_df = pd.read_csv(ERR_FILE, sep="\t")
    print(f"Loaded {len(err_df)} ERR rows")

    # Load NOT rows (no injection applied)
    not_df = pd.read_csv(NOT_FILE, sep="\t")
    print(f"Loaded {len(not_df)} NOT rows")

    # For ERR: use injected version, keep original columns
    err_df = err_df[["rid", "src_en", "mt_de_injected", "target_err"]]
    err_df.rename(columns={"mt_de_injected": "mt_de"}, inplace=True)

    # For NOT: keep as is
    not_df = not_df[["rid", "src_en", "mt_de", "target_err"]]

    # Merge
    merged = pd.concat([err_df, not_df], ignore_index=True)

    # Sanity checks
    print(f"Final merged dataset has {len(merged)} rows")
    print("Label distribution:")
    print(merged["target_err"].value_counts())

    # Save
    merged.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"✅ Wrote final dataset to {OUT_FILE}")

if __name__ == "__main__":
    main()
