#!/usr/bin/env python3
# judge_quantify.py
#
# Purpose:
#   Run LLM-based annotation for SynCED-EnDe:
#   - Back-translate German translations (mt_de → bt_en)
#   - Judge quality on 5 rating dimensions (1–5 scale)
#   - Save ratings + texts to TSV
#
# Input : ced_final_injected.tsv
# Output: judged_quantified_nolabel.tsv

import os
import uuid
import time
import random
import logging
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm
from openai import (
    OpenAI, BadRequestError, RateLimitError,
    APIConnectionError, APITimeoutError, InternalServerError
)

# ─── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
IN_TSV   = os.path.join(DATA_DIR, "ced_final_injected.tsv")
OUT_TSV  = os.path.join(DATA_DIR, "judged_quantified_nolabel.tsv")

MODEL    = "gpt-4o"
client   = OpenAI(api_key=os.getenv("OAPI") or os.getenv("OPENAI_API_KEY"))

MAX_RETRIES     = 6
BACKOFF_BASE_S  = 1.0
JITTER_S        = 0.6
ROW_LIMIT       = 0     # >0 to test on a subset

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="judge_quantify.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Retry wrapper ───────────────────────────────────────────────────────
def _with_retries(create_kwargs: dict):
    """Call OpenAI with exponential backoff + jitter."""
    idem_key = str(uuid.uuid4())
    for attempt in range(MAX_RETRIES):
        try:
            return client.responses.create(
                extra_headers={"Idempotency-Key": idem_key},
                parallel_tool_calls=False,
                **create_kwargs
            )
        except BadRequestError as e:
            logging.error(f"BadRequestError: {e}")
            raise
        except (RateLimitError, APIConnectionError, APITimeoutError,
                InternalServerError, Exception) as e:
            wait = BACKOFF_BASE_S * (2 ** attempt) + random.uniform(0, JITTER_S)
            logging.warning(f"Retryable error ({attempt+1}/{MAX_RETRIES}): {e}. Sleeping {wait:.2f}s")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

def _extract_output_text(resp) -> str:
    """Extract assistant text from response object."""
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    try:
        return resp.output[0].content[0].text.strip()
    except Exception:
        out = []
        for item in getattr(resp, "output", []):
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []):
                    if hasattr(c, "text") and c.text:
                        out.append(c.text)
        return "\n".join(out).strip()

# ─── Judging rubric ──────────────────────────────────────────────────────
RUBRIC = """You are a bilingual annotator judging EN→DE translation quality...

[ shortened for clarity: same rubric you wrote before ]
"""

# ─── Back-translation ────────────────────────────────────────────────────
def backtranslate(mt_de: str) -> str:
    """Back-translate German sentence into English (faithfully)."""
    prompt = (
        "Translate the following German sentence into English faithfully. "
        "Return only the English sentence.\n\n"
        f"DE: {mt_de}"
    )
    resp = _with_retries({
        "model": MODEL,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 200,
        "temperature": 0.2
    })
    return _extract_output_text(resp)

# ─── Judge (parse 5 integers) ────────────────────────────────────────────
def _parse_five_ints(raw: str) -> Dict[str, int]:
    raw = raw.strip()
    if "\t" in raw:
        parts = raw.split("\t")
    elif "," in raw:
        parts = [p.strip() for p in raw.split(",")]
    else:
        parts = raw.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 values, got {len(parts)}: {raw}")
    vals = [int(p) for p in parts]
    for v in vals:
        if v < 1 or v > 5:
            raise ValueError(f"Out of range 1–5: {vals}")
    return {
        "error_obviousness": vals[0],
        "error_severity": vals[1],
        "localization_complexity": vals[2],
        "contextual_dependency": vals[3],
        "adequacy_deviation": vals[4],
    }

def judge_case(src_en: str, mt_de: str, bt_en: str) -> Dict[str, Any]:
    """Judge one (src, mt, bt) triple on 5 dimensions."""
    user_prompt = (
        f"Source (src_en): {src_en}\n"
        f"Translation (mt_de): {mt_de}\n"
        f"Back-translation (bt_en): {bt_en}\n\n"
        f"{RUBRIC}"
    )
    resp = _with_retries({
        "model": MODEL,
        "input": [{"role": "user", "content": user_prompt}],
        "max_output_tokens": 20,
        "temperature": 0.0
    })
    text = _extract_output_text(resp)
    try:
        return _parse_five_ints(text)
    except Exception as e:
        logging.warning(f"Parse failed: '{text}' ({e})")
        return {}

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    try:
        df = pd.read_csv(IN_TSV, sep="\t", header=None, names=["src_en", "mt_de", "label"])
        logging.info(f"Loaded {len(df)} rows from {IN_TSV}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {IN_TSV}")
        raise

    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)
        logging.info(f"Row limit applied: {ROW_LIMIT}")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging (LLM)"):
        src, mt, lbl = str(row["src_en"]), str(row["mt_de"]), str(row["label"])
        try:
            bt = backtranslate(mt)
            scores = judge_case(src, mt, bt)
            rows.append({
                "src_en": src, "mt_de": mt, "bt_en": bt, "label": lbl,
                **({"error_obviousness": None,
                    "error_severity": None,
                    "localization_complexity": None,
                    "contextual_dependency": None,
                    "adequacy_deviation": None} if not scores else scores)
            })
        except Exception as e:
            logging.error(f"Row failed: {e}")
            rows.append({
                "src_en": src, "mt_de": mt, "bt_en": "", "label": lbl,
                "error_obviousness": None, "error_severity": None,
                "localization_complexity": None, "contextual_dependency": None,
                "adequacy_deviation": None
            })

    out_df = pd.DataFrame(rows)
    COL_ORDER = [
        "src_en", "mt_de", "bt_en", "label",
        "error_obviousness", "error_severity",
        "localization_complexity", "contextual_dependency", "adequacy_deviation"
    ]
    remaining = [c for c in out_df.columns if c not in COL_ORDER]
    out_df[COL_ORDER + remaining].to_csv(OUT_TSV, sep="\t", index=False)

    print(f"✅ Wrote judged dataset to {OUT_TSV}")
    logging.info(f"Finished writing {len(out_df)} rows to {OUT_TSV}")

if __name__ == "__main__":
    main()
