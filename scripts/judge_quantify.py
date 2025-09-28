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
RUBRIC = """You are a bilingual annotator judging EN→DE translation quality for critical error detection.

You will receive:
- English source sentence (src_en)
- German translation (mt_de)
- Back-translation of German into English (bt_en)

⚠️ You must rate FIVE aspects using a full 1–5 scale. Avoid extreme ratings (1 or 5) unless clearly justified. Use 2–4 generously for most cases.
Think carefully: many errors are not totally obvious or totally subtle, and not harmless or catastrophic. For example:
- A slight word shift might be severity=2 or 3, not 1 or 5.
- A wrong country name might be obvious=2, not 1, if you have to re-read to spot it.
⚠️ Use each value (1–5) across the dataset when appropriate. Do NOT default to extremes.

You must return 5 integers in this order, separated by TABS:
error_obviousness    error_severity    localization_complexity    contextual_dependency    adequacy_deviation

Definitions (use carefully!):

1) error_obviousness
How easy is the error to spot?
1 = Very obvious (e.g., “do not” → “do”)
2 = Easy to notice during reading
3 = Requires careful reading
4 = Subtle, needs close bilingual check
5 = Extremely subtle, almost hidden

2) error_severity
Real-world impact if error goes unnoticed?
1 = Harmless (stylistic/tone)
2 = Minor inconvenience (slight confusion)
3 = Moderate misunderstanding
4 = Risky (legal, medical, procedural harm)
5 = Critical inversion (opposite intent)

3) localization_complexity
How large/spread is the error?
1 = One token/phrase
2 = Short phrase or clause
3 = Multiple locations or sentence span
4 = Sentence structure reshaped
5 = Distributed / structural scope

4) contextual_dependency
How much context is needed to judge the error?
1 = Visible in sentence alone
2 = Light background (e.g., unit, time)
3 = Some procedural/domain knowledge
4 = Strong technical/domain knowledge
5 = Only experts/doc-level view

5) adequacy_deviation
How far does the meaning drift?
1 = Minimal or stylistic only
2 = Small nuance shift
3 = Moderate loss or shift
4 = Major meaning change
5 = Opposite/false meaning

OUTPUT FORMAT (strict):
Return ONLY five integers separated by TABS in this order:
error_obviousness    error_severity    localization_complexity    contextual_dependency    adequacy_deviation
Example:  2\t3\t2\t3\t4
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
