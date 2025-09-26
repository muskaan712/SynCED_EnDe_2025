#!/usr/bin/env python3
# inject.py
#
# Purpose:
#   Inject controlled errors into existing EN→DE translations using GPT-4o.
#
# Input : err_for_injection.tsv  (rid, src_en, mt_de, target_err)
# Output: injected_err.tsv       (rid, src_en, mt_de, target_err, mt_de_injected)

import os
import uuid
import time
import random
import logging
import pandas as pd
from tqdm import tqdm
from openai import (
    OpenAI, BadRequestError, RateLimitError,
    APIConnectionError, APITimeoutError, InternalServerError
)

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="gpt4o_inject_errors.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Config ──────────────────────────────────────────────────────────────
IN_TSV  = "err_for_injection.tsv"
OUT_TSV = "injected_err.tsv"

client  = OpenAI(api_key=os.getenv("OAPI") or os.getenv("OPENAI_API_KEY"))
MODEL   = "gpt-4o"

MAX_RETRIES    = 6
BACKOFF_BASE_S = 1.0
JITTER_S       = 0.6
BATCH_SIZE     = 5   # rows per request

# ─── Retry wrapper for OpenAI calls ──────────────────────────────────────
def _with_retries(create_kwargs: dict):
    """Call `client.responses.create` with exponential backoff and jitter."""
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
            logging.warning(f"Retryable error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Sleeping {wait:.2f}s")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

def _extract_output_text(resp) -> list[str]:
    """Extract assistant text responses from OpenAI Response object."""
    texts = []
    for item in getattr(resp, "output", []):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            for c in item.content:
                if hasattr(c, "text") and c.text:
                    texts.append(c.text.strip())
    return texts

# ─── Prompt template ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a data generator for critical error detection (CED).\n"
    "Given a correct German translation, inject a *controlled* critical error "
    "according to the requested error type. Only return the rewritten German.\n\n"
    "Error types:\n"
    "- NAM: corrupt, drop, or replace a named entity.\n"
    "- NUM: change a number, unit, or date/time.\n"
    "- SEN: flip negation or sentiment polarity.\n"
    "- SAF: remove or change a safety/health-critical phrase.\n"
    "- TOX: hallucinate toxicity (mild hate/profanity) not present in the source.\n\n"
    "Do not explain. Return only the modified German sentence."
)

def inject_batch(rows: pd.DataFrame) -> list[str]:
    """Send a batch of rows to GPT-4o for error injection."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for _, row in rows.iterrows():
        messages.append({
            "role": "user",
            "content": (
                f"Source (EN): {row['src_en']}\n"
                f"German (DE): {row['mt_de']}\n"
                f"Inject error type: {row['target_err']}"
            )
        })

    resp = _with_retries({
        "model": MODEL,
        "input": messages,
        "max_output_tokens": 220,
        "temperature": 0.6,
    })
    return _extract_output_text(resp)

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(IN_TSV, sep="\t")
    print(f"Loaded {len(df)} ERR rows")

    injected_rows = []
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Injecting"):
        chunk = df.iloc[start:start + BATCH_SIZE]
        try:
            outputs = inject_batch(chunk)
        except Exception as e:
            logging.error(f"Batch {start} failed: {e}")
            outputs = [""] * len(chunk)

        for (i, row), inj in zip(chunk.iterrows(), outputs):
            injected_rows.append((row.rid, row.src_en, row.mt_de, row.target_err, inj))

    out_df = pd.DataFrame(
        injected_rows,
        columns=["rid", "src_en", "mt_de", "target_err", "mt_de_injected"]
    )
    out_df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"✅ Wrote {len(out_df)} injected rows to {OUT_TSV}")

if __name__ == "__main__":
    main()
