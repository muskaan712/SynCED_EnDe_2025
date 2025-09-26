#!/usr/bin/env python3
# reinject.py
#
# Purpose:
#   Re-run injection only for rows where mt_de_injected is empty
#   in previously generated files. Ensures all ERR rows get injected.
#
# Input : injected_err.tsv (with possible empty injections)
# Output: reinjected_err.tsv (all rows with injections)

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
    filename="gpt4o_reinject.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Config ──────────────────────────────────────────────────────────────
IN_TSV  = "injected_err.tsv"
OUT_TSV = "reinjected_err.tsv"

client  = OpenAI(api_key=os.getenv("OAPI") or os.getenv("OPENAI_API_KEY"))
MODEL   = "gpt-4o"

MAX_RETRIES    = 6
BACKOFF_BASE_S = 1.0
JITTER_S       = 0.6
BATCH_SIZE     = 5

# ─── Retry wrapper ───────────────────────────────────────────────────────
def _with_retries(create_kwargs: dict):
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
    texts = []
    for item in getattr(resp, "output", []):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            for c in item.content:
                if hasattr(c, "text") and c.text:
                    texts.append(c.text.strip())
    return texts

# ─── Prompt ──────────────────────────────────────────────────────────────
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

def reinject_batch(rows: pd.DataFrame) -> list[str]:
    """Send a batch of rows to GPT-4o for reinjection."""
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
    to_fix = df[df["mt_de_injected"].isna() | (df["mt_de_injected"].str.strip() == "")]
    print(f"Found {len(to_fix)} rows needing reinjection")

    reinjected = []
    for start in tqdm(range(0, len(to_fix), BATCH_SIZE), desc="Reinjecting"):
        chunk = to_fix.iloc[start:start + BATCH_SIZE]
        try:
            outputs = reinject_batch(chunk)
        except Exception as e:
            logging.error(f"Batch {start} failed: {e}")
            outputs = [""] * len(chunk)

        for (i, row), inj in zip(chunk.iterrows(), outputs):
            reinjected.append((row.rid, row.src_en, row.mt_de, row.target_err, inj))

    # Merge reinjected back into df
    reinjected_df = pd.DataFrame(
        reinjected,
        columns=["rid", "src_en", "mt_de", "target_err", "mt_de_injected"]
    )
    fixed_df = pd.concat([df[df["mt_de_injected"].notna() & (df["mt_de_injected"].str.strip() != "")],
                          reinjected_df], ignore_index=True)

    fixed_df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"✅ Wrote {len(fixed_df)} rows to {OUT_TSV}")

if __name__ == "__main__":
    main()
