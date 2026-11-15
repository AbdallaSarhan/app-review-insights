"""Classify review intent with a local LLM served by Ollama.

Each review gets one primary intent and a sentiment. Output is constrained to a
JSON schema, temperature is 0, and results are cached by review_id so re-runs
only classify new reviews.

    python -m reviewinsights.llm            # classify everything not yet cached
    python -m reviewinsights.llm --limit 20 # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
import pandas as pd

from .config import LLM_LABELS, LLM_MODEL, RAW_REVIEWS

# Ordered by priority: when a review fits several, the earliest one wins.
# The human labelling guide (data/labels/README.md) uses the same rule.
INTENTS = {
    "bug": "Something is broken: crashes, lost data or progress, features not working, sync or login failures.",
    "pricing": "Complaints or questions about cost, subscriptions, paywalls, trials, refunds or ads.",
    "feature_request": "Asks for something new or for an existing feature to work differently.",
    "usability": "Confusing, tedious or annoying design that is not broken: onboarding, navigation, notifications, UI.",
    "praise": "Positive feedback with no actionable request or problem.",
    "other": "Too vague to act on, off-topic, or not about the app.",
}
SENTIMENTS = ["negative", "neutral", "positive"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
    },
    "required": ["intent", "sentiment"],
}

SYSTEM_PROMPT = (
    "You label Google Play reviews of habit-tracking and self-improvement apps for a product team.\n\n"
    "Choose ONE primary intent. If several apply, pick the one listed first:\n"
    + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    + "\n\nAlso give the overall sentiment of the text (negative, neutral or positive), "
    "judged from the words, not from any star rating.\n"
    'Reply with JSON only, e.g. {"intent": "bug", "sentiment": "negative"}.'
)


def classify(text: str, model: str = LLM_MODEL) -> dict[str, str]:
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Review:\n{text}"}],
        format=SCHEMA,
        think=False,
        options={"temperature": 0},
    )
    result = json.loads(response.message.content)
    if result.get("intent") not in INTENTS or result.get("sentiment") not in SENTIMENTS:
        raise ValueError(f"unexpected output: {result}")
    return result


def classify_all(reviews: pd.DataFrame, out: Path = LLM_LABELS, model: str = LLM_MODEL,
                 workers: int = 4) -> pd.DataFrame:
    """Classify reviews not already in `out`, appending results as they finish."""
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["review_id", "intent", "sentiment", "model"])
    todo = reviews[~reviews["review_id"].isin(done["review_id"])]
    print(f"{len(done)} cached, {len(todo)} to classify with {model}")
    if todo.empty:
        return done

    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    start, failures = time.time(), 0
    with out.open("a") as fh, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(classify, row.text, model): row.review_id for row in todo.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as exc:  # keep going; failed reviews are retried on the next run
                failures += 1
                print(f"  failed {futures[fut]}: {exc}")
                continue
            pd.DataFrame([{"review_id": futures[fut], **res, "model": model}]).to_csv(
                fh, header=write_header, index=False)
            write_header = False
            fh.flush()
            if i % 100 == 0:
                rate = i / (time.time() - start)
                print(f"  {i}/{len(todo)}  ({rate:.1f}/s, ~{(len(todo) - i) / rate / 60:.0f} min left)")
    print(f"Done in {(time.time() - start) / 60:.1f} min, {failures} failures")
    return pd.read_csv(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only classify this many (for testing)")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    reviews = pd.read_csv(RAW_REVIEWS).dropna(subset=["text"])
    if args.limit:
        reviews = reviews.sample(args.limit, random_state=0)
    classify_all(reviews, model=args.model, workers=args.workers)


if __name__ == "__main__":
    main()
