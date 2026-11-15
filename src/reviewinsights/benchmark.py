"""Human-labelled benchmark for the LLM intent classifier.

    python -m reviewinsights.benchmark sample   # write data/labels/human_labels.csv to fill in
    python -m reviewinsights.benchmark score    # compare the LLM with your labels

The sample is drawn without looking at LLM output and the file hides star
ratings, so neither can bias the human labels.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import APPS, HUMAN_LABELS, LLM_LABELS, RAW_REVIEWS, SEED, TARGET_APP
from .llm import INTENTS

# Blind labels from a second annotator (Claude), made from the same guide without
# seeing star ratings, LLM output or the human labels.
SECOND_ANNOTATOR = HUMAN_LABELS.parent / "claude_labels.csv"

# Reviews per app, split evenly across 1-2, 3 and 4-5 stars where possible.
# Balanced on purpose: a random sample would be mostly short praise, which says
# little about whether the model can tell a bug from a feature request.
SAMPLE_PLAN = {TARGET_APP: 120, "com.kryptoniteappstudio.kaizen": 40, "com.habitrpg.android.habitica": 40}


def _bucket(rating: int) -> str:
    return "low" if rating <= 2 else "mid" if rating == 3 else "high"


def make_sample(min_chars: int = 15) -> pd.DataFrame:
    reviews = pd.read_csv(RAW_REVIEWS).dropna(subset=["text"])
    reviews = reviews[reviews["text"].str.len() >= min_chars]
    reviews["bucket"] = reviews["rating"].map(_bucket)
    parts = []
    for app, n in SAMPLE_PLAN.items():
        pool = reviews[reviews["app"] == app]
        per_bucket = n // 3
        picked = [g.sample(min(per_bucket, len(g)), random_state=SEED) for _, g in pool.groupby("bucket")]
        picked = pd.concat(picked)
        if len(picked) < n:  # top up from whatever is left if a bucket was small
            rest = pool.drop(picked.index)
            picked = pd.concat([picked, rest.sample(n - len(picked), random_state=SEED)])
        parts.append(picked)
    sample = pd.concat(parts).sample(frac=1, random_state=SEED)  # shuffle so apps are mixed
    out = sample[["review_id", "app", "text"]].assign(app=lambda d: d["app"].map(APPS), intent="")
    return out


def score() -> None:
    human = pd.read_csv(HUMAN_LABELS)
    human["intent"] = human["intent"].str.strip().str.lower()
    labelled = human[human["intent"].isin(INTENTS)]
    bad = human[human["intent"].notna() & (human["intent"] != "") & ~human["intent"].isin(INTENTS)]
    if len(bad):
        print(f"Ignoring {len(bad)} rows with unknown labels: {sorted(bad['intent'].unique())}")
    if labelled.empty:
        raise SystemExit(f"No labels yet. Fill in the `intent` column of {HUMAN_LABELS}.")

    llm = pd.read_csv(LLM_LABELS)[["review_id", "intent"]].rename(columns={"intent": "llm"})
    df = labelled.merge(llm, on="review_id", how="left")
    missing = df["llm"].isna().sum()
    if missing:
        print(f"{missing} labelled reviews have no LLM label yet; run `python -m reviewinsights.llm`.")
        df = df.dropna(subset=["llm"])

    print(f"\n{len(df)} reviews labelled\n")
    report = intent_report(df["intent"], df["llm"])
    print(report.round(2).to_string())
    acc = (df["intent"] == df["llm"]).mean()
    majority = df["intent"].value_counts(normalize=True).iloc[0]
    print(f"\naccuracy {acc:.1%}   macro-F1 {report.loc[list(INTENTS), 'f1'].mean():.2f}"
          f"   (always guessing the most common label: {majority:.1%})")
    print(f"Cohen's kappa (you vs LLM): {cohen_kappa(df['intent'], df['llm']):.2f}")
    print("\nconfusion (rows = you, columns = LLM)")
    print(pd.crosstab(df["intent"], df["llm"]).reindex(index=list(INTENTS), columns=list(INTENTS), fill_value=0))

    # A second annotator (Claude, labelling blind from the same guide) gives a ceiling:
    # if two careful annotators only agree X% of the time, the LLM can't be expected to beat X.
    if SECOND_ANNOTATOR.exists():
        second = pd.read_csv(SECOND_ANNOTATOR)[["review_id", "intent"]].rename(columns={"intent": "second"})
        both = labelled.merge(second, on="review_id")
        agree = (both["intent"] == both["second"]).mean()
        print(f"\nsecond annotator: {len(both)} reviews, agreement {agree:.1%}, "
              f"Cohen's kappa {cohen_kappa(both['intent'], both['second']):.2f}")
        print("confusion (rows = you, columns = second annotator)")
        print(pd.crosstab(both["intent"], both["second"])
              .reindex(index=list(INTENTS), columns=list(INTENTS), fill_value=0))


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement corrected for chance: 0 = no better than chance, 1 = perfect."""
    observed = (a.values == b.values).mean()
    expected = sum(a.value_counts(normalize=True).get(k, 0) * b.value_counts(normalize=True).get(k, 0)
                   for k in INTENTS)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def intent_report(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    rows = {}
    for label in INTENTS:
        tp = ((pred == label) & (truth == label)).sum()
        precision = tp / max((pred == label).sum(), 1)
        recall = tp / max((truth == label).sum(), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[label] = {"precision": precision, "recall": recall, "f1": f1, "support": (truth == label).sum()}
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "score"])
    parser.add_argument("--force", action="store_true", help="overwrite an existing label file")
    args = parser.parse_args()
    if args.command == "sample":
        if HUMAN_LABELS.exists() and not args.force:
            raise SystemExit(f"{HUMAN_LABELS} exists (it may hold your labels). Use --force to overwrite.")
        HUMAN_LABELS.parent.mkdir(parents=True, exist_ok=True)
        make_sample().to_csv(HUMAN_LABELS, index=False)
        print(f"Wrote {HUMAN_LABELS}")
    else:
        score()


if __name__ == "__main__":
    main()
