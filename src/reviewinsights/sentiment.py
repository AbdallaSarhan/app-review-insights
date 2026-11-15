"""Lexicon-based sentiment scoring and evaluation against star ratings.

Star ratings are the reviewer's own verdict, which makes them a free (if noisy)
ground truth: 1-2 stars = negative, 4-5 = positive, 3 = neutral.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER's documented thresholds for its compound score.
VADER_NEUTRAL_BAND = 0.05


@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def vader_score(text: str) -> float:
    return _vader().polarity_scores(text)["compound"] if isinstance(text, str) else 0.0


def textblob_score(text: str) -> float:
    return TextBlob(text).sentiment.polarity if isinstance(text, str) else 0.0


def rating_label(rating: int) -> str:
    return "negative" if rating <= 2 else "positive" if rating >= 4 else "neutral"


def score_label(score: float, band: float = VADER_NEUTRAL_BAND) -> str:
    return "negative" if score <= -band else "positive" if score >= band else "neutral"


def evaluate(scores: pd.Series, ratings: pd.Series) -> dict[str, float]:
    """How well a sentiment score agrees with the star rating.

    * spearman: rank correlation with the 1-5 rating (uses every review).
    * balanced_accuracy: positive vs negative only (3-star reviews dropped),
      averaged per class so the 5-star majority can't inflate it.
    * neg_recall: share of 1-2 star reviews the model calls negative. This is
      the number that matters for finding complaints.
    """
    rho = spearmanr(scores, ratings).correlation
    truth = ratings.map(rating_label)
    pred = scores.map(score_label)
    polar = truth != "neutral"
    t, p = truth[polar], pred[polar]
    pos_recall = float((p[t == "positive"] == "positive").mean())
    neg_recall = float((p[t == "negative"] == "negative").mean())
    return {
        "spearman": float(rho),
        "balanced_accuracy": (pos_recall + neg_recall) / 2,
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
    }


def confusion(scores: pd.Series, ratings: pd.Series) -> pd.DataFrame:
    order = ["negative", "neutral", "positive"]
    table = pd.crosstab(ratings.map(rating_label), scores.map(score_label),
                        rownames=["stars say"], colnames=["model says"], normalize="index")
    return table.reindex(index=order, columns=order, fill_value=0.0).mul(100).round(1)


def disagreements(df: pd.DataFrame, score_col: str, n: int = 10, seed: int = 0) -> pd.DataFrame:
    """Reviews where the text and the star rating point in opposite directions.

    High-star reviews that read negative often hide bug reports or feature
    requests ("love it, but it crashes when..."), which is exactly what a
    product team wants to find.
    """
    mask = (df["rating"] >= 4) & (df[score_col] <= -0.3)
    hits = df.loc[mask]
    return hits.sample(min(n, len(hits)), random_state=seed) if len(hits) else hits


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, len(values))) for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
