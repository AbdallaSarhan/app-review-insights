"""Collect Google Play reviews and save them untouched.

Raw text is never modified on disk: every later step reads `data/raw/reviews.csv`
and derives what it needs, so the analysis can always be traced back to what the
reviewer actually wrote.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews_all

from .config import APPS, RAW_REVIEWS, SINCE

COLUMNS = {
    "reviewId": "review_id",
    "content": "text",
    "score": "rating",
    "thumbsUpCount": "thumbs_up",
    "reviewCreatedVersion": "app_version",
    "at": "posted_at",
}


def fetch_reviews(package: str, lang: str = "en", country: str = "us") -> pd.DataFrame:
    """Return every review the Play Store API will give us for one app."""
    raw = reviews_all(package, lang=lang, country=country, sort=Sort.NEWEST, sleep_milliseconds=0)
    df = pd.DataFrame(raw, columns=list(COLUMNS)).rename(columns=COLUMNS)
    df.insert(0, "app", package)
    return df


def scrape(apps: dict[str, str], out: Path = RAW_REVIEWS, since: str = SINCE) -> pd.DataFrame:
    frames = []
    for package, name in apps.items():
        df = fetch_reviews(package)
        df = df[df["posted_at"] >= pd.Timestamp(since)]
        if df.empty:
            print(f"WARNING: no reviews returned for {name} ({package})")
        else:
            print(f"{name:<22} {len(df):>6} reviews")
        frames.append(df)

    reviews = pd.concat(frames, ignore_index=True).drop_duplicates("review_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(out, index=False)
    print(f"Saved {len(reviews)} reviews to {out}")
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_REVIEWS)
    args = parser.parse_args()
    scrape(APPS, args.out)


if __name__ == "__main__":
    main()
