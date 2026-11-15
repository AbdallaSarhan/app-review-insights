"""Keyboard labeller for the human benchmark.

    python -m reviewinsights.label

Shows one review at a time from data/labels/human_labels.csv. Press 1-6 to pick
an intent; the file is saved after every key press, so you can quit at any time
and pick up where you left off. Run it in a real terminal (it reads single keys).

Keys:  1-6 label   b back   n skip   q quit
"""
from __future__ import annotations

import os
import shutil
import sys
import termios
import textwrap
import tty

import pandas as pd

from .config import HUMAN_LABELS
from .llm import INTENTS

KEYS = {str(i): label for i, label in enumerate(INTENTS, 1)}


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _save(df: pd.DataFrame) -> None:
    tmp = HUMAN_LABELS.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, HUMAN_LABELS)  # atomic: a crash never leaves a half-written file


def _render(df: pd.DataFrame, i: int, note: str) -> None:
    width = min(shutil.get_terminal_size().columns, 100)
    done = df["intent"].isin(INTENTS).sum()
    row = df.iloc[i]
    print("\033[2J\033[H", end="")  # clear screen
    print(f"Review {i + 1}/{len(df)}   ·   {done} labelled   ·   app: {row['app']}\n")
    print(textwrap.fill(str(row["text"]), width), "\n")
    current = row["intent"] if row["intent"] in INTENTS else "—"
    print(f"current label: {current}\n")
    for key, label in KEYS.items():
        print(f"  {key}  {label:<16} {textwrap.shorten(INTENTS[label], width - 22)}")
    print("\n  b back   n skip   q quit    (if several fit, pick the lowest number)")
    if note:
        print(f"\n{note}")


def main() -> None:
    if not sys.stdin.isatty():
        raise SystemExit("Run this in a terminal: python -m reviewinsights.label")
    df = pd.read_csv(HUMAN_LABELS, dtype=str, keep_default_na=False)
    unlabelled = [i for i in range(len(df)) if df.at[i, "intent"] not in INTENTS]
    i = unlabelled[0] if unlabelled else 0
    note = ""
    while True:
        _render(df, i, note)
        note = ""
        key = _getch().lower()
        if key in ("q", "\x03"):  # q or Ctrl+C
            break
        if key in KEYS:
            df.at[i, "intent"] = KEYS[key]
            _save(df)
            rest = [j for j in range(i + 1, len(df)) if df.at[j, "intent"] not in INTENTS]
            rest = rest or [j for j in range(len(df)) if df.at[j, "intent"] not in INTENTS]
            if not rest:
                print("\033[2J\033[HAll reviews labelled. Next: python -m reviewinsights.benchmark score")
                return
            i = rest[0]
        elif key == "b":
            i = max(i - 1, 0)
        elif key == "n":
            i = (i + 1) % len(df)
        else:
            note = f"Unknown key {key!r}. Use 1-6, b, n or q."
    done = df["intent"].isin(INTENTS).sum()
    print(f"\nSaved. {done}/{len(df)} labelled. Run again to continue.")


if __name__ == "__main__":
    main()
