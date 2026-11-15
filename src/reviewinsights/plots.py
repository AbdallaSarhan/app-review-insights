"""Shared chart styling: one palette, recessive axes, direct labels."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green). Never cycled.
# Validated for colour-vision deficiency with the dataviz palette checker.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# Fills light enough that labels on them need dark ink instead of white.
LIGHT_FILLS = {"#1baf7a", "#eda100", "#e87ba4"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK_2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def grouped_hbar(df: pd.DataFrame, title: str, xlabel: str, fmt: str = "{:.0%}"):
    """Rows are categories, columns are series. Each bar is direct-labelled."""
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(df) + 1.2))
    n = len(df.columns)
    height = 0.8 / n
    for i, col in enumerate(df.columns):
        ys = [r + (i - (n - 1) / 2) * height for r in range(len(df))]
        bars = ax.barh(ys, df[col], height=height - 0.04, color=SERIES[i], label=col)
        for bar, v in zip(bars, df[col]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, fmt.format(v),
                    va="center", fontsize=10, color=INK_2)
    ax.set_yticks(range(len(df)), df.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if n > 1:  # a single series is named by the title
        ax.legend(loc="lower right")
    return fig, ax


def stacked_share(df: pd.DataFrame, title: str, counts: pd.Series | None = None):
    """100% stacked horizontal bars. Rows are categories, columns sum to 1."""
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(df) + 1.6))
    left = pd.Series(0.0, index=df.index)
    for i, col in enumerate(df.columns):
        ax.barh(df.index, df[col], left=left, color=SERIES[i], label=col,
                edgecolor=SURFACE, linewidth=2, height=0.62)
        for y, (l, v) in enumerate(zip(left, df[col])):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=9.5,
                        color=INK if SERIES[i] in LIGHT_FILLS else "#ffffff")
        left += df[col]
    labels = [f"{a}  (n={counts[a]:,})" if counts is not None else a for a in df.index]
    ax.set_yticks(range(len(df)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    ax.legend(ncol=min(len(df.columns), 3), loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=10)
    return fig, ax


def save(fig, name: str, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")
