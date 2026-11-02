"""Static charts for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def churn_rate_bars(table: pd.DataFrame, col: str, title: str, path: Path, sort: bool = True) -> None:
    t = table.sort_values("churn_rate") if sort else table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(t) + 1.2))
    ax.barh(t[col].astype(str), t["churn_rate"] * 100, color=BLUE, height=0.6)
    for y, (rate, n) in enumerate(zip(t["churn_rate"], t["customers"])):
        ax.text(rate * 100 + 0.8, y, f"{rate:.0%}  (n={n:,})", va="center", color=INK_2, fontsize=9)
    ax.set_xlabel("Churn rate (%)")
    ax.set_xlim(0, max(t["churn_rate"]) * 100 * 1.35)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    _save(fig, path)


def gains_curve(y_true, proba, highlight: float, captured: float, path: Path) -> None:
    ranked = np.asarray(y_true)[np.argsort(-np.asarray(proba))]
    x = np.arange(1, len(ranked) + 1) / len(ranked)
    cum = np.cumsum(ranked) / ranked.sum()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1.2, linestyle="--", label="Random outreach")
    ax.plot(np.r_[0, x], np.r_[0, cum], color=BLUE, linewidth=2, label="Model ranking")
    ax.scatter([highlight], [captured], s=50, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"Top {highlight:.0%} contacted\n→ {captured:.0%} of churners reached",
                (highlight, captured), xytext=(0.04, 0.78),
                fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": INK_2})
    ax.set(xlabel="Share of customers contacted (highest risk first)",
           ylabel="Share of all churners reached", xlim=(0, 1), ylim=(0, 1.02))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Model ranking reaches churners far faster than random")
    _save(fig, path)


def driver_bars(coefs: pd.Series, path: Path, n: int = 6) -> None:
    sel = pd.concat([coefs.head(n), coefs.tail(n)])
    colors = [ORANGE if v > 0 else BLUE for v in sel]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(sel.index, sel.values, color=colors, height=0.6)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Logistic regression coefficient (standardised / one-hot)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["Higher churn risk", "Lower churn risk"], frameon=False, loc="upper left")
    ax.set_title("Strongest associations with churn (not causal effects)")
    _save(fig, path)
