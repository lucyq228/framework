"""Small matplotlib helpers. Every chart is saved under runs/<run>/charts/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#DA291C", "#FFC72C", "#27251F", "#4A7FB5", "#7A9A01", "#8E6C8A"]


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def bar(path: Path, labels, values, title: str, xlabel: str = "", ylabel: str = "", pct: bool = True,
        errors=None) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(len(values)), values, color=PALETTE[0], yerr=errors, capsize=3 if errors is not None else 0)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save(fig, path)


def lines(path: Path, series: dict, title: str, xlabel: str = "", ylabel: str = "", pct: bool = False,
          bands: dict | None = None) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (name, (xs, ys)) in enumerate(series.items()):
        ax.plot(range(len(xs)), ys, marker="o", color=PALETTE[i % len(PALETTE)], label=name)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(range(len(xs)), lo, hi, color=PALETTE[i % len(PALETTE)], alpha=.15)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axhline(0, color="#999", lw=.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return _save(fig, path)


def forest(path: Path, labels, estimates, lows, highs, title: str, xlabel: str = "correlation") -> str:
    fig, ax = plt.subplots(figsize=(7, max(2.5, .32 * len(labels) + 1)))
    ys = list(range(len(labels)))[::-1]
    ax.errorbar(estimates, ys, xerr=[[e - lo for e, lo in zip(estimates, lows)],
                                     [hi - e for e, hi in zip(estimates, highs)]],
                fmt="o", color=PALETTE[0], ecolor="#888", capsize=3)
    ax.axvline(0, color="#999", lw=.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    return _save(fig, path)


def scatter(path: Path, groups: dict, title: str, xlabel: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, (name, (xs, ys)) in enumerate(groups.items()):
        ax.scatter(xs, ys, s=12, alpha=.6, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return _save(fig, path)
