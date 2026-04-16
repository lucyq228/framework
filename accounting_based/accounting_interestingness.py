
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DOMAIN_COLORS: Dict[str, str] = {
    "menu_type": "#1f77b4",
    "channel": "#ff7f0e",
    "daypart": "#2ca02c",
    "promo_type": "#d62728",
    "is_offer": "#9467bd",
    "is_reward": "#8c564b",
    "is_bogo": "#e377c2",
    "ROOT": "#7f7f7f",
}


def _normalize_0_1(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _pick_ratio_effect_col(df: pd.DataFrame) -> str:
    if "total_pct_of_root_metric" in df.columns:
        return "total_pct_of_root_metric"
    if "total_abs_root" in df.columns:
        return "total_abs_root"
    raise ValueError("Could not find ratio effect column.")


def _classify_driver(base_share: float, contribution: float, growth_strength: float) -> str:
    if contribution < 0:
        return "declining"
    if base_share >= 0.12 and contribution >= 0.01:
        return "core_driver"
    if base_share < 0.12 and (contribution >= 0.005 or growth_strength >= 0.15):
        return "emerging_driver"
    return "stable"


def compute_additive_interestingness(
    decomp_df: pd.DataFrame,
    *,
    metric_name: Optional[str] = "gc",
    hierarchy_name: Optional[str] = None,
    min_level: int = 1,
    max_level: Optional[int] = None,
    contribution_col: str = "contribution_pct_of_total",
    share_col: str = "base_share_of_total",
    growth_col: str = "growth_pct",
    sibling_skew_weight: float = 0.30,
    score_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
    rank_within_hierarchy: bool = False,
) -> pd.DataFrame:
    df = decomp_df.copy()
    if "metric_name" in df.columns and metric_name is not None:
        df = df[df["metric_name"] == metric_name]
    if hierarchy_name is not None:
        df = df[df["hierarchy_name"] == hierarchy_name]
    df = df[df["level"] >= min_level].copy()
    if max_level is not None:
        df = df[df["level"] <= max_level].copy()

    df["abs_contribution"] = df[contribution_col].abs()
    df["abs_growth"] = df[growth_col].abs()
    df["base_share"] = df[share_col].fillna(0.0)

    sib = (
        df.groupby(["hierarchy_name", "level", "parent_key"], dropna=False)["abs_contribution"]
        .agg(["sum", "max", "mean", "count"])
        .reset_index()
        .rename(columns={
            "sum": "sib_abs_contribution_sum",
            "max": "sib_abs_contribution_max",
            "mean": "sib_abs_contribution_mean",
            "count": "sib_count",
        })
    )
    df = df.merge(sib, on=["hierarchy_name", "level", "parent_key"], how="left")
    df["sibling_share_of_abs_contribution"] = np.where(
        df["sib_abs_contribution_sum"] > 0,
        df["abs_contribution"] / df["sib_abs_contribution_sum"],
        0.0,
    )

    w_contrib, w_growth, w_share = score_weights
    df["score_contribution"] = _normalize_0_1(df["abs_contribution"])
    df["score_growth"] = _normalize_0_1(df["abs_growth"])
    df["score_share"] = _normalize_0_1(df["base_share"])
    df["score_sibling_skew"] = _normalize_0_1(df["sibling_share_of_abs_contribution"])

    df["interestingness_score"] = (
        w_contrib * df["score_contribution"]
        + w_growth * df["score_growth"]
        + w_share * df["score_share"]
        + sibling_skew_weight * df["score_sibling_skew"]
    )

    if rank_within_hierarchy and "hierarchy_name" in df.columns and df["hierarchy_name"].nunique(dropna=False) > 1:
        df["interestingness_rank"] = (
            df.groupby("hierarchy_name", dropna=False)["interestingness_score"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
    else:
        df["interestingness_rank"] = (
            df["interestingness_score"].rank(method="dense", ascending=False).astype(int)
        )

    df["driver_segment"] = [
        _classify_driver(bs, c, g)
        for bs, c, g in zip(df["base_share"], df[contribution_col], df["abs_growth"])
    ]
    return df.sort_values(["interestingness_score", "abs_contribution"], ascending=[False, False]).reset_index(drop=True)


def compute_ratio_interestingness(
    decomp_df: pd.DataFrame,
    *,
    metric_name: Optional[str] = "ac",
    hierarchy_name: Optional[str] = None,
    min_level: int = 1,
    max_level: Optional[int] = None,
    effect_col: Optional[str] = None,
    share_col: str = "base_weight_of_root",
    ratio_delta_col: str = "ratio_delta",
    sibling_skew_weight: float = 0.30,
    score_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
    rank_within_hierarchy: bool = False,
    rank_within_metric: bool = False,
) -> pd.DataFrame:
    df = decomp_df.copy()
    if "metric_name" in df.columns and metric_name is not None:
        df = df[df["metric_name"] == metric_name]
    if hierarchy_name is not None:
        df = df[df["hierarchy_name"] == hierarchy_name]
    df = df[df["level"] >= min_level].copy()
    if max_level is not None:
        df = df[df["level"] <= max_level].copy()

    effect_col = effect_col or _pick_ratio_effect_col(df)
    df["abs_effect"] = df[effect_col].abs()
    df["abs_ratio_delta"] = df[ratio_delta_col].abs()
    df["base_weight"] = df[share_col].fillna(0.0)

    sib = (
        df.groupby(["hierarchy_name", "level", "parent_key"], dropna=False)["abs_effect"]
        .agg(["sum", "max", "mean", "count"])
        .reset_index()
        .rename(columns={
            "sum": "sib_abs_effect_sum",
            "max": "sib_abs_effect_max",
            "mean": "sib_abs_effect_mean",
            "count": "sib_count",
        })
    )
    df = df.merge(sib, on=["hierarchy_name", "level", "parent_key"], how="left")
    df["sibling_share_of_abs_effect"] = np.where(
        df["sib_abs_effect_sum"] > 0,
        df["abs_effect"] / df["sib_abs_effect_sum"],
        0.0,
    )

    w_effect, w_delta, w_weight = score_weights
    df["score_effect"] = _normalize_0_1(df["abs_effect"])
    df["score_ratio_delta"] = _normalize_0_1(df["abs_ratio_delta"])
    df["score_weight"] = _normalize_0_1(df["base_weight"])
    df["score_sibling_skew"] = _normalize_0_1(df["sibling_share_of_abs_effect"])

    df["interestingness_score"] = (
        w_effect * df["score_effect"]
        + w_delta * df["score_ratio_delta"]
        + w_weight * df["score_weight"]
        + sibling_skew_weight * df["score_sibling_skew"]
    )

    rank_keys: List[str] = []
    if rank_within_metric and "metric_name" in df.columns:
        rank_keys.append("metric_name")
    if rank_within_hierarchy and "hierarchy_name" in df.columns:
        rank_keys.append("hierarchy_name")
    if rank_keys:
        df["interestingness_rank"] = (
            df.groupby(rank_keys, dropna=False)["interestingness_score"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
    else:
        df["interestingness_rank"] = (
            df["interestingness_score"].rank(method="dense", ascending=False).astype(int)
        )

    df["driver_segment"] = [
        _classify_driver(bs, c, g)
        for bs, c, g in zip(df["base_weight"], df[effect_col], df["abs_ratio_delta"])
    ]
    return df.sort_values(["interestingness_score", "abs_effect"], ascending=[False, False]).reset_index(drop=True)


def _find_interesting_branches_impl(
    scored_df: pd.DataFrame,
    *,
    top_k_level1: int = 3,
    child_top_k: int = 3,
) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df.copy()

    lev = pd.to_numeric(scored_df["level"], errors="coerce")
    work = scored_df.loc[lev.notna()].copy()
    lev = lev.loc[work.index]
    min_level = int(lev.min())
    lvl1 = work[lev == min_level].copy()
    lvl1 = lvl1.sort_values("interestingness_score", ascending=False).head(top_k_level1)

    keep_rows = [lvl1]
    next_level = min_level + 1
    if next_level in set(lev.dropna().astype(int).unique()):
        child_frames = []
        for parent_key in lvl1["node_key"].tolist():
            child = work[
                (lev == next_level) & (work["parent_key"] == parent_key)
            ].copy()
            child = child.sort_values("interestingness_score", ascending=False).head(child_top_k)
            if not child.empty:
                child_frames.append(child)
        if child_frames:
            keep_rows.append(pd.concat(child_frames, ignore_index=True))

    out = pd.concat(keep_rows, ignore_index=True)
    return out.sort_values(["level", "interestingness_score"], ascending=[True, False]).reset_index(drop=True)


def find_interesting_branches(
    scored_df: pd.DataFrame,
    *,
    top_k_level1: int = 3,
    child_top_k: int = 3,
    by_hierarchy: bool = False,
    by_metric: bool = False,
) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df.copy()
    group_keys: List[str] = []
    if by_metric and "metric_name" in scored_df.columns:
        group_keys.append("metric_name")
    if by_hierarchy and "hierarchy_name" in scored_df.columns:
        group_keys.append("hierarchy_name")
    if group_keys:
        parts = [
            _find_interesting_branches_impl(g, top_k_level1=top_k_level1, child_top_k=child_top_k)
            for _, g in scored_df.groupby(group_keys, dropna=False)
        ]
        return pd.concat(parts, ignore_index=True)
    return _find_interesting_branches_impl(
        scored_df, top_k_level1=top_k_level1, child_top_k=child_top_k
    )


def _resolve_plot_level_filter(
    scored_df: pd.DataFrame,
    level_filter: Optional[Sequence[int]],
) -> Tuple[List[int], List[int]]:
    """Return (wanted_levels, distinct_levels_present) using numeric levels."""
    lev = pd.to_numeric(scored_df["level"], errors="coerce")
    present_sorted = sorted({int(x) for x in lev.dropna().unique()})
    if level_filter is None:
        if len(present_sorted) <= 2:
            wanted = present_sorted
        else:
            wanted = present_sorted[:2]
    else:
        wanted = [int(x) for x in level_filter]
    return wanted, present_sorted


def plot_two_level_driver_scatter(
    scored_df: pd.DataFrame,
    *,
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
    domain_color_map: Optional[Dict[str, str]] = None,
    color_by: str = "node_dimension",
    x_col: str = "base_share_of_total",
    y_col: str = "contribution_pct_of_total",
    size_col: str = "growth_pct",
    level_filter: Optional[Sequence[int]] = None,
    annotate_top_n: int = 12,
    label_col: str = "path",
    alpha: float = 0.8,
) -> Path | None:
    if scored_df.empty:
        raise ValueError("scored_df is empty; nothing to plot.")

    wanted, present_sorted = _resolve_plot_level_filter(scored_df, level_filter)
    lev = pd.to_numeric(scored_df["level"], errors="coerce")
    df = scored_df.loc[lev.isin(wanted)].copy()
    if df.empty:
        raise ValueError(
            "No rows available after applying level_filter. "
            f"Requested levels {wanted!r}; distinct levels in scored_df are {present_sorted!r}. "
            "Pass level_filter to match your data (e.g. (0, 1) if levels are zero-based)."
        )

    if color_by not in df.columns:
        raise ValueError(
            f"color_by={color_by!r} not in columns; available: {list(df.columns)!r}"
        )

    color_map = dict(DEFAULT_DOMAIN_COLORS)
    if domain_color_map:
        color_map.update(domain_color_map)

    bubble_raw = df[size_col].abs().fillna(0.0)
    if bubble_raw.max() > 0:
        sizes = 100 + 900 * (bubble_raw / bubble_raw.max())
    else:
        sizes = pd.Series(np.repeat(160, len(df)), index=df.index)

    fig, ax = plt.subplots(figsize=(12, 8))

    cats = pd.unique(df[color_by].values)
    n_cat = len(cats)
    palette = plt.cm.tab20(np.linspace(0, 1, min(max(n_cat, 1), 20)))

    for i, cat in enumerate(cats):
        sub = df[df[color_by] == cat]
        if color_by == "node_dimension":
            color = color_map.get(str(cat), "#7f7f7f")
        else:
            color = palette[i % len(palette)]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=sizes.loc[sub.index],
            alpha=alpha,
            label=str(cat),
            color=color,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.25)

    ann = df.sort_values("interestingness_score", ascending=False).head(annotate_top_n)
    for _, row in ann.iterrows():
        ax.annotate(
            str(row[label_col]),
            (row[x_col], row[y_col]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    legend_title = {
        "node_dimension": "Domain",
        "hierarchy_name": "Hierarchy",
        "metric_name": "Ratio metric",
    }.get(color_by, color_by.replace("_", " ").title())
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    if save_path is None:
        raise ValueError("Provide save_path when show=False, or pass show=True to display only.")
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def score_gc_from_outputs(
    outputs: Dict[str, pd.DataFrame],
    *,
    hierarchy_name: Optional[str] = None,
    min_level: int = 1,
    max_level: Optional[int] = 2,
    rank_within_hierarchy: bool = False,
) -> pd.DataFrame:
    """Score GC interestingness. Pass ``hierarchy_name=None`` to include every hierarchy in ``additive__gc``."""
    gc_df = outputs["additive__gc"]
    return compute_additive_interestingness(
        gc_df,
        metric_name="gc",
        hierarchy_name=hierarchy_name,
        min_level=min_level,
        max_level=max_level,
        rank_within_hierarchy=rank_within_hierarchy,
    )


def list_ratio_metric_keys(outputs: Dict[str, pd.DataFrame]) -> List[str]:
    """Suffixes ``m`` such that ``outputs`` has ``ratio__{m}`` (e.g. ``ac``, ``upt``, ``aur``)."""
    out: List[str] = []
    for k in sorted(outputs.keys()):
        if k.startswith("ratio__"):
            out.append(k[len("ratio__") :])
    return out


def score_ratio_metric_from_outputs(
    outputs: Dict[str, pd.DataFrame],
    *,
    metric_name: str,
    hierarchy_name: Optional[str] = None,
    min_level: int = 1,
    max_level: Optional[int] = 2,
    rank_within_hierarchy: bool = False,
    rank_within_metric: bool = False,
) -> pd.DataFrame:
    """Score one ratio decomposition table: ``outputs[f'ratio__{metric_name}']``."""
    key = f"ratio__{metric_name}"
    if key not in outputs:
        avail = [k for k in outputs if k.startswith("ratio__")]
        raise KeyError(f"Missing {key!r}. Available ratio outputs: {avail!r}")
    return compute_ratio_interestingness(
        outputs[key],
        metric_name=metric_name,
        hierarchy_name=hierarchy_name,
        min_level=min_level,
        max_level=max_level,
        rank_within_hierarchy=rank_within_hierarchy,
        rank_within_metric=rank_within_metric,
    )


def score_all_ratio_metrics_from_outputs(
    outputs: Dict[str, pd.DataFrame],
    *,
    hierarchy_name: Optional[str] = None,
    min_level: int = 1,
    max_level: Optional[int] = 2,
    rank_within_hierarchy: bool = False,
    rank_within_metric: bool = False,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Score every ratio output (AC, UPT, AUR, discount, original price, …) and row-bind.

    Each ratio metric is scored with its own min–max normalization, then concatenated.
    Set ``metrics`` to a subset of :func:`list_ratio_metric_keys` to limit which tables are included.
    """
    names = list(metrics) if metrics is not None else list_ratio_metric_keys(outputs)
    if not names:
        return pd.DataFrame()
    parts = [
        score_ratio_metric_from_outputs(
            outputs,
            metric_name=m,
            hierarchy_name=hierarchy_name,
            min_level=min_level,
            max_level=max_level,
            rank_within_hierarchy=rank_within_hierarchy,
            rank_within_metric=rank_within_metric,
        )
        for m in names
    ]
    return pd.concat(parts, ignore_index=True)


def score_ac_from_outputs(
    outputs: Dict[str, pd.DataFrame],
    *,
    hierarchy_name: Optional[str] = None,
    min_level: int = 1,
    max_level: Optional[int] = 2,
    rank_within_hierarchy: bool = False,
    rank_within_metric: bool = False,
) -> pd.DataFrame:
    """Score AC only (``ratio__ac``). See :func:`score_all_ratio_metrics_from_outputs` for all ratio KPIs."""
    return score_ratio_metric_from_outputs(
        outputs,
        metric_name="ac",
        hierarchy_name=hierarchy_name,
        min_level=min_level,
        max_level=max_level,
        rank_within_hierarchy=rank_within_hierarchy,
        rank_within_metric=rank_within_metric,
    )
