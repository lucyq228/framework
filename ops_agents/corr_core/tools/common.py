"""Shared parameter handling for analysis tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from corr_core import stats
from corr_core.datasets import Ctx, x_col, y_col


@dataclass
class ToolContext:
    run_dir: Path
    scope: dict | None
    entry: dict
    label: str
    data: Ctx | None = None
    extra: dict = field(default_factory=dict)

    def chart_path(self, name: str) -> Path:
        return self.run_dir / "charts" / f"{self.label}__{name}.png"


def measures(params: dict) -> list[str]:
    m = params.get("measures") or ["taste", "hospitality"]
    return [m] if isinstance(m, str) else list(m)


def scoring(tc: ToolContext, params: dict) -> str:
    return params.get("scoring") or tc.scope.get("primary_scoring", "progressive")


def grain(tc: ToolContext, params: dict) -> str:
    return params.get("grain") or tc.scope.get("primary_grain", "visit")


def min_n(tc: ToolContext, params: dict) -> int:
    return int(params.get("min_n") or tc.scope.get("min_sample", {}).get("min_n_correlation", 30))


def min_responses(tc: ToolContext, params: dict) -> int:
    return int(params.get("min_csat_responses") or tc.scope.get("min_sample", {}).get("min_csat_responses", 20))


def frame(tc: ToolContext, params: dict) -> pd.DataFrame:
    """Analysis frame for the requested grain, window, CSAT channel, and visit types,
    with rows below the minimum CSAT response count removed."""
    g = grain(tc, params)
    f = tc.data.frame(g, window=params.get("window"), csat_channel=params.get("csat_channel", "all"),
                      visit_types=params.get("visit_types"), period=params.get("period"))
    floor = min_responses(tc, params)
    keep = (f["csat_taste_n"].fillna(0) >= floor) | (f["csat_hosp_n"].fillna(0) >= floor)
    f = f[keep]
    if params.get("market_group"):
        f = f[f["market_group"] == params["market_group"]]
    if params.get("market"):
        f = f[f["market"] == params["market"]]
    return f


def xy(tc: ToolContext, params: dict, measure: str, scoring_override: str | None = None) -> tuple[str, str, str]:
    """(x column, y column, response-count column) for a measure."""
    s = scoring_override or scoring(tc, params)
    y = y_col(measure)
    return x_col(measure, s), y, f"{y}_n"


def corr_with_ci(df: pd.DataFrame, x: str, y: str, method: str = "pearson", bootstrap: bool = False,
                 cluster: str | None = "glin", n_boot: int = 300) -> tuple[float, float, tuple, int]:
    d = stats.clean_xy(df, x, y, [cluster] if cluster else None)
    r, p = stats.corr(d[x], d[y], method)
    if bootstrap and len(d) >= 10:
        ci = stats.cluster_bootstrap(d, lambda s: stats.corr(s[x], s[y], method)[0], cluster=cluster, n_boot=n_boot)
    else:
        ci = stats.fisher_ci(r, len(d))
    return r, p, ci, len(d)


def guard(n: int, minimum: int) -> str | None:
    return None if n >= minimum else f"n={n} below the minimum {minimum}; estimate suppressed"


def pct(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.1%}"


def fmt(v, d: int = 2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"
