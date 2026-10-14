"""Component decomposition: component-correlation, failure-reason-analysis, taste-vs-hospitality-crossover."""

from __future__ import annotations

import numpy as np
import pandas as pd

from corr_core import charts, stats
from corr_core.datasets import y_col
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, fmt, frame, grain, measures, min_n, xy

COMPONENT_PREFIX = {"taste": ("cat_",), "hospitality": ("ch_", "tp_")}
CHANNEL_NAMES = {"dt": "Drive Thru", "ir": "In Restaurant", "dl": "Delivery", "cb": "Curbside"}


def component_columns(f: pd.DataFrame, measure: str) -> list[str]:
    cols = [c for c in f.columns if c.startswith(COMPONENT_PREFIX[measure])]
    if measure == "hospitality" and "general_service_rate" in f.columns:
        cols.append("general_service_rate")
    return cols


def component_label(col: str) -> str:
    core = col.replace("_pass", "")
    if core.startswith("ch_"):
        return f"channel: {CHANNEL_NAMES.get(core[3:], core[3:])}"
    if core.startswith("tp_"):
        return f"touchpoint: {core[3:]}"
    if core.startswith("cat_"):
        return f"product: {core[4:].replace('_', ' ')}"
    return core.replace("_", " ")


@tool("component-correlation")
def component_correlation(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    findings, rows, chart_paths = [], [], []
    floor = min_n(tc, params)
    for m in measures(params):
        y = y_col(m)
        for c in component_columns(f, m):
            d = stats.clean_xy(f, c, y)
            r, p = stats.corr(d[c], d[y]) if len(d) >= floor else (np.nan, np.nan)
            ci = stats.fisher_ci(r, len(d))
            rows.append({"measure": m, "component": component_label(c), "column": c, "mean": d[c].mean(),
                         "r": r, "ci_low": ci[0], "ci_high": ci[1], "p": p, "n": len(d)})
            findings.append(finding(f"{m} {component_label(c)}", m, "pearson_r", r, len(d), ci, p, component=c))
        sub = pd.DataFrame([r for r in rows if r["measure"] == m and np.isfinite(r["r"])])
        if len(sub):
            chart_paths.append(charts.forest(tc.chart_path(f"components_{m}"), sub["component"].tolist(), sub["r"].tolist(),
                                             sub["ci_low"].tolist(), sub["ci_high"].tolist(), f"{m}: component correlation with CSAT"))
    t = pd.DataFrame(rows)
    best = t.dropna(subset=["r"]).sort_values("r", ascending=False).groupby("measure").head(1)
    return {"summary": f"[{grain(tc, params)}] Strongest component: " + "; ".join(
                f"{r.measure} -> {r.component} r={fmt(r.r)} (n={r.n})" for r in best.itertuples()),
            "findings": findings, "tables": {"components": records(t)}, "charts": chart_paths,
            "notes": ["Diagnostic categories (Hot Coffee, McCafe) are outside the Taste Index but shown for comparison."]}


@tool("failure-reason-analysis")
def failure_reason_analysis(tc: ToolContext, params: dict) -> dict:
    p2 = dict(params, grain="visit")
    f = frame(tc, p2)
    y = y_col("taste")
    findings, rows = [], []
    for c in [c for c in f.columns if c.startswith("fail_")]:
        d = stats.clean_xy(f, c, y)
        flag = (d[c] > 0).astype(float)
        r, p = stats.corr(flag, d[y])
        rows.append({"dimension": c.replace("fail_", ""), "visits_with_failure": int(flag.sum()), "share": flag.mean(),
                     "csat_taste_with_failure": d.loc[flag == 1, y].mean(), "csat_taste_without": d.loc[flag == 0, y].mean(),
                     "r_pb": r, "p": p, "n": len(d)})
        findings.append(finding(f"taste failure {c.replace('fail_', '')}", "taste", "point_biserial_r", r, len(d),
                                stats.fisher_ci(r, len(d)), p))
    t = pd.DataFrame(rows)
    return {"summary": "Visits with a failure on: " + "; ".join(
                f"{r.dimension} CSAT Taste {r.csat_taste_with_failure:.1%} vs {r.csat_taste_without:.1%} (n fail={r.visits_with_failure})"
                for r in t.itertuples() if r.visits_with_failure > 0),
            "status": "partial", "findings": findings, "tables": {"failure_dimensions": records(t)},
            "notes": ["PARTIAL: dimensions are only recorded on failed checks and CSAT has no dimension split, "
                      "so this compares visits with and without a failure type, not dimension-level performance."]}


@tool("taste-vs-hospitality-crossover")
def taste_vs_hospitality_crossover(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    tx, ty, _ = xy(tc, params, "taste")
    hx, hy, _ = xy(tc, params, "hospitality")
    pairs = [(tx, ty, "same: PACE Taste ~ CSAT Taste"), (hx, hy, "same: PACE Hospitality ~ CSAT Hospitality"),
             (tx, hy, "cross: PACE Taste ~ CSAT Hospitality"), (hx, ty, "cross: PACE Hospitality ~ CSAT Taste"),
             (tx, hx, "PACE Taste ~ PACE Hospitality"), (ty, hy, "CSAT Taste ~ CSAT Hospitality (halo)")]
    findings, rows = [], []
    for a, b, label in pairs:
        d = stats.clean_xy(f, a, b)
        r, p = stats.corr(d[a], d[b])
        ci = stats.fisher_ci(r, len(d))
        rows.append({"pair": label, "r": r, "ci_low": ci[0], "ci_high": ci[1], "p": p, "n": len(d)})
        findings.append(finding(label, None, "pearson_r", r, len(d), ci, p))
    t = pd.DataFrame(rows)
    matrix = stats.clean_xy(f, tx, ty, [hx, hy])[[tx, hx, ty, hy]].corr()
    return {"summary": f"[{grain(tc, params)}] " + "; ".join(f"{r.pair} r={fmt(r.r)}" for r in t.itertuples()),
            "findings": findings, "tables": {"pairs": records(t), "matrix": records(matrix.reset_index())},
            "notes": ["If cross correlations are close to same-measure correlations, the two indexes overlap (halo) and "
                      "the 50/50 split partly double-counts one general factor."]}
