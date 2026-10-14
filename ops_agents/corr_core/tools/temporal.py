"""Time & longitudinal: lag-window-scan, rolling-correlation, change-on-change, seasonality-adjust."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from corr_core import charts, features, stats
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, fmt, frame, measures, min_n, xy


@tool("lag-window-scan")
def lag_window_scan(tc: ToolContext, params: dict) -> dict:
    windows = params.get("windows") or list(features.WINDOWS)
    rows, findings = [], []
    for w in windows:
        f = frame(tc, dict(params, grain="visit", window=w))
        for m in measures(params):
            x, y, n = xy(tc, params, m)
            d = stats.clean_xy(f, x, y, [n])
            r, p = stats.corr(d[x], d[y])
            ci = stats.fisher_ci(r, len(d))
            rows.append({"measure": m, "window": w, "r": r, "ci_low": ci[0], "ci_high": ci[1], "p": p, "n": len(d),
                         "median_csat_responses": d[n].median()})
            findings.append(finding(f"{m} window {w}", m, "pearson_r", r, len(d), ci, p, window=w))
    t = pd.DataFrame(rows)
    series, bands = {}, {}
    for m in measures(params):
        sub = t[t["measure"] == m]
        series[m] = (sub["window"].tolist(), sub["r"].tolist())
        bands[m] = (sub["ci_low"].tolist(), sub["ci_high"].tolist())
    chart = charts.lines(tc.chart_path("lag_windows"), series, "PACE-CSAT correlation by CSAT window around the visit",
                         xlabel="CSAT window", ylabel="r", bands=bands)
    best = t.sort_values("r", ascending=False).groupby("measure").head(1)
    return {"summary": "Best-aligned window: " + "; ".join(f"{r.measure} {r.window} r={fmt(r.r)}" for r in best.itertuples())
                       + " | year window: " + "; ".join(f"{r.measure} r={fmt(r.r)}" for r in t[t["window"] == "year"].itertuples()),
            "findings": findings, "tables": {"windows": records(t)}, "charts": [chart],
            "notes": ["Wider windows add CSAT responses (less noise) but move further from the visit date; the peak shows the "
                      "time span over which a PACE visit is representative of what customers experience.",
                      "pre-windows test whether CSAT before the visit already reflects execution (persistence), "
                      "post-windows test whether it lasts."]}


@tool("rolling-correlation")
def rolling_correlation(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, dict(params, grain="visit"))
    period = params.get("period", "half")
    f = f.assign(_period=features.period_label(f["visit_date"], period))
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        for label, sub in f.groupby("_period"):
            d = stats.clean_xy(sub, x, y)
            r, p = stats.corr(d[x], d[y]) if len(d) >= min_n(tc, params) else (np.nan, np.nan)
            ci = stats.fisher_ci(r, len(d))
            rows.append({"measure": m, "period": label, "r": r, "ci_low": ci[0], "ci_high": ci[1], "p": p, "n": len(d)})
            findings.append(finding(f"{m} {label}", m, "pearson_r", r, len(d), ci, p, period=label))
    t = pd.DataFrame(rows)
    series = {m: (t[t["measure"] == m]["period"].tolist(), t[t["measure"] == m]["r"].tolist()) for m in measures(params)}
    chart = charts.lines(tc.chart_path("rolling"), series, f"PACE-CSAT correlation by {period}", ylabel="r")
    spread = t.groupby("measure")["r"].agg(["min", "max"])
    return {"summary": "Range across periods: " + "; ".join(f"{k} {fmt(v['min'])} to {fmt(v['max'])}" for k, v in spread.iterrows()),
            "findings": findings, "tables": {"rolling": records(t)}, "charts": [chart]}


@tool("change-on-change")
def change_on_change(tc: ToolContext, params: dict) -> dict:
    p2 = dict(params, grain="visit", visit_types=params.get("visit_types") or ["CFV"])
    f = frame(tc, p2).sort_values(["glin", "visit_date"])
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, ["glin", "visit_date"])
        d = d.assign(dx=d.groupby("glin")[x].diff(), dy=d.groupby("glin")[y].diff(),
                     gap_days=d.groupby("glin")["visit_date"].diff().dt.days).dropna(subset=["dx", "dy"])
        r, p = stats.corr(d["dx"], d["dy"])
        ci = stats.fisher_ci(r, len(d))
        rows.append({"measure": m, "pairs": len(d), "median_days_between_visits": d["gap_days"].median(),
                     "r_change": r, "ci_low": ci[0], "ci_high": ci[1], "p": p,
                     "csat_change_when_pace_improves": d.loc[d["dx"] > 0, "dy"].mean(),
                     "csat_change_when_pace_drops": d.loc[d["dx"] < 0, "dy"].mean()})
        findings.append(finding(f"{m} change-on-change", m, "pearson_r", r, len(d), ci, p))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: r(ΔPACE, ΔCSAT)={fmt(r.r_change)} over {r.pairs} consecutive-visit pairs" for r in t.itertuples()),
            "findings": findings, "tables": {"changes": records(t)},
            "notes": ["Uses consecutive visits of the same type at the same restaurant; it removes stable restaurant differences "
                      "but is sensitive to noise in single visits."]}


@tool("seasonality-adjust")
def seasonality_adjust(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, dict(params, grain="visit"))
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, ["month_of_year", "market"])
        raw, _ = stats.corr(d[x], d[y])
        rx = smf.ols(f"{x} ~ C(month_of_year) + C(market)", d).fit().resid
        ry = smf.ols(f"{y} ~ C(month_of_year) + C(market)", d).fit().resid
        adj, p = stats.corr(rx, ry)
        season = d.groupby("month_of_year")[y].mean()
        rows.append({"measure": m, "raw_r": raw, "season_market_adjusted_r": adj, "p": p, "n": len(d),
                     "csat_seasonal_range": season.max() - season.min(),
                     "lowest_csat_month": int(season.idxmin()), "highest_csat_month": int(season.idxmax())})
        findings.append(finding(f"{m} seasonality-adjusted", m, "pearson_r", adj, len(d), stats.fisher_ci(adj, len(d)), p))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure} raw r={fmt(r.raw_r)} -> adjusted {fmt(r.season_market_adjusted_r)}; CSAT seasonal "
                                 f"range {r.csat_seasonal_range:.1%} (low month {r.lowest_csat_month})" for r in t.itertuples()),
            "findings": findings, "tables": {"seasonality": records(t)}}
