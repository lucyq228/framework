"""Distributions & variation: describe-distributions, variance-decomposition, csat-reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, frame, grain, measures, pct, xy


def icc_oneway(values: pd.Series, groups: pd.Series) -> tuple[float, float]:
    """ICC(1) from one-way ANOVA, and the mean group size used."""
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    sizes = d.groupby("g").size()
    d = d[d["g"].isin(sizes[sizes >= 2].index)]
    if d["g"].nunique() < 5:
        return np.nan, np.nan
    k = d.groupby("g").size()
    n, a = len(d), len(k)
    grand = d["v"].mean()
    means = d.groupby("g")["v"].mean()
    ssb = (k * (means - grand) ** 2).sum()
    ssw = ((d["v"] - d["g"].map(means)) ** 2).sum()
    msb, msw = ssb / (a - 1), ssw / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw) if (msb + (k0 - 1) * msw) > 0 else np.nan
    return float(max(icc, 0.0)), float(k.mean())


def csat_reliability_estimate(y: pd.Series, n: pd.Series) -> tuple[float, float, float]:
    """(reliability, true-score variance, mean sampling variance) for a set of top-box rates."""
    d = pd.DataFrame({"y": y, "n": n}).dropna()
    d = d[d["n"] > 0]
    samp = (d["y"] * (1 - d["y"]) / d["n"]).mean()
    total = d["y"].var()
    true_var = max(total - samp, 1e-9)
    return float(true_var / total) if total > 0 else np.nan, float(true_var), float(samp)


@tool("describe-distributions")
def describe_distributions(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows = []
    for m in measures(params):
        for s in ("progressive", "conservative"):
            x, y, _ = xy(tc, params, m, s)
            v = f[x].dropna()
            rows.append({"measure": m, "variable": x, "n": len(v), "mean": v.mean(), "sd": v.std(),
                         "p10": v.quantile(.1), "median": v.median(), "p90": v.quantile(.9),
                         "share_at_0": (v == 0).mean(), "share_at_1": (v == 1).mean(),
                         "distinct_values": v.round(4).nunique(), "skew": sps.skew(v) if len(v) > 2 else np.nan})
        v = f[y].dropna()
        rows.append({"measure": m, "variable": y, "n": len(v), "mean": v.mean(), "sd": v.std(),
                     "p10": v.quantile(.1), "median": v.median(), "p90": v.quantile(.9),
                     "share_at_0": (v == 0).mean(), "share_at_1": (v == 1).mean(),
                     "distinct_values": v.round(4).nunique(), "skew": sps.skew(v) if len(v) > 2 else np.nan})
    t = pd.DataFrame(rows)
    ceil = t[t["variable"].str.endswith("_prog")].set_index("measure")["share_at_1"]
    return {"summary": f"[{grain(tc, params)}] Ceiling share (progressive score = 100%): "
                       + ", ".join(f"{k} {pct(v)}" for k, v in ceil.items())
                       + "; PACE scores are discrete and bounded, so rank methods are a needed cross-check",
            "tables": {"distributions": records(t)}}


@tool("variance-decomposition")
def variance_decomposition(tc: ToolContext, params: dict) -> dict:
    v = tc.data.visits
    rows, findings = [], []
    for m in measures(params):
        x, _, _ = xy(tc, params, m)
        icc, k = icc_oneway(v[x], v["glin"])
        rows.append({"measure": m, "variable": x, "icc_between_restaurant": icc, "within_share": 1 - icc if np.isfinite(icc) else np.nan,
                     "mean_visits_per_restaurant": k})
        findings.append(finding(f"ICC {x}", m, "icc", icc, int(v[x].notna().sum())))
    mo = tc.data.monthly
    mo = mo[mo["market"].isin(tc.scope["markets"])]
    rm = mo.groupby(["glin", "month"])[["taste_hi", "taste_n", "hosp_hi", "hosp_n"]].sum().reset_index()
    for m, (hi, n) in {"taste": ("taste_hi", "taste_n"), "hospitality": ("hosp_hi", "hosp_n")}.items():
        if m not in measures(params):
            continue
        r = rm[rm[n] >= 20]
        icc, k = icc_oneway(r[hi] / r[n], r["glin"])
        rows.append({"measure": m, "variable": f"csat_{m} monthly", "icc_between_restaurant": icc,
                     "within_share": 1 - icc if np.isfinite(icc) else np.nan, "mean_visits_per_restaurant": k})
        findings.append(finding(f"ICC monthly csat_{m}", m, "icc", icc, len(r)))
    t = pd.DataFrame(rows)
    return {"summary": "Between-restaurant share of variance: " + "; ".join(
                f"{r.variable} {pct(r.icc_between_restaurant)}" for r in t.itertuples()),
            "tables": {"variance_decomposition": records(t)}, "findings": findings,
            "notes": ["PACE visit scores are dominated by within-restaurant (visit-to-visit) variation; "
                      "a single visit is a noisy measure of a restaurant."]}


@tool("csat-reliability")
def csat_reliability(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    target = float(params.get("target_reliability", .7))
    rows = []
    by = "volume_band" if "volume_band" in f.columns else None
    for m in measures(params):
        _, y, n = xy(tc, params, m)
        rel, true_var, samp = csat_reliability_estimate(f[y], f[n])
        p = f[y].mean()
        needed = target / (1 - target) * p * (1 - p) / true_var
        rows.append({"measure": m, "group": "all", "rows": int(f[y].notna().sum()), "median_responses": f[n].median(),
                     "reliability": rel, "responses_needed_for_target": needed})
        groups = f.groupby(by) if by else f.groupby(pd.qcut(f[n], 3, labels=["low n", "mid n", "high n"], duplicates="drop"), observed=True)
        for g, sub in groups:
            r, _, _ = csat_reliability_estimate(sub[y], sub[n])
            rows.append({"measure": m, "group": str(g), "rows": int(sub[y].notna().sum()),
                         "median_responses": sub[n].median(), "reliability": r, "responses_needed_for_target": np.nan})
    t = pd.DataFrame(rows)
    top = t[t["group"] == "all"]
    return {"summary": f"[{grain(tc, params)}] CSAT reliability: " + "; ".join(
                f"{r.measure} {r.reliability:.2f} (median {r.median_responses:.0f} responses; "
                f"~{r.responses_needed_for_target:.0f} needed for {target:.1f})" for r in top.itertuples()),
            "tables": {"reliability": records(t)},
            "reliability": {r.measure: r.reliability for r in top.itertuples()},
            "notes": ["Reliability = true-score variance / observed variance, with binomial sampling variance p(1-p)/n."]}
