"""Relative importance & shared information: dominance-analysis, commonality-analysis,
collinearity-check, index-weight-sensitivity."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from corr_core import features, stats
from corr_core.datasets import y_col
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, frame, measures, pct, xy
from corr_core.tools.components import component_label

DEFAULT_PREDICTORS = {
    "taste": ["cat_beef_pass", "cat_chicken_pass", "cat_fries_pass", "cat_beverage_pass", "cat_dessert_pass"],
    "hospitality": ["tp_order_pass", "tp_pay_pass", "tp_present_pass", "tp_care_pass"],
}


def _r2(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return float(1 - resid.var() / y.var())


def _restaurant(tc, params):
    return frame(tc, dict(params, grain=params.get("grain") or "restaurant_period"))


@tool("dominance-analysis")
def dominance_analysis(tc: ToolContext, params: dict) -> dict:
    f = _restaurant(tc, params)
    rows, findings = [], []
    for m in measures(params):
        preds = [p for p in (params.get("predictors") or DEFAULT_PREDICTORS[m]) if p in f.columns][:8]
        y = y_col(m)
        d = f[preds + [y]].dropna()
        Y = d[y].to_numpy()
        k = len(preds)
        r2 = {}
        for size in range(k + 1):
            for combo in combinations(range(k), size):
                r2[combo] = _r2(d[[preds[i] for i in combo]].to_numpy(), Y)
        total = r2[tuple(range(k))]
        for i, p in enumerate(preds):
            shap = 0.0
            for size in range(k):
                for combo in combinations([j for j in range(k) if j != i], size):
                    w = factorial(size) * factorial(k - size - 1) / factorial(k)
                    shap += w * (r2[tuple(sorted(combo + (i,)))] - r2[combo])
            rows.append({"measure": m, "predictor": component_label(p), "shapley_r2": shap,
                         "share_of_explained": shap / total if total > 0 else np.nan, "n": len(d)})
            findings.append(finding(f"{m} dominance {component_label(p)}", m, "shapley_r2", shap, len(d)))
        rows.append({"measure": m, "predictor": "TOTAL R²", "shapley_r2": total, "share_of_explained": 1.0, "n": len(d)})
    t = pd.DataFrame(rows)
    top = t[t["predictor"] != "TOTAL R²"].sort_values("shapley_r2", ascending=False).groupby("measure").head(1)
    tot = t[t["predictor"] == "TOTAL R²"].set_index("measure")["shapley_r2"]
    return {"summary": "; ".join(f"{r.measure}: components explain R²={tot[r.measure]:.3f}; largest share {r.predictor} "
                                 f"({pct(r.share_of_explained)})" for r in top.itertuples()),
            "findings": findings, "tables": {"dominance": records(t)}}


@tool("commonality-analysis")
def commonality_analysis(tc: ToolContext, params: dict) -> dict:
    f = _restaurant(tc, params)
    tx, _, _ = xy(tc, params, "taste")
    hx, _, _ = xy(tc, params, "hospitality")
    rows = []
    for m in measures(params):
        y = y_col(m)
        d = f[[tx, hx, y, "market_group"]].dropna()
        for scope_label, dd in (("pooled", d), ("within market group", d.assign(**{
                c: d[c] - d.groupby("market_group")[c].transform("mean") for c in (tx, hx, y)}))):
            Y = dd[y].to_numpy()
            r_both = _r2(dd[[tx, hx]].to_numpy(), Y)
            r_t = _r2(dd[[tx]].to_numpy(), Y)
            r_h = _r2(dd[[hx]].to_numpy(), Y)
            rows.append({"csat_outcome": m, "scope": scope_label, "r2_both": r_both, "unique_pace_taste": r_both - r_h,
                         "unique_pace_hosp": r_both - r_t, "shared": r_t + r_h - r_both, "n": len(dd)})
    t = pd.DataFrame(rows)
    w = t[t["scope"] == "within market group"]
    return {"summary": "Within market group: " + "; ".join(
                f"CSAT {r.csat_outcome} R²={r.r2_both:.3f} (unique Taste {r.unique_pace_taste:.3f}, unique Hospitality "
                f"{r.unique_pace_hosp:.3f}, shared {r.shared:.3f})" for r in w.itertuples()),
            "tables": {"commonality": records(t)},
            "notes": ["Shared variance is what PACE Taste and PACE Hospitality explain together; a large shared part means "
                      "both indexes reflect a general 'well-run restaurant' factor."]}


@tool("collinearity-check")
def collinearity_check(tc: ToolContext, params: dict) -> dict:
    f = _restaurant(tc, params)
    rows = []
    sets = {"taste components": DEFAULT_PREDICTORS["taste"], "hospitality touchpoints": DEFAULT_PREDICTORS["hospitality"],
            "PACE measures": ["taste_prog", "hosp_prog"]}
    for name, cols in sets.items():
        cols = [c for c in cols if c in f.columns]
        d = f[cols].dropna()
        X = np.column_stack([np.ones(len(d)), d.to_numpy()])
        cond = float(np.linalg.cond(d.to_numpy() - d.to_numpy().mean(0)))
        for i, c in enumerate(cols):
            rows.append({"set": name, "variable": component_label(c), "vif": variance_inflation_factor(X, i + 1),
                         "condition_number": cond, "n": len(d)})
    t = pd.DataFrame(rows)
    worst = t.sort_values("vif", ascending=False).iloc[0]
    return {"summary": f"Highest VIF {worst['vif']:.2f} ({worst['variable']} in {worst['set']}); "
                       + ("no collinearity concern" if worst["vif"] < 5 else "collinearity: interpret component weights with care"),
            "tables": {"vif": records(t)}}


@tool("index-weight-sensitivity")
def index_weight_sensitivity(tc: ToolContext, params: dict) -> dict:
    rp = _restaurant(tc, params).dropna(subset=["csat_taste", "csat_hosp", "taste_prog", "hosp_prog"])
    weights = params.get("pace_weights") or [.3, .4, .5, .6, .7]
    s = "prog" if (params.get("scoring") or tc.scope.get("primary_scoring", "progressive")).startswith("prog") else "cons"
    base = features.gse_index(rp, .5)[f"gse_kpi_{s}"]
    base_q = pd.qcut(base.rank(method="first"), 5, labels=False)
    rows = []
    for w in weights:
        idx = features.gse_index(rp, w)[f"gse_kpi_{s}"]
        q = pd.qcut(idx.rank(method="first"), 5, labels=False)
        rho, _ = stats.corr(idx, base, "spearman")
        rows.append({"pace_weight": w, "rank_corr_with_50_50": rho, "share_changing_quintile": float((q != base_q).mean())})
    pace_part = (rp[f"taste_{s}"] + rp[f"hosp_{s}"]) / 2
    csat_part = (rp["csat_taste"] + rp["csat_hosp"]) / 2
    idx = features.gse_index(rp, .5)[f"gse_kpi_{s}"]
    var_pace, var_csat = (.5 * pace_part).var(), (.5 * csat_part).var()
    cov = 2 * np.cov(.5 * pace_part, .5 * csat_part)[0, 1]
    share_pace = (var_pace + cov / 2) / idx.var()
    r_pace, _ = stats.corr(idx, pace_part)
    r_csat, _ = stats.corr(idx, csat_part)
    t = pd.DataFrame(rows)
    return {"summary": f"At 50/50, PACE contributes {pct(share_pace)} of the GSE KPI's variation across restaurants "
                       f"(sd PACE part {pace_part.std():.3f} vs CSAT part {csat_part.std():.3f}); index correlates "
                       f"{r_pace:.2f} with PACE and {r_csat:.2f} with CSAT",
            "findings": [finding("PACE share of GSE KPI variance", None, "variance_share", share_pace, len(rp)),
                         finding("GSE KPI vs PACE part", None, "pearson_r", r_pace, len(rp)),
                         finding("GSE KPI vs CSAT part", None, "pearson_r", r_csat, len(rp))],
            "tables": {"weights": records(t)},
            "notes": ["Nominal weights are not effective weights: the component with more spread across restaurants drives "
                      "the ranking regardless of the 50/50 split."]}
