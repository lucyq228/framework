"""Panel effect: between-within-correlation, mixed-effects-model, fixed-effects-panel."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from corr_core import stats
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, fmt, frame, measures, xy


def _visit_frame(tc, params):
    return frame(tc, dict(params, grain="visit"))


@tool("between-within-correlation")
def between_within_correlation(tc: ToolContext, params: dict) -> dict:
    f = _visit_frame(tc, params)
    findings, rows = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, ["glin"])
        means = d.groupby("glin")[[x, y]].mean()
        rb, pb = stats.corr(means[x], means[y])
        counts = d.groupby("glin")[x].transform("size")
        w = d[counts >= 2]
        dx = w[x] - w.groupby("glin")[x].transform("mean")
        dy = w[y] - w.groupby("glin")[y].transform("mean")
        rw, pw = stats.corr(dx, dy)
        n_eff = len(w) - w["glin"].nunique()
        rows += [{"measure": m, "component": "between restaurants", "r": rb, "p": pb, "n": len(means)},
                 {"measure": m, "component": "within restaurant (over time)", "r": rw, "p": pw, "n": n_eff}]
        findings += [finding(f"{m} between", m, "pearson_r", rb, len(means), stats.fisher_ci(rb, len(means)), pb),
                     finding(f"{m} within", m, "pearson_r", rw, n_eff, stats.fisher_ci(rw, n_eff), pw)]
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure} {r.component} r={fmt(r.r)} (n={r.n})" for r in t.itertuples()),
            "findings": findings, "tables": {"between_within": records(t)},
            "notes": ["Within-restaurant correlation asks: when a restaurant's PACE is better than its own usual, "
                      "is its CSAT around that visit better too?"]}


def _controls(f: pd.DataFrame, params: dict, default: list[str]) -> list[str]:
    wanted = params.get("controls") or default
    return [c for c in wanted if c in f.columns and f[c].nunique() > 1]


@tool("mixed-effects-model")
def mixed_effects_model(tc: ToolContext, params: dict) -> dict:
    f = _visit_frame(tc, params)
    findings, rows = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        ctrl = _controls(f, params, ["market", "survey_name"])
        d = stats.clean_xy(f, x, y, ["glin"] + ctrl)
        formula = f"{y} ~ {x}" + "".join(f" + C({c})" for c in ctrl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(formula, d, groups=d["glin"]).fit(reml=True, method="lbfgs")
        coef, se = res.params[x], res.bse[x]
        re_var = float(res.cov_re.iloc[0, 0])
        icc = re_var / (re_var + res.scale)
        rows.append({"measure": m, "formula": formula, "coef_per_10pp": coef * .1, "ci_low": (coef - 1.96 * se) * .1,
                     "ci_high": (coef + 1.96 * se) * .1, "p": res.pvalues[x], "restaurant_icc": icc, "n": len(d),
                     "restaurants": d["glin"].nunique()})
        findings.append(finding(f"{m} mixed-model slope per 10pp", m, "coef_per_10pp", coef * .1, len(d),
                                ((coef - 1.96 * se) * .1, (coef + 1.96 * se) * .1), res.pvalues[x]))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: +10pp PACE -> {r.coef_per_10pp * 100:+.2f}pp CSAT "
                                 f"[{r.ci_low * 100:+.2f}, {r.ci_high * 100:+.2f}] (restaurant ICC {r.restaurant_icc:.2f})"
                                 for r in t.itertuples()),
            "findings": findings, "tables": {"mixed_model": records(t)}}


@tool("fixed-effects-panel")
def fixed_effects_panel(tc: ToolContext, params: dict) -> dict:
    f = _visit_frame(tc, params)
    findings, rows = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, ["glin", "market", "month_of_year"])
        specs = {"pooled": f"{y} ~ {x}", "market FE": f"{y} ~ {x} + C(market)",
                 "market + month FE": f"{y} ~ {x} + C(market) + C(month_of_year)", "restaurant FE": f"{y} ~ {x} + C(glin)"}
        for name, formula in specs.items():
            res = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["glin"])[0]})
            coef, ci = res.params[x], res.conf_int().loc[x].tolist()
            rows.append({"measure": m, "specification": name, "coef_per_10pp": coef * .1, "ci_low": ci[0] * .1,
                         "ci_high": ci[1] * .1, "p": res.pvalues[x], "n": len(d)})
            findings.append(finding(f"{m} {name} slope per 10pp", m, "coef_per_10pp", coef * .1, len(d),
                                    (ci[0] * .1, ci[1] * .1), res.pvalues[x], specification=name))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure} {r.specification}: {r.coef_per_10pp * 100:+.2f}pp" for r in t.itertuples()),
            "findings": findings, "tables": {"fixed_effects": records(t)},
            "notes": ["Restaurant fixed effects remove every stable difference between restaurants; what is left is "
                      "visit-to-visit co-movement, which is small and noisy with 3-5 visits per restaurant per year."]}
