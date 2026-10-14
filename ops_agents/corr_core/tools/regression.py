"""Multivariable regression: controlled-regression, binomial-glm, partial-correlation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from corr_core import stats
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, fmt, frame, grain, measures, xy

VISIT_CONTROLS = ["market", "survey_name", "month_of_year", "restaurant_type", "ownership_type", "volume_band"]
RESTAURANT_CONTROLS = ["market", "restaurant_type", "ownership_type", "volume_band"]


def controls(tc: ToolContext, params: dict, f: pd.DataFrame) -> list[str]:
    wanted = params.get("controls") or (VISIT_CONTROLS if grain(tc, params) == "visit" else RESTAURANT_CONTROLS)
    return [c for c in wanted if c in f.columns and f[c].nunique() > 1]


def _rhs(ctrl: list[str]) -> str:
    return "".join(f" + C({c})" for c in ctrl)


def _groups(d: pd.DataFrame) -> np.ndarray:
    return pd.factorize(d["glin"])[0]


@tool("controlled-regression")
def controlled_regression(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows, findings = [], []
    for m in measures(params):
        x, y, n = xy(tc, params, m)
        ctrl = controls(tc, params, f)
        d = stats.clean_xy(f, x, y, [n, "glin"] + ctrl).dropna(subset=ctrl)
        for label, rhs in (("no controls", ""), ("with controls", _rhs(ctrl))):
            res = smf.wls(f"{y} ~ {x}{rhs}", d, weights=d[n]).fit(cov_type="cluster", cov_kwds={"groups": _groups(d)})
            ci = res.conf_int().loc[x].tolist()
            rows.append({"measure": m, "model": label, "controls": ", ".join(ctrl) if rhs else "", "coef_per_10pp": res.params[x] * .1,
                         "ci_low": ci[0] * .1, "ci_high": ci[1] * .1, "p": res.pvalues[x], "r2": res.rsquared, "n": len(d)})
            findings.append(finding(f"{m} WLS {label} slope per 10pp", m, "coef_per_10pp", res.params[x] * .1, len(d),
                                    (ci[0] * .1, ci[1] * .1), res.pvalues[x], model=label))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure} {r.model}: +10pp PACE -> {r.coef_per_10pp * 100:+.2f}pp CSAT (p={fmt(r.p, 3)})"
                                 for r in t.itertuples()),
            "findings": findings, "tables": {"regression": records(t)},
            "notes": ["Weighted by CSAT responses; standard errors clustered by restaurant. Coefficients are associations."]}


@tool("binomial-glm")
def binomial_glm(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows, findings = [], []
    for m in measures(params):
        x, y, n = xy(tc, params, m)
        ctrl = controls(tc, params, f)
        d = stats.clean_xy(f, x, y, [n, "glin"] + ctrl).dropna(subset=ctrl)
        d = d[d[n] > 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.glm(f"{y} ~ {x}{_rhs(ctrl)}", d, family=sm.families.Binomial(), var_weights=d[n]).fit(
                cov_type="cluster", cov_kwds={"groups": _groups(d)})
        b, ci = res.params[x], res.conf_int().loc[x].tolist()
        disp = res.pearson_chi2 / res.df_resid
        rows.append({"measure": m, "odds_ratio_per_10pp": np.exp(b * .1), "ci_low": np.exp(ci[0] * .1), "ci_high": np.exp(ci[1] * .1),
                     "p": res.pvalues[x], "overdispersion": disp, "controls": ", ".join(ctrl), "n": len(d)})
        findings.append(finding(f"{m} odds ratio per 10pp", m, "odds_ratio_per_10pp", np.exp(b * .1), len(d),
                                (np.exp(ci[0] * .1), np.exp(ci[1] * .1)), res.pvalues[x]))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: odds of a top-box rating x{r.odds_ratio_per_10pp:.3f} per +10pp PACE "
                                 f"(overdispersion {r.overdispersion:.1f})" for r in t.itertuples()),
            "findings": findings, "tables": {"glm": records(t)},
            "notes": ["Overdispersion well above 1 means restaurants differ for reasons beyond PACE; clustered errors account for it."]}


@tool("partial-correlation")
def partial_correlation(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        ctrl = controls(tc, params, f)
        d = stats.clean_xy(f, x, y, ctrl).dropna(subset=ctrl)
        rhs = _rhs(ctrl).lstrip(" +") or "1"
        rx = smf.ols(f"{x} ~ {rhs}", d).fit().resid
        ry = smf.ols(f"{y} ~ {rhs}", d).fit().resid
        r, p = stats.corr(rx, ry)
        raw, _ = stats.corr(d[x], d[y])
        rows.append({"measure": m, "raw_r": raw, "partial_r": r, "p": p, "controls": ", ".join(ctrl), "n": len(d)})
        findings.append(finding(f"{m} partial r", m, "partial_r", r, len(d), stats.fisher_ci(r, len(d) - len(ctrl)), p))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: raw r={fmt(r.raw_r)} -> partial r={fmt(r.partial_r)} controlling {r.controls}"
                                 for r in t.itertuples()),
            "findings": findings, "tables": {"partial": records(t)}}
