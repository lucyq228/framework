"""Nonlinearity & thresholds: binned-means, lowess-spline, breakpoint-regression, nonlinear-dependence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from sklearn.feature_selection import mutual_info_regression

from corr_core import charts, stats
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, fmt, frame, grain, measures, xy


def _bins(x: pd.Series, n_bins: int) -> pd.Series:
    if x.round(4).nunique() <= n_bins + 2:
        return x.round(2).astype(str)
    return pd.qcut(x, n_bins, duplicates="drop").astype(str)


@tool("binned-means")
def binned_means(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows, chart_paths = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y)
        d = d.assign(bin=_bins(d[x], int(params.get("bins", 5))))
        g = d.groupby("bin").agg(pace_mean=(x, "mean"), csat_mean=(y, "mean"), csat_sd=(y, "std"), n=(y, "size")).reset_index()
        g = g.sort_values("pace_mean")
        g["ci_half"] = 1.96 * g["csat_sd"] / np.sqrt(g["n"])
        g["measure"] = m
        rows.append(g)
        chart_paths.append(charts.bar(tc.chart_path(f"binned_{m}"), g["bin"].tolist(), g["csat_mean"].tolist(),
                                      f"{m}: mean CSAT by PACE band", xlabel="PACE band", errors=g["ci_half"].tolist()))
    t = pd.concat(rows)
    spread = t.groupby("measure")["csat_mean"].agg(lambda s: s.iloc[-1] - s.iloc[0])
    return {"summary": f"[{grain(tc, params)}] CSAT difference from lowest to highest PACE band: "
                       + "; ".join(f"{k} {v:+.1%}" for k, v in spread.items()),
            "tables": {"bins": records(t)}, "charts": chart_paths}


@tool("lowess-spline")
def lowess_spline(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows, curves, findings = [], {}, []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y).rename(columns={x: "px", y: "cy"})
        fit = sm.nonparametric.lowess(d["cy"], d["px"], frac=float(params.get("frac", .6)), return_sorted=True)
        grid = np.linspace(d["px"].quantile(.05), d["px"].quantile(.95), 12)
        curves[m] = [{"pace": float(g), "csat_smoothed": float(np.interp(g, fit[:, 0], fit[:, 1]))} for g in grid]
        lin = smf.ols("cy ~ px", d).fit()
        df = int(params.get("spline_df", 4)) if d["px"].nunique() > 6 else 2
        spl = smf.ols(f"cy ~ bs(px, df={df}, degree=min({df}, 3))", d).fit() if d["px"].nunique() > 6 else smf.ols("cy ~ px + I(px**2)", d).fit()
        ftest = spl.compare_f_test(lin) if spl.df_model > lin.df_model else (np.nan, np.nan, 0)
        rows.append({"measure": m, "linear_r2": lin.rsquared, "spline_r2": spl.rsquared, "nonlinearity_F": ftest[0],
                     "nonlinearity_p": ftest[1], "n": len(d)})
        findings.append(finding(f"{m} spline vs linear", m, "r2_gain", spl.rsquared - lin.rsquared, len(d), p=ftest[1]))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: linear R²={r.linear_r2:.3f}, spline R²={r.spline_r2:.3f} (p={fmt(r.nonlinearity_p, 3)})"
                                 for r in t.itertuples()),
            "findings": findings, "tables": {"fit": records(t)}, "curves": curves}


@tool("breakpoint-regression")
def breakpoint_regression(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y)
        xs, ys = d[x].to_numpy(), d[y].to_numpy()
        lin = np.polyfit(xs, ys, 1)
        sse_lin = ((ys - np.polyval(lin, xs)) ** 2).sum()
        best = None
        for c in np.unique(np.quantile(xs, np.linspace(.15, .85, 15))):
            X = np.column_stack([np.ones_like(xs), xs, np.maximum(xs - c, 0)])
            beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
            sse = ((ys - X @ beta) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, c, beta)
        sse, c, beta = best
        n = len(xs)
        f_stat = (sse_lin - sse) / (sse / (n - 3)) if sse > 0 else np.nan
        p = sps.f.sf(f_stat, 1, n - 3) if np.isfinite(f_stat) else np.nan
        rows.append({"measure": m, "breakpoint": c, "slope_below": beta[1], "slope_above": beta[1] + beta[2],
                     "F_vs_linear": f_stat, "p_naive": p, "n": n})
        findings.append(finding(f"{m} breakpoint at {c:.2f}", m, "slope_change", beta[2], n, p=p, breakpoint=round(float(c), 3)))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: best breakpoint PACE={r.breakpoint:.0%}, slope below {r.slope_below:.2f} vs above "
                                 f"{r.slope_above:.2f} (naive p={fmt(r.p_naive, 3)})" for r in t.itertuples()),
            "findings": findings, "tables": {"breakpoints": records(t)},
            "notes": ["The breakpoint is chosen by search, so the naive p-value is optimistic; treat it as exploratory unless "
                      "it also appears in binned means and LOWESS."]}


@tool("nonlinear-dependence")
def nonlinear_dependence(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    rows, findings = [], []
    rng = np.random.default_rng(27)
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y)
        mi = mutual_info_regression(d[[x]], d[y], random_state=27)[0]
        mi_null = np.mean([mutual_info_regression(d[[x]], rng.permutation(d[y].to_numpy()), random_state=27)[0] for _ in range(5)])
        dcor = stats.distance_corr(d[x], d[y])
        r, _ = stats.corr(d[x], d[y])
        rho, _ = stats.corr(d[x], d[y], "spearman")
        rows.append({"measure": m, "pearson": r, "spearman": rho, "distance_corr": dcor, "mutual_info": mi,
                     "mutual_info_null": mi_null, "n": len(d)})
        findings.append(finding(f"{m} distance correlation", m, "distance_corr", dcor, len(d)))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: Pearson {fmt(r.pearson)}, Spearman {fmt(r.spearman)}, distance corr {fmt(r.distance_corr)}"
                                 for r in t.itertuples()),
            "findings": findings, "tables": {"dependence": records(t)},
            "notes": ["Distance correlation well above |Pearson| suggests a non-linear or non-monotonic relationship."]}
