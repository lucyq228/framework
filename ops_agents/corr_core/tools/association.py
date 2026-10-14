"""Pairwise association: pairwise-correlation, weighted-correlation, point-biserial,
disattenuated-correlation, permutation-test, multiple-testing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from corr_core import stats
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, corr_with_ci, fmt, frame, grain, guard, measures, min_n, xy
from corr_core.tools.distributions import csat_reliability_estimate, icc_oneway


@tool("pairwise-correlation")
def pairwise_correlation(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    boot = bool(params.get("bootstrap", True))
    findings, rows, notes = [], [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        for method in params.get("methods", stats.METHODS):
            r, p, ci, n = corr_with_ci(f, x, y, method, bootstrap=boot and method != "kendall")
            g = guard(n, min_n(tc, params))
            if g:
                notes.append(f"{m}/{method}: {g}")
                r = p = np.nan
            rows.append({"measure": m, "x": x, "y": y, "method": method, "r": r, "ci_low": ci[0], "ci_high": ci[1], "p": p, "n": n})
            findings.append(finding(f"{m} {method}", m, f"{method}_r", r, n, ci, p))
    t = pd.DataFrame(rows)
    pear = t[t["method"] == "pearson"]
    return {"summary": f"[{grain(tc, params)}, window={params.get('window') or tc.scope.get('primary_csat_window')}] "
                       + "; ".join(f"{r.measure} r={fmt(r.r)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}] n={r.n}" for r in pear.itertuples()),
            "findings": findings, "tables": {"correlations": records(t)}, "notes": notes}


@tool("weighted-correlation")
def weighted_correlation(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    findings, rows = [], []
    for m in measures(params):
        x, y, n = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, [n, "glin"])
        w = d[n] if grain(tc, params) == "visit" else d[n] * f.loc[d.index, "n_visits"] / (d[n] + f.loc[d.index, "n_visits"])
        rw = stats.weighted_corr(d[x], d[y], w)
        ru, _ = stats.corr(d[x], d[y])
        ci = stats.cluster_bootstrap(d.assign(_w=w.values), lambda s: stats.weighted_corr(s[x], s[y], s["_w"]), n_boot=300)
        rows.append({"measure": m, "unweighted_r": ru, "weighted_r": rw, "ci_low": ci[0], "ci_high": ci[1], "n": len(d)})
        findings.append(finding(f"{m} weighted", m, "weighted_r", rw, len(d), ci))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure} weighted r={fmt(r.weighted_r)} vs unweighted {fmt(r.unweighted_r)}" for r in t.itertuples()),
            "findings": findings, "tables": {"weighted": records(t)},
            "notes": ["Weights are CSAT responses (visit grain) or the harmonic combination of responses and visits (restaurant grain)."]}


@tool("point-biserial")
def point_biserial(tc: ToolContext, params: dict) -> dict:
    p2 = dict(params, grain="visit")
    f = frame(tc, p2)
    findings, rows = [], []
    for m in measures(params):
        x, y, _ = xy(tc, p2, m, "conservative")
        d = stats.clean_xy(f, x, y)
        r, p = stats.corr(d[x], d[y])
        ci = stats.fisher_ci(r, len(d))
        rows.append({"measure": m, "pass_share": d[x].mean(), "csat_if_pass": d.loc[d[x] == 1, y].mean(),
                     "csat_if_fail": d.loc[d[x] == 0, y].mean(), "r_pb": r, "p": p, "n": len(d)})
        findings.append(finding(f"{m} conservative pass vs CSAT", m, "point_biserial_r", r, len(d), ci, p))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: CSAT {r.csat_if_pass:.1%} when visit passes vs {r.csat_if_fail:.1%} when it fails "
                                 f"(r_pb={fmt(r.r_pb)})" for r in t.itertuples()),
            "findings": findings, "tables": {"point_biserial": records(t)}}


@tool("disattenuated-correlation")
def disattenuated_correlation(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    g = grain(tc, params)
    findings, rows = [], []
    for m in measures(params):
        x, y, n = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, [n, "glin"])
        r, _ = stats.corr(d[x], d[y])
        rel_y, _, _ = csat_reliability_estimate(d[y], d[n])
        icc, _ = icc_oneway(tc.data.visits[x], tc.data.visits["glin"])
        k = float(f["n_visits"].mean()) if g == "restaurant_period" else 1.0
        rel_x = k * icc / (1 + (k - 1) * icc) if np.isfinite(icc) else np.nan
        corrected = r / np.sqrt(rel_x * rel_y) if rel_x > 0 and rel_y > 0 else np.nan
        rows.append({"measure": m, "observed_r": r, "reliability_pace": rel_x, "reliability_csat": rel_y,
                     "disattenuated_r": min(corrected, 1.0) if np.isfinite(corrected) else np.nan, "n": len(d)})
        findings.append(finding(f"{m} disattenuated", m, "disattenuated_r", rows[-1]["disattenuated_r"], len(d)))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure} observed r={fmt(r.observed_r)} -> corrected {fmt(r.disattenuated_r)} "
                                 f"(PACE reliability {fmt(r.reliability_pace)}, CSAT {fmt(r.reliability_csat)})" for r in t.itertuples()),
            "findings": findings, "tables": {"disattenuation": records(t)},
            "notes": ["PACE reliability treats the restaurant's average execution as the true score (ICC with Spearman-Brown); "
                      "the corrected value is an upper bound on the restaurant-level relationship, not an observed fact."]}


@tool("permutation-test")
def permutation_test(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    n_perm = int(params.get("n_permutations", 1000))
    strata = params.get("strata", "market_group")
    rng = np.random.default_rng(27)
    findings, rows = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, [strata])
        r, _ = stats.corr(d[x], d[y])
        ys = d[y].to_numpy()
        groups = [np.flatnonzero((d[strata] == s).to_numpy()) for s in d[strata].unique()]
        xs = d[x].to_numpy()
        perm = np.empty(n_perm)
        for i in range(n_perm):
            yp = ys.copy()
            for idx in groups:
                yp[idx] = ys[rng.permutation(idx)]
            perm[i] = np.corrcoef(xs, yp)[0, 1]
        p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (n_perm + 1)
        rows.append({"measure": m, "r": r, "perm_p": p, "null_95th": np.quantile(np.abs(perm), .95), "n": len(d)})
        findings.append(finding(f"{m} permutation", m, "pearson_r", r, len(d), stats.fisher_ci(r, len(d)), p))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure} r={fmt(r.r)}, permutation p={r.perm_p:.4f} (shuffled within {strata})" for r in t.itertuples()),
            "findings": findings, "tables": {"permutation": records(t)}}


@tool("multiple-testing")
def multiple_testing(tc: ToolContext, params: dict) -> dict:
    alpha = float(params.get("alpha", .05))
    rows = []
    for path in sorted((tc.run_dir / "results").glob("*.json")):
        if path.stem == tc.label:
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        if res.get("tool") == "multiple-testing":
            continue
        for fi in res.get("findings", []):
            if fi.get("p") is not None:
                rows.append({"result": path.stem, "tool": res.get("tool"), "label": fi["label"], "metric": fi["metric"],
                             "estimate": fi.get("estimate"), "n": fi.get("n"), "p": fi["p"]})
    if not rows:
        return {"summary": "No p-values found yet; run this after the other analyses", "tables": {"fdr": []}}
    t = pd.DataFrame(rows)
    t["q_bh"] = stats.bh_fdr(t["p"].tolist())
    t["significant_after_fdr"] = t["q_bh"] < alpha
    return {"summary": f"{len(t)} tests across {t['result'].nunique()} results; {int((t['p'] < alpha).sum())} nominally significant, "
                       f"{int(t['significant_after_fdr'].sum())} survive Benjamini-Hochberg at q<{alpha}",
            "tables": {"fdr": records(t.sort_values("q_bh"))}, "alpha": alpha}
