"""Consistency analysis: quadrant-agreement, rank-agreement, rgrv-cfv-gap, scoring-variant-comparison, bland-altman."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score

from corr_core import stats
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, fmt, frame, measures, pct, xy

GROUP = "market_group"


def _restaurant(tc, params):
    return frame(tc, dict(params, grain="restaurant_period"))


@tool("quadrant-agreement")
def quadrant_agreement(tc: ToolContext, params: dict) -> dict:
    f = _restaurant(tc, params)
    rows, findings, detail = [], [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, ["glin", "period", GROUP] + [c for c in ("restaurant_type", "volume_band") if c in f.columns])
        hx = d[x] > d.groupby(GROUP)[x].transform("median")
        hy = d[y] > d.groupby(GROUP)[y].transform("median")
        quad = np.select([hx & hy, ~hx & ~hy, hx & ~hy], ["aligned high", "aligned low", "high PACE / low CSAT"], "low PACE / high CSAT")
        d = d.assign(quadrant=quad)
        share = d["quadrant"].value_counts(normalize=True)
        aligned = share.get("aligned high", 0) + share.get("aligned low", 0)
        phi, p = stats.corr(hx.astype(float), hy.astype(float))
        rows += [{"measure": m, "quadrant": q, "share": s, "restaurants": int((d["quadrant"] == q).sum())} for q, s in share.items()]
        findings.append(finding(f"{m} quadrant agreement (phi)", m, "phi", phi, len(d), stats.fisher_ci(phi, len(d)), p,
                                aligned_share=round(float(aligned), 4)))
        detail.append(d.assign(measure=m)[["measure", "glin", "period", GROUP, x, y, "quadrant"]
                                           + [c for c in ("restaurant_type", "volume_band") if c in d.columns]]
                      .rename(columns={x: "pace", y: "csat"}))
        if "restaurant_type" in d.columns:
            mix = pd.crosstab(d["restaurant_type"], d["quadrant"], normalize="index").reset_index()
            rows += [{"measure": m, "quadrant": "type mix", **r} for r in mix.to_dict("records")]
    out = pd.concat(detail)
    path = tc.run_dir / "results" / f"{tc.label}__restaurants.csv"
    out.to_csv(path, index=False)
    t = pd.DataFrame(rows)
    al = {fi["measure"]: fi["aligned_share"] for fi in findings}
    return {"summary": "Share of restaurant-periods on the same side of the market median for both: "
                       + "; ".join(f"{k} {pct(v)}" for k, v in al.items()) + " (50% = no agreement)",
            "findings": findings, "tables": {"quadrants": records(t)}, "restaurant_detail_csv": path.name,
            "notes": ["Divergent quadrants are where execution and perception disagree; the restaurant list is for "
                      "follow-up, not for ranking individual restaurants."]}


@tool("rank-agreement")
def rank_agreement(tc: ToolContext, params: dict) -> dict:
    f = _restaurant(tc, params)
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, [GROUP])
        rx = d.groupby(GROUP)[x].rank(pct=True)
        ry = d.groupby(GROUP)[y].rank(pct=True)
        tau, p = sps.kendalltau(rx, ry)
        top = ((rx > .8) & (ry > .8)).sum() / max((rx > .8).sum(), 1)
        bottom = ((rx <= .2) & (ry <= .2)).sum() / max((rx <= .2).sum(), 1)
        tx = pd.cut(rx, [0, 1 / 3, 2 / 3, 1], labels=False, include_lowest=True)
        ty = pd.cut(ry, [0, 1 / 3, 2 / 3, 1], labels=False, include_lowest=True)
        kappa = cohen_kappa_score(tx, ty)
        rows.append({"measure": m, "kendall_tau": tau, "p": p, "top_quintile_overlap": top, "bottom_quintile_overlap": bottom,
                     "chance_overlap": .2, "tertile_kappa": kappa, "n": len(d)})
        findings += [finding(f"{m} Kendall tau within {GROUP}", m, "kendall_tau", tau, len(d), p=p),
                     finding(f"{m} tertile kappa", m, "kappa", kappa, len(d))]
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}: top-quintile overlap {pct(r.top_quintile_overlap)} vs 20% chance, "
                                 f"bottom {pct(r.bottom_quintile_overlap)}, tertile kappa {fmt(r.tertile_kappa)}" for r in t.itertuples()),
            "findings": findings, "tables": {"rank_agreement": records(t)}}


@tool("rgrv-cfv-gap")
def rgrv_cfv_gap(tc: ToolContext, params: dict) -> dict:
    rows, findings = [], []
    rp = _restaurant(tc, params)
    for m in measures(params):
        short = "taste" if m == "taste" else "hosp"
        _, y, _ = xy(tc, params, m)
        d = rp.dropna(subset=[f"rgr_{short}_prog", f"cfv_{short}_prog", y])
        gap = d[f"rgr_{short}_prog"] - d[f"cfv_{short}_prog"]
        t_stat, p_gap = sps.ttest_rel(d[f"rgr_{short}_prog"], d[f"cfv_{short}_prog"]) if len(d) > 3 else (np.nan, np.nan)
        r_rgr, p_rgr = stats.corr(d[f"rgr_{short}_prog"], d[y])
        r_cfv, p_cfv = stats.corr(d[f"cfv_{short}_prog"], d[y])
        r_gap, _ = stats.corr(gap, d[y])
        rows.append({"measure": m, "restaurant_periods": len(d), "mean_gap_rgr_minus_cfv": gap.mean(), "sd_gap": gap.std(),
                     "share_rgr_higher": (gap > 0).mean(), "gap_p": p_gap, "r_rgr_vs_csat": r_rgr, "r_cfv_vs_csat": r_cfv,
                     "r_gap_vs_csat": r_gap})
        findings += [finding(f"{m} RGRV-CFV gap", m, "mean_gap", gap.mean(), len(d), p=p_gap),
                     finding(f"{m} RGRV vs CSAT", m, "pearson_r", r_rgr, len(d), stats.fisher_ci(r_rgr, len(d)), p_rgr),
                     finding(f"{m} CFV vs CSAT", m, "pearson_r", r_cfv, len(d), stats.fisher_ci(r_cfv, len(d)), p_cfv)]
        vf = frame(tc, dict(params, grain="visit"))
        for vt in ("RGR", "CFV"):
            x, _, _ = xy(tc, params, m)
            dv = stats.clean_xy(vf[vf["survey_name"] == vt], x, y)
            r, p = stats.corr(dv[x], dv[y])
            rows.append({"measure": m, "visit_level": vt, "r_visit_vs_window_csat": r, "n": len(dv)})
            findings.append(finding(f"{m} visit-level {vt} vs window CSAT", m, "pearson_r", r, len(dv), stats.fisher_ci(r, len(dv)), p))
    t = pd.DataFrame(rows)
    main = t[t["restaurant_periods"].notna()] if "restaurant_periods" in t else t
    return {"summary": "; ".join(f"{r.measure}: RGRV exceeds CFV by {r.mean_gap_rgr_minus_cfv:+.1%} on average; r with CSAT "
                                 f"RGRV {fmt(r.r_rgr_vs_csat)} vs CFV {fmt(r.r_cfv_vs_csat)}" for r in main.itertuples()),
            "findings": findings, "tables": {"rgrv_cfv": records(t)},
            "notes": ["RGRV is announced (tests capability); CFV is unannounced (tests the everyday experience). "
                      "The two r values share the same CSAT, so their difference is indicative, not a formal test."]}


@tool("scoring-variant-comparison")
def scoring_variant_comparison(tc: ToolContext, params: dict) -> dict:
    rows, findings = [], []
    for g in ("visit", "restaurant_period"):
        f = frame(tc, dict(params, grain=g))
        for m in measures(params):
            _, y, _ = xy(tc, params, m)
            for s in ("progressive", "conservative"):
                x, _, _ = xy(tc, params, m, s)
                d = stats.clean_xy(f, x, y)
                r, p = stats.corr(d[x], d[y])
                mode_share = d[x].round(4).value_counts(normalize=True).iloc[0]
                rows.append({"grain": g, "measure": m, "scoring": s, "mean": d[x].mean(), "sd": d[x].std(),
                             "distinct_values": d[x].round(4).nunique(), "share_at_mode": mode_share,
                             "r_with_csat": r, "p": p, "n": len(d)})
                findings.append(finding(f"{m} {s} at {g}", m, "pearson_r", r, len(d), stats.fisher_ci(r, len(d)), p,
                                        scoring=s, grain=g))
            px, cx = xy(tc, params, m, "progressive")[0], xy(tc, params, m, "conservative")[0]
            rho, _ = stats.corr(f[px].dropna(), f.loc[f[px].notna(), cx], "spearman")
            rows.append({"grain": g, "measure": m, "scoring": "progressive vs conservative rank agreement", "r_with_csat": rho})
    t = pd.DataFrame(rows)
    rp = t[(t["grain"] == "restaurant_period") & t["scoring"].isin(["progressive", "conservative"])]
    return {"summary": "Restaurant level: " + "; ".join(f"{r.measure} {r.scoring} sd={r.sd:.3f}, distinct={r.distinct_values}, "
                                                        f"r={fmt(r.r_with_csat)}" for r in rp.itertuples()),
            "findings": findings, "tables": {"variants": records(t)},
            "notes": ["Conservative scoring is pass/fail per visit; with few visits it produces many ties, which limits "
                      "how well it can differentiate restaurants."]}


@tool("bland-altman")
def bland_altman(tc: ToolContext, params: dict) -> dict:
    f = _restaurant(tc, params)
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, [GROUP])
        for level, sub in [("all", d)] + list(d.groupby(GROUP)):
            diff = sub[x] - sub[y]
            mean = (sub[x] + sub[y]) / 2
            r_prop, p_prop = stats.corr(mean, diff)
            rows.append({"measure": m, "group": level, "mean_diff_pace_minus_csat": diff.mean(), "sd_diff": diff.std(),
                         "loa_low": diff.mean() - 1.96 * diff.std(), "loa_high": diff.mean() + 1.96 * diff.std(),
                         "proportional_bias_r": r_prop, "n": len(sub)})
            findings.append(finding(f"{m} PACE-CSAT offset {level}", m, "mean_diff", diff.mean(), len(sub)))
    t = pd.DataFrame(rows)
    return {"summary": "; ".join(f"{r.measure}/{r.group}: PACE - CSAT = {r.mean_diff_pace_minus_csat:+.1%} "
                                 f"(limits {r.loa_low:+.0%} to {r.loa_high:+.0%})" for r in t.itertuples()),
            "findings": findings, "tables": {"bland_altman": records(t)},
            "notes": ["A systematic offset matters for the GSE KPI: with a 50/50 weight, the side with the higher level "
                      "and larger spread dominates the index."]}
