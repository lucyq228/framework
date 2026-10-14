"""Segmentation & heterogeneity: segment-correlation, segment-difference-test, simpsons-paradox-check, min-sample-guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from corr_core import charts, features, stats
from corr_core.stats import finding, records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, fmt, frame, grain, measures, min_n, xy

CSAT_CHANNEL_SEGMENT = "csat_channel"


def requested_segments(tc: ToolContext, params: dict, f: pd.DataFrame) -> tuple[list[str], list[str]]:
    wanted = params.get("segments") or (tc.data.segments + ([CSAT_CHANNEL_SEGMENT] if grain(tc, params) == "visit" else []))
    ok = [s for s in wanted if s == CSAT_CHANNEL_SEGMENT or (s in f.columns and f[s].notna().any())]
    return ok, [s for s in wanted if s not in ok]


def _level_rows(f, x, y, seg, level_col, floor, m):
    rows = []
    for level, sub in f.groupby(level_col, dropna=True):
        d = stats.clean_xy(sub, x, y)
        r, p = stats.corr(d[x], d[y]) if len(d) >= floor else (np.nan, np.nan)
        ci = stats.fisher_ci(r, len(d))
        rows.append({"measure": m, "segment": seg, "level": str(level), "r": r, "ci_low": ci[0], "ci_high": ci[1],
                     "p": p, "n": len(d), "suppressed": len(d) < floor,
                     "pace_mean": d[x].mean(), "csat_mean": d[y].mean()})
    return rows


@tool("segment-correlation")
def segment_correlation(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    segs, unavailable = requested_segments(tc, params, f)
    floor = min_n(tc, params)
    rows = []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        for seg in segs:
            if seg == CSAT_CHANNEL_SEGMENT:
                keys = [k for k in features.CSAT_CHANNELS if k != "all"]
                for key in keys + (["matched"] if grain(tc, params) == "visit" else []):
                    fc = frame(tc, dict(params, csat_channel=key))
                    rows += [dict(r, level=key) for r in _level_rows(fc.assign(_all="x"), x, y, seg, "_all", floor, m)]
            else:
                rows += _level_rows(f, x, y, seg, seg, floor, m)
    t = pd.DataFrame(rows)
    findings = [finding(f"{r.measure} {r.segment}={r.level}", r.measure, "pearson_r", r.r, r.n, (r.ci_low, r.ci_high), r.p,
                        segment=r.segment, level=r.level) for r in t.itertuples() if not r.suppressed]
    chart_paths = []
    for m in measures(params):
        sub = t[(t["measure"] == m) & ~t["suppressed"]].dropna(subset=["r"])
        if len(sub):
            chart_paths.append(charts.forest(tc.chart_path(f"segments_{m}"), (sub["segment"] + ": " + sub["level"]).tolist(),
                                             sub["r"].tolist(), sub["ci_low"].tolist(), sub["ci_high"].tolist(),
                                             f"{m}: PACE-CSAT correlation by segment"))
    notes = [f"Segments not available with current data: {unavailable}. Acquire restaurant_attributes to unlock them."] if unavailable else []
    sup = t[t["suppressed"]]
    if len(sup):
        notes.append(f"{len(sup)} segment levels suppressed below n={floor}.")
    return {"summary": f"{len(t) - len(sup)} segment levels estimated across {len(segs)} dimensions; "
                       f"range of r: {fmt(t['r'].min())} to {fmt(t['r'].max())}",
            "findings": findings, "tables": {"segments": records(t)}, "charts": chart_paths, "notes": notes,
            "unavailable_segments": unavailable, **({"status": "partial"} if unavailable else {})}


@tool("segment-difference-test")
def segment_difference_test(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    segs, unavailable = requested_segments(tc, params, f)
    segs = [s for s in segs if s != CSAT_CHANNEL_SEGMENT]
    floor = min_n(tc, params)
    rows, findings = [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        for seg in segs:
            d = stats.clean_xy(f, x, y, [seg]).dropna(subset=[seg])
            counts = d[seg].value_counts()
            d = d[d[seg].isin(counts[counts >= floor].index)]
            if d[seg].nunique() < 2:
                continue
            d = d.assign(_seg=d[seg].astype(str))
            full = smf.ols(f"{y} ~ {x} * C(_seg)", d).fit()
            reduced = smf.ols(f"{y} ~ {x} + C(_seg)", d).fit()
            ftest = full.compare_f_test(reduced)
            rows.append({"measure": m, "segment": seg, "levels": int(d[seg].nunique()), "interaction_F": ftest[0],
                         "interaction_p": ftest[1], "n": len(d)})
            findings.append(finding(f"{m} slope differs by {seg}", m, "interaction_F", ftest[0], len(d), p=ftest[1], segment=seg))
            for level in d["_seg"].unique():
                a, b = d[d["_seg"] == level], d[d["_seg"] != level]
                r1, _ = stats.corr(a[x], a[y])
                r2, _ = stats.corr(b[x], b[y])
                z, p = stats.fisher_diff(r1, len(a), r2, len(b))
                rows.append({"measure": m, "segment": seg, "level": level, "r_level": r1, "r_rest": r2, "fisher_z": z,
                             "p": p, "n": len(a)})
    t = pd.DataFrame(rows)
    inter = t[t.get("interaction_p").notna()] if "interaction_p" in t else t.iloc[0:0]
    return {"summary": "Interaction tests: " + "; ".join(f"{r.measure}/{r.segment} p={r.interaction_p:.3f}" for r in inter.itertuples()),
            "findings": findings, "tables": {"differences": records(t)},
            "notes": [f"Unavailable segments: {unavailable}"] if unavailable else []}


@tool("simpsons-paradox-check")
def simpsons_paradox_check(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    group = params.get("group_by", "market_group")
    rows, findings, flags, chart_paths = [], [], [], []
    for m in measures(params):
        x, y, _ = xy(tc, params, m)
        d = stats.clean_xy(f, x, y, [group]).dropna(subset=[group])
        rp, pp = stats.corr(d[x], d[y])
        within = []
        for level, sub in d.groupby(group):
            r, p = stats.corr(sub[x], sub[y])
            within.append((level, r, len(sub)))
            rows.append({"measure": m, "level": level, "r": r, "n": len(sub), "pace_mean": sub[x].mean(), "csat_mean": sub[y].mean()})
        dx = d[x] - d.groupby(group)[x].transform("mean")
        dy = d[y] - d.groupby(group)[y].transform("mean")
        rw, pw = stats.corr(dx, dy)
        rows.append({"measure": m, "level": "POOLED", "r": rp, "n": len(d), "pace_mean": d[x].mean(), "csat_mean": d[y].mean()})
        rows.append({"measure": m, "level": f"WITHIN {group} (demeaned)", "r": rw, "n": len(d)})
        sign_flip = any(np.sign(r) != np.sign(rp) for _, r, _ in within if np.isfinite(r))
        gap = rw - rp
        flagged = sign_flip or abs(gap) >= float(params.get("flag_gap", .1))
        if flagged:
            flags.append(f"{m}: pooled r={rp:.2f} vs within-{group} r={rw:.2f}" + (" (sign flip)" if sign_flip else ""))
        findings += [finding(f"{m} pooled", m, "pearson_r", rp, len(d), stats.fisher_ci(rp, len(d)), pp),
                     finding(f"{m} within {group}", m, "pearson_r", rw, len(d), stats.fisher_ci(rw, len(d)), pw,
                             confound_gap=round(float(gap), 4), flagged=bool(flagged))]
        groups = {str(level): (sub[x].tolist(), sub[y].tolist()) for level, sub in d.groupby(group)}
        chart_paths.append(charts.scatter(tc.chart_path(f"simpson_{m}"), groups, f"{m}: PACE vs CSAT by {group}",
                                          x, y))
    t = pd.DataFrame(rows)
    return {"summary": ("CONFOUNDING FLAG: " + "; ".join(flags)) if flags else f"No Simpson's-paradox pattern by {group}",
            "flags": flags, "findings": findings, "tables": {"simpson": records(t)}, "charts": chart_paths,
            "notes": [f"When groups differ in average PACE and CSAT, pooled correlations mix the group difference with the "
                      f"within-group relationship. Report within-{group} results as the primary estimate."]}


@tool("min-sample-guard")
def min_sample_guard(tc: ToolContext, params: dict) -> dict:
    f = frame(tc, params)
    floor = min_n(tc, params)
    segs, unavailable = requested_segments(tc, params, f)
    rows = []
    for seg in [s for s in segs if s != CSAT_CHANNEL_SEGMENT]:
        counts = f.groupby(seg).agg(rows=("glin", "size"), restaurants=("glin", "nunique"))
        for level, r in counts.iterrows():
            rows.append({"segment": seg, "level": str(level), "rows": int(r["rows"]), "restaurants": int(r["restaurants"]),
                         "supported": bool(r["rows"] >= floor)})
    t = pd.DataFrame(rows)
    blocked = t[~t["supported"]] if len(t) else t
    return {"summary": f"{len(t) - len(blocked)} of {len(t)} segment levels meet n>={floor} at {grain(tc, params)} grain",
            "tables": {"sample_sizes": records(t)}, "suppressed_levels": records(blocked),
            "notes": [f"Unavailable segments: {unavailable}"] if unavailable else []}
