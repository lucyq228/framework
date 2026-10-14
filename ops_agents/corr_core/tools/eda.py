"""EDA tools. Two modes:

- baseline (no approved scope yet): all markets and dates, default scoring decisions.
- descriptive (locked scope): the run's scored, in-scope datasets.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from corr_core import charts, features, scoring, sources
from corr_core.scope import DEFAULT_DECISIONS
from corr_core.stats import records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, pct


@lru_cache(maxsize=2)
def _baseline_visits() -> pd.DataFrame:
    visits, _ = features.build_visit_scores(DEFAULT_DECISIONS)
    return visits


def _inputs(tc: ToolContext) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    csat, _ = features.clean_csat()
    csat = csat.assign(market=csat["terr_cd"].map(sources.MARKET_BY_TERR))
    if tc.scope and tc.data is not None:
        start, end = pd.Timestamp(tc.scope["period"]["start"]), pd.Timestamp(tc.scope["period"]["end"])
        csat = csat[csat["market"].isin(tc.scope["markets"]) & (csat["cal_dt"] >= start) & (csat["cal_dt"] <= end)]
        return tc.data.visits, csat, tc.scope["decisions"], "descriptive"
    return _baseline_visits(), csat, DEFAULT_DECISIONS, "baseline"


def _rate(df: pd.DataFrame, by: list[str], hi: str, n: str) -> pd.DataFrame:
    g = df.groupby(by)[[hi, n]].sum().reset_index()
    g["rate"] = g[hi] / g[n].where(g[n] > 0)
    return g.rename(columns={n: "responses"}).drop(columns=[hi])


@tool("eda-coverage")
def eda_coverage(tc: ToolContext, params: dict) -> dict:
    visits, csat, _, mode = _inputs(tc)
    rest = visits.groupby("market")["glin"].nunique().rename("restaurants_with_pace")
    by_type = visits.groupby(["market", "visit_year", "survey_name"]).size().unstack(fill_value=0)
    per_rest = (visits.groupby(["glin", "visit_year", "survey_name"]).size().unstack(fill_value=0)
                .groupby(level="visit_year").agg(["mean", "min", "max"]))
    per_rest.columns = [f"{a}_{b}" for a, b in per_rest.columns]
    c = csat.assign(month=csat["cal_dt"].dt.to_period("M").astype(str))
    rm = c.groupby(["market", "glin", "month"])["frnd_ansr_cnt_qt"].sum().reset_index()
    resp = rm.groupby("market")["frnd_ansr_cnt_qt"].describe(percentiles=[.1, .5, .9])[["10%", "50%", "90%"]]
    floor = int(params.get("min_csat_responses", 30))
    thin = rm.assign(thin=rm["frnd_ansr_cnt_qt"] < floor).groupby("market")["thin"].mean().rename("share_below_floor")
    both = len(set(visits["glin"]) & set(csat["glin"]))
    mean_visits = visits.groupby(["glin", "visit_year"]).size().groupby(level="visit_year").mean()
    return {
        "summary": (f"[{mode}] {visits['glin'].nunique()} restaurants with PACE, {len(visits)} visits; "
                    f"{both} restaurants have both PACE and CSAT; mean visits per restaurant per year "
                    + ", ".join(f"{y}: {v:.1f}" for y, v in mean_visits.items())),
        "mode": mode,
        "tables": {
            "restaurants_by_market": records(rest.reset_index()),
            "visits_by_market_year_type": records(by_type.reset_index()),
            "visits_per_restaurant_per_year": records(per_rest.reset_index()),
            "csat_hosp_responses_per_restaurant_month": records(resp.join(thin).reset_index()),
        },
        "restaurants_with_both_sources": both,
    }


@tool("eda-pace-pass-rates")
def eda_pace_pass_rates(tc: ToolContext, params: dict) -> dict:
    visits, _, decisions, mode = _inputs(tc)
    q = scoring.question_level(decisions)
    prod, serv = q["product"], q["service"]
    vids = set(visits["visit_id"])
    prod = prod[prod["visit_id"].isin(vids) & (prod["question_exception_flag"] >= 0)]
    serv = serv[serv["visit_id"].isin(vids) & (serv["question_exception_flag"] >= 0)]
    prod = prod.assign(passed=(prod["question_exception_flag"] == 0).astype(float))
    serv = serv.assign(passed=(serv["question_exception_flag"] == 0).astype(float))
    by_cat = prod.groupby("category")["passed"].agg(["mean", "size"]).rename(columns={"mean": "pass_rate", "size": "checks"})
    by_prod = prod.groupby(["category", "product_name"])["passed"].agg(["mean", "size"]).rename(
        columns={"mean": "pass_rate", "size": "checks"}).reset_index()
    by_ch = serv.groupby("channel")["passed"].agg(["mean", "size"]).rename(columns={"mean": "question_pass_rate", "size": "questions"})
    by_tp = serv.groupby("touchpoint")["passed"].agg(["mean", "size"]).rename(columns={"mean": "question_pass_rate", "size": "questions"})
    by_q = serv.groupby(["channel", "question_code"])["passed"].agg(["mean", "size"]).rename(
        columns={"mean": "pass_rate", "size": "questions"}).reset_index()
    cols = ["taste_prog", "taste_cons", "hosp_prog", "hosp_cons"]
    visit_level = visits.groupby(["market", "survey_name"])[cols].mean().reset_index()
    overall = visits.groupby("survey_name")[cols].mean()
    dist = {c: visits[c].round(2).value_counts(normalize=True).sort_index().round(4).to_dict() for c in ("taste_prog", "hosp_prog")}
    ch_pass = {c.replace("ch_", "").replace("_pass", ""): float(visits[c].mean()) for c in visits.columns if c.startswith("ch_")}
    chart_cat = charts.bar(tc.chart_path("taste_category_pass"), by_cat.index.tolist(), by_cat["pass_rate"].tolist(),
                           "PACE product pass rate by category")
    chart_ch = charts.bar(tc.chart_path("channel_journey_pass"), list(ch_pass), list(ch_pass.values()),
                          "PACE channel journey pass rate")
    weakest = by_cat.loc[by_cat.index.isin(sources.TASTE_INDEX_CODES.values()), "pass_rate"].idxmin()
    return {
        "summary": (f"[{mode}] Visit-level progressive Taste {pct(visits['taste_prog'].mean())}, Hospitality "
                    f"{pct(visits['hosp_prog'].mean())}; conservative Taste {pct(visits['taste_cons'].mean())}, "
                    f"Hospitality {pct(visits['hosp_cons'].mean())}; RGRV vs CFV Hospitality "
                    f"{pct(overall.loc['RGR', 'hosp_prog'] if 'RGR' in overall.index else np.nan)} vs "
                    f"{pct(overall.loc['CFV', 'hosp_prog'] if 'CFV' in overall.index else np.nan)}; weakest index category {weakest}"),
        "mode": mode,
        "tables": {"pass_rate_by_category": records(by_cat.reset_index()),
                   "pass_rate_by_product": records(by_prod),
                   "question_pass_rate_by_channel": records(by_ch.reset_index()),
                   "question_pass_rate_by_touchpoint": records(by_tp.reset_index()),
                   "pass_rate_by_service_question": records(by_q),
                   "visit_scores_by_market_type": records(visit_level),
                   "channel_journey_pass_rate": [{"channel": k, "pass_rate": round(v, 4)} for k, v in ch_pass.items()]},
        "score_distributions": dist,
        "charts": [chart_cat, chart_ch],
    }


@tool("eda-failure-reasons")
def eda_failure_reasons(tc: ToolContext, params: dict) -> dict:
    visits, _, decisions, mode = _inputs(tc)
    q = scoring.question_level(decisions)
    vids = set(visits["visit_id"])
    prod = q["product"][q["product"]["visit_id"].isin(vids) & (q["product"]["question_exception_flag"] == 1)]
    serv = q["service"][q["service"]["visit_id"].isin(vids) & (q["service"]["question_exception_flag"] == 1)]
    dims = pd.crosstab(prod["category"], prod["fail_dimension"])
    serv = serv.assign(reason=serv["reasons"].str.strip('[]"'))
    top = serv.groupby(["touchpoint", "reason"]).size().rename("failures").reset_index().sort_values("failures", ascending=False)
    total_dim = prod["fail_dimension"].value_counts(normalize=True)
    return {
        "summary": (f"[{mode}] {len(prod)} product failures; most common dimension "
                    f"{total_dim.index[0] if len(total_dim) else 'n/a'} ({pct(total_dim.iloc[0] if len(total_dim) else np.nan)}); "
                    f"{len(serv)} service failures; top reason '{top.iloc[0]['reason'] if len(top) else 'n/a'}'"),
        "mode": mode,
        "tables": {"product_failures_by_category_dimension": records(dims.reset_index()),
                   "service_failure_reasons": records(top.head(25))},
        "notes": ["Dimension is only recorded on failures; passing checks carry no dimension detail."],
    }


@tool("eda-csat-rates")
def eda_csat_rates(tc: ToolContext, params: dict) -> dict:
    _, csat, _, mode = _inputs(tc)
    tables = {}
    for label, by in (("by_market", ["market"]), ("by_presented_channel", ["market", "cust_surv_psnt_pos_chnl_ds"]),
                      ("by_placed_channel", ["market", "cust_surv_plcd_pos_chnl_ds"])):
        t = _rate(csat, by, "qlty_high_stsf_cnt_qt", "qlty_ansr_cnt_qt").rename(columns={"rate": "csat_taste", "responses": "taste_responses"})
        h = _rate(csat, by, "frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt").rename(columns={"rate": "csat_hosp", "responses": "hosp_responses"})
        tables[label] = records(t.merge(h, on=by))
    wd = csat.assign(weekday=csat["cal_dt"].dt.day_name())
    h = _rate(wd, ["weekday"], "frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt").rename(columns={"rate": "csat_hosp"})
    tables["hosp_by_weekday"] = records(h)
    mk = pd.DataFrame(tables["by_market"])
    chart = charts.bar(tc.chart_path("csat_by_market"), mk["market"].tolist() * 2,
                       mk["csat_taste"].tolist() + mk["csat_hosp"].tolist(), "CSAT top-two-box: Taste (left) and Hospitality (right) by market")
    return {"summary": f"[{mode}] CSAT by market: " + "; ".join(
                f"{r['market']} Taste {pct(r['csat_taste'])} / Hospitality {pct(r['csat_hosp'])}" for r in tables["by_market"]),
            "mode": mode, "tables": tables, "charts": [chart],
            "total_responses": int(csat["qlty_ansr_cnt_qt"].sum())}


@tool("eda-trends")
def eda_trends(tc: ToolContext, params: dict) -> dict:
    visits, csat, _, mode = _inputs(tc)
    pace = visits.groupby(["visit_month", "market_group"])[["taste_prog", "hosp_prog"]].mean().reset_index()
    c = csat.assign(month=csat["cal_dt"].dt.to_period("M").astype(str), market_group=csat["market"].map(sources.MARKET_GROUP))
    cs = c.groupby(["month", "market_group"])[["qlty_high_stsf_cnt_qt", "qlty_ansr_cnt_qt", "frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt"]].sum()
    cs = pd.DataFrame({"csat_taste": cs["qlty_high_stsf_cnt_qt"] / cs["qlty_ansr_cnt_qt"],
                       "csat_hosp": cs["frnd_high_stsf_cnt_qt"] / cs["frnd_ansr_cnt_qt"]}).reset_index()
    series = {}
    for g, sub in cs.groupby("market_group"):
        series[f"CSAT Hospitality {g}"] = (sub["month"].tolist(), sub["csat_hosp"].tolist())
    for g, sub in pace.groupby("market_group"):
        series[f"PACE Hospitality {g}"] = (sub["visit_month"].tolist(), sub["hosp_prog"].tolist())
    chart = charts.lines(tc.chart_path("monthly_trends"), series, "Monthly Hospitality: CSAT vs PACE progressive", pct=True)
    yearly = visits.groupby(["visit_year", "market_group"])[["taste_prog", "hosp_prog"]].mean().reset_index()
    return {"summary": f"[{mode}] Monthly trends for {pace['visit_month'].nunique()} months; PACE monthly means rest on "
                       f"about {visits.groupby('visit_month').size().median():.0f} visits per month, so they are noisy",
            "mode": mode, "tables": {"pace_monthly": records(pace), "csat_monthly": records(cs), "pace_yearly": records(yearly)},
            "charts": [chart]}


@tool("eda-segment-profile")
def eda_segment_profile(tc: ToolContext, params: dict) -> dict:
    if not sources.available("restaurant_attributes"):
        return {"summary": "Restaurant attributes not available: only market-level profiles are possible",
                "status": "partial", "notes": ["Acquire restaurant_attributes to profile by type, ownership, volume, region."]}
    visits, csat, _, mode = _inputs(tc)
    dims = [d for d in ("restaurant_type", "ownership_type", "volume_band", "region") if d in visits.columns]
    attrs = sources.load("restaurant_attributes")
    cs = csat.groupby("glin")[["qlty_high_stsf_cnt_qt", "qlty_ansr_cnt_qt", "frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt"]].sum()
    cs = cs.join(attrs.set_index("glin")[dims], how="left")
    tables = {}
    for d in dims:
        p = visits.groupby(d).agg(restaurants=("glin", "nunique"), visits=("visit_id", "count"),
                                  taste_prog=("taste_prog", "mean"), hosp_prog=("hosp_prog", "mean"))
        g = cs.groupby(d)[["qlty_high_stsf_cnt_qt", "qlty_ansr_cnt_qt", "frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt"]].sum()
        p["csat_taste"] = g["qlty_high_stsf_cnt_qt"] / g["qlty_ansr_cnt_qt"]
        p["csat_hosp"] = g["frnd_high_stsf_cnt_qt"] / g["frnd_ansr_cnt_qt"]
        tables[f"by_{d}"] = records(p.reset_index())
    return {"summary": f"[{mode}] Profiles by {', '.join(dims)}", "mode": mode, "tables": tables}


@tool("eda-summary-pack")
def eda_summary_pack(tc: ToolContext, params: dict) -> dict:
    visits, csat, _, mode = _inputs(tc)
    obs = []
    per_year = visits.groupby(["glin", "visit_year"]).size().groupby(level="visit_year").mean()
    obs.append(f"Each restaurant has about {per_year.mean():.1f} PACE visits per year, so restaurant scores move in coarse steps.")
    by_type = visits.groupby("survey_name")[["taste_prog", "hosp_prog"]].mean()
    if {"RGR", "CFV"} <= set(by_type.index):
        gap = by_type.loc["RGR"] - by_type.loc["CFV"]
        obs.append(f"Announced RGRV visits score higher than unannounced CFV: Taste +{gap['taste_prog']:.1%}, "
                   f"Hospitality +{gap['hosp_prog']:.1%} (progressive).")
    cons_gap = visits[["taste_prog", "taste_cons", "hosp_prog", "hosp_cons"]].mean()
    obs.append(f"Conservative scoring is much harsher: Taste {cons_gap['taste_cons']:.1%} vs {cons_gap['taste_prog']:.1%} progressive; "
               f"Hospitality {cons_gap['hosp_cons']:.1%} vs {cons_gap['hosp_prog']:.1%}.")
    m = csat.groupby("market")[["frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt"]].sum()
    m_rate = (m["frnd_high_stsf_cnt_qt"] / m["frnd_ansr_cnt_qt"]).sort_values()
    p_rate = visits.groupby("market")["hosp_prog"].mean()
    obs.append("CSAT Hospitality by market: " + ", ".join(f"{k} {v:.1%}" for k, v in m_rate.items())
               + "; PACE Hospitality by market: " + ", ".join(f"{k} {v:.1%}" for k, v in p_rate.items())
               + ". Markets that rank differently on the two sides signal a market-level confound for pooled analysis.")
    cat_cols = [c for c in visits.columns if c.startswith("cat_") and c.split("_")[1] in {k.lower() for k in sources.TASTE_INDEX_CODES.values()}]
    if cat_cols:
        cats = visits[cat_cols].mean().sort_values()
        obs.append(f"Weakest Taste category: {cats.index[0].replace('cat_', '').replace('_pass', '')} ({cats.iloc[0]:.1%} pass); "
                   f"strongest: {cats.index[-1].replace('cat_', '').replace('_pass', '')} ({cats.iloc[-1]:.1%}).")
    tp_cols = [c for c in visits.columns if c.startswith("tp_")]
    if tp_cols:
        tps = visits[tp_cols].mean().sort_values()
        obs.append(f"Weakest Hospitality touchpoint: {tps.index[0].replace('tp_', '').replace('_pass', '')} ({tps.iloc[0]:.1%} pass).")
    c = csat.assign(month=csat["cal_dt"].dt.to_period("M"))
    rm = c.groupby(["glin", "month"])["frnd_ansr_cnt_qt"].sum()
    obs.append(f"Median CSAT Hospitality responses per restaurant-month: {rm.median():.0f}; "
               f"{(rm < 30).mean():.0%} of restaurant-months have fewer than 30.")
    return {"summary": f"[{mode}] {len(obs)} general observations", "mode": mode, "observations": obs}
