"""Feature creation tools. build-analysis-dataset writes the run's datasets; the others report on one
step of the build so A5 can check each step."""

from __future__ import annotations

import pandas as pd

from corr_core import datasets, features, scoring
from corr_core.stats import records
from corr_core.tools import tool
from corr_core.tools.common import ToolContext, pct


@tool("build-analysis-dataset")
def build_analysis_dataset(tc: ToolContext, params: dict) -> dict:
    return datasets.build_all(tc.run_dir, tc.scope)


@tool("apply-question-mapping")
def apply_question_mapping(tc: ToolContext, params: dict) -> dict:
    decisions = tc.scope["decisions"]
    table = scoring.mapping_table(decisions)
    used = {k: v for k, v in table.items() if v}
    dropped = [k for k, v in table.items() if not v]
    return {"summary": f"{len(used)} local codes mapped to global ({', '.join(f'{k}->{v}' for k, v in used.items())}); "
                       f"{len(dropped)} excluded ({', '.join(dropped)}) under proposed_mappings="
                       f"{decisions.get('proposed_mappings')}, local_core_products={decisions.get('local_core_products')}",
            "mapping": table}


@tool("score-pace-taste")
def score_pace_taste(tc: ToolContext, params: dict) -> dict:
    v = tc.data.visits
    t = v.groupby(["market", "survey_name"]).agg(visits=("visit_id", "count"), eligible_products=("taste_eligible", "mean"),
                                                 taste_prog=("taste_prog", "mean"), taste_cons=("taste_cons", "mean")).reset_index()
    return {"summary": f"{int(v['taste_prog'].notna().sum())} visits scored; progressive {pct(v['taste_prog'].mean())}, "
                       f"conservative {pct(v['taste_cons'].mean())}",
            "tables": {"taste_by_market_type": records(t)}}


@tool("score-pace-hospitality")
def score_pace_hospitality(tc: ToolContext, params: dict) -> dict:
    v = tc.data.visits
    t = v.groupby(["market", "survey_name"]).agg(visits=("visit_id", "count"), channels=("hosp_channels", "mean"),
                                                 hosp_prog=("hosp_prog", "mean"), hosp_cons=("hosp_cons", "mean")).reset_index()
    tested = v[v["survey_name"] == "CFV"]["channel_tested"].value_counts().to_dict()
    return {"summary": f"{int(v['hosp_prog'].notna().sum())} visits scored; progressive {pct(v['hosp_prog'].mean())}, "
                       f"conservative {pct(v['hosp_cons'].mean())}; CFV channels tested {tested}",
            "tables": {"hospitality_by_market_type": records(t)}}


@tool("aggregate-csat")
def aggregate_csat(tc: ToolContext, params: dict) -> dict:
    _, report = features.clean_csat()
    mo = tc.data.monthly
    mo = mo[mo["market"].isin(tc.scope["markets"])]
    g = mo.groupby("market")[["taste_hi", "taste_n", "hosp_hi", "hosp_n"]].sum()
    g["csat_taste"] = g["taste_hi"] / g["taste_n"]
    g["csat_hosp"] = g["hosp_hi"] / g["hosp_n"]
    return {"summary": f"CSAT cleaned ({report['duplicates_removed']} duplicates and {report['invalid_rows_removed']} impossible rows "
                       f"removed); " + "; ".join(f"{k} Taste {pct(r.csat_taste)} / Hospitality {pct(r.csat_hosp)}" for k, r in g.iterrows()),
            "cleaning": report, "tables": {"csat_by_market": records(g.reset_index())}}


@tool("align-time-windows")
def align_time_windows(tc: ToolContext, params: dict) -> dict:
    w = tc.data.windows
    w = w[w["csat_channel"] == "all"]
    cov = w.groupby("window").agg(visits=("visit_id", "nunique"), median_hosp_responses=("hosp_n", "median"),
                                  share_with_csat=("hosp_n", lambda s: float((s > 0).mean()))).reindex(list(features.WINDOWS))
    return {"summary": "Median CSAT responses per visit window: " + ", ".join(f"{k} {v:.0f}" for k, v in cov["median_hosp_responses"].items()),
            "tables": {"window_coverage": records(cov.reset_index())}}


@tool("compute-gse-index")
def compute_gse_index(tc: ToolContext, params: dict) -> dict:
    sc = tc.data.scorecard
    rp = pd.read_parquet(tc.run_dir / "datasets" / "restaurant_period.parquet")
    dist = rp[["taste_index_prog", "hosp_index_prog", "gse_kpi_prog", "gse_kpi_cons"]].describe().T.reset_index()
    total = sc[sc["market"] == "All in scope"].iloc[0]
    return {"summary": f"In-scope GSE KPI {pct(total['gse_kpi_prog'])} progressive / {pct(total['gse_kpi_cons'])} conservative "
                       f"(Taste Index {pct(total['taste_index_prog'])}, Hospitality Index {pct(total['hosp_index_prog'])})",
            "tables": {"market_scorecard": records(sc), "restaurant_index_distribution": records(dist)}}
