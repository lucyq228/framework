"""Analysis datasets for one run, and the frame accessor every shelf tool uses.

build_all() writes runs/<run>/datasets/*.parquet. Ctx.frame() turns them into
an analysis frame with standard columns:

    x columns   taste_prog, taste_cons, hosp_prog, hosp_cons (+ component columns)
    y columns   csat_taste, csat_hosp, with response counts csat_taste_n, csat_hosp_n
    keys        glin, market, market_group (+ restaurant attributes when acquired)
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from corr_core import features, sources

MEASURE_COLS = {"taste": ("taste", "csat_taste"), "hospitality": ("hosp", "csat_hosp")}


def x_col(measure: str, scoring: str = "progressive") -> str:
    return f"{MEASURE_COLS[measure][0]}_{'prog' if scoring.startswith('prog') else 'cons'}"


def y_col(measure: str) -> str:
    return MEASURE_COLS[measure][1]


def _rates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["csat_taste_n"] = frame["taste_n"]
    frame["csat_hosp_n"] = frame["hosp_n"]
    frame["csat_taste"] = np.where(frame["taste_n"] > 0, frame["taste_hi"] / frame["taste_n"].where(frame["taste_n"] > 0), np.nan)
    frame["csat_hosp"] = np.where(frame["hosp_n"] > 0, frame["hosp_hi"] / frame["hosp_n"].where(frame["hosp_n"] > 0), np.nan)
    return frame.drop(columns=list(features.MEASURES))


def build_all(run_dir: Path, scope: dict) -> dict:
    out_dir = run_dir / "datasets"
    out_dir.mkdir(exist_ok=True)
    decisions = scope["decisions"]
    visits_all, scoring_report = features.build_visit_scores(decisions)
    visits, filter_report = features.filter_scope(visits_all, scope)
    csat, csat_report = features.clean_csat()
    cube = features.CsatCube(csat)
    windows = features.align_time_windows(cube, visits)
    start, end = pd.Timestamp(scope["period"]["start"]), pd.Timestamp(scope["period"]["end"])
    rp_csat = pd.concat([features.restaurant_period_csat(cube, p, start, end).assign(period_type=p)
                         for p in features.PERIODS], ignore_index=True)
    monthly = features.csat_monthly(csat)
    monthly = monthly.merge(sources.restaurant_dim()[["glin", "market", "market_group"]], on="glin", how="left")
    scorecard = features.market_scorecard(visits, csat, scope)

    visits.to_parquet(out_dir / "visit_scores.parquet", index=False)
    windows.to_parquet(out_dir / "csat_windows.parquet", index=False)
    rp_csat.to_parquet(out_dir / "restaurant_period_csat.parquet", index=False)
    monthly.to_parquet(out_dir / "csat_monthly.parquet", index=False)
    scorecard.to_parquet(out_dir / "market_scorecard.parquet", index=False)

    primary = scope.get("primary_csat_window", "pm30")
    w = windows[(windows["window"] == primary) & (windows["csat_channel"] == "all")]
    matched = w.set_index("visit_id")["hosp_n"].reindex(visits["visit_id"]).fillna(0)
    ctx = Ctx(run_dir, scope)
    rp = ctx.restaurant_frame()
    rp = features.gse_index(rp.dropna(subset=["csat_taste", "csat_hosp"]))
    rp.to_parquet(out_dir / "restaurant_period.parquet", index=False)
    return {
        "summary": (f"{len(visits)} in-scope visits across {visits['glin'].nunique()} restaurants; "
                    f"{(matched > 0).mean():.1%} have CSAT in the {primary} window; "
                    f"{len(rp)} restaurant-{scope.get('restaurant_period', 'year')} rows"),
        "scoring": scoring_report, "scope_filter": filter_report, "csat_cleaning": csat_report,
        "visits_by_market_type": visits.groupby(["market", "survey_name"]).size().rename("visits").reset_index().to_dict("records"),
        "csat_match_rate_primary_window": round(float((matched > 0).mean()), 4),
        "median_csat_responses_primary_window": float(matched.median()),
        "restaurant_period_rows": int(len(rp)),
        "segment_columns": sources.segment_columns(),
        "files": sorted(p.name for p in out_dir.glob("*.parquet")),
        "market_scorecard": scorecard.round(4).to_dict("records"),
    }


class Ctx:
    """Loads a run's datasets lazily and hands tools ready-made analysis frames."""

    def __init__(self, run_dir: Path, scope: dict):
        self.run_dir = run_dir
        self.scope = scope
        self.dir = run_dir / "datasets"

    def _load(self, name: str) -> pd.DataFrame:
        path = self.dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Dataset {name} is missing. The analysis_datasets milestone must run first.")
        return pd.read_parquet(path)

    @cached_property
    def visits(self) -> pd.DataFrame:
        return self._load("visit_scores")

    @cached_property
    def windows(self) -> pd.DataFrame:
        return self._load("csat_windows")

    @cached_property
    def rp_csat(self) -> pd.DataFrame:
        return self._load("restaurant_period_csat")

    @cached_property
    def monthly(self) -> pd.DataFrame:
        return self._load("csat_monthly")

    @cached_property
    def scorecard(self) -> pd.DataFrame:
        return self._load("market_scorecard")

    @property
    def segments(self) -> list[str]:
        cols = [c for c in sources.segment_columns() if c in self.visits.columns]
        return cols + ["survey_name", "channel_tested"]

    def visit_frame(self, window: str | None = None, csat_channel: str = "all",
                    visit_types: list[str] | None = None) -> pd.DataFrame:
        """One row per visit with CSAT from the chosen window. csat_channel='matched' pairs each
        CFV with CSAT from the channel it tested (Drive Thru, In Restaurant, Curbside)."""
        window = window or self.scope.get("primary_csat_window", "pm30")
        v = self.visits
        if visit_types:
            v = v[v["survey_name"].isin(visit_types)]
        w = self.windows[self.windows["window"] == window]
        if csat_channel == "matched":
            v = v[v["channel_tested"].isin(features.MATCHED_CSAT)].copy()
            v["csat_channel"] = v["channel_tested"].map(features.MATCHED_CSAT)
            frame = v.merge(w, on=["visit_id", "csat_channel"], how="left")
        else:
            frame = v.merge(w[w["csat_channel"] == csat_channel], on="visit_id", how="left")
        return _rates(frame.drop(columns=["window"]))

    def restaurant_frame(self, period: str | None = None, csat_channel: str = "all",
                         visit_types: list[str] | None = None) -> pd.DataFrame:
        """One row per restaurant x period: PACE restaurant result (progressive = mean visit score,
        conservative = visits passed / eligible visits) and CSAT over the same period."""
        period = period or self.scope.get("restaurant_period", "year")
        v = self.visits
        if visit_types:
            v = v[v["survey_name"].isin(visit_types)]
        v = v.assign(period=features.period_label(v["visit_date"], period))
        comp = [c for c in v.columns if c.startswith(("cat_", "ch_", "tp_", "fail_"))] + ["general_service_rate"]
        agg = {"taste_prog": "mean", "taste_cons": "mean", "hosp_prog": "mean", "hosp_cons": "mean",
               "visit_id": "count", **{c: "mean" for c in comp if c in v.columns}}
        dims = [c for c in sources.segment_columns() if c in v.columns]
        rp = v.groupby(["glin", "period"] + dims, dropna=False).agg(agg).rename(columns={"visit_id": "n_visits"}).reset_index()
        for name, types in (("cfv", ["CFV"]), ("rgr", ["RGR"])):
            sub = v[v["survey_name"].isin(types)].groupby(["glin", "period"])[["taste_prog", "hosp_prog"]].mean()
            sub.columns = [f"{name}_{c}" for c in sub.columns]
            rp = rp.merge(sub, left_on=["glin", "period"], right_index=True, how="left")
        c = self.rp_csat[(self.rp_csat["period_type"] == period) & (self.rp_csat["csat_channel"] == csat_channel)]
        rp = rp.merge(c.drop(columns=["period_type", "csat_channel"]), on=["glin", "period"], how="left")
        return _rates(rp)

    def frame(self, grain: str = "visit", **params) -> pd.DataFrame:
        keys = {"window", "csat_channel", "visit_types"} if grain == "visit" else {"period", "csat_channel", "visit_types"}
        args = {k: v for k, v in params.items() if k in keys}
        return self.visit_frame(**args) if grain == "visit" else self.restaurant_frame(**args)
