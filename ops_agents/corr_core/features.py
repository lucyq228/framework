"""Feature creation: CSAT aggregation, time-window alignment, analysis datasets, GSE index.

CSAT is held as a dense array [restaurant, channel key, day, measure] so every
window sum is two cumulative-sum lookups. Measures: taste_hi, taste_n, hosp_hi, hosp_n.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corr_core import scoring, sources

WINDOWS = {
    # name: (offset start, offset end) in days relative to the visit date, inclusive.
    "pre7": (-7, -1), "pre30": (-30, -1), "post30": (1, 30),
    "pm7": (-7, 7), "pm14": (-14, 14), "pm30": (-30, 30), "pm60": (-60, 60), "pm90": (-90, 90),
    # calendar windows containing the visit date
    "month": "M", "quarter": "Q", "year": "Y",
}
CSAT_CHANNELS = [
    "all",
    "psnt:Drive Thru", "psnt:Front Counter", "psnt:Table Service", "psnt:Curbside", "psnt:In Restaurant",
    "plcd:Mobile", "plcd:Self Order Kiosk", "plcd:Front Counter", "plcd:Drive Thru",
]
MATCHED_CSAT = {"Drive Thru": "psnt:Drive Thru", "In Restaurant": "psnt:In Restaurant", "Curbside": "psnt:Curbside"}
MEASURES = ("taste_hi", "taste_n", "hosp_hi", "hosp_n")
PERIODS = {"year": "Y", "half": "H", "quarter": "Q", "all": None}


def clean_csat() -> tuple[pd.DataFrame, dict]:
    raw = sources.load("csat_daily")
    n = len(raw)
    deduped = raw.drop_duplicates()
    invalid = (deduped["frnd_high_stsf_cnt_qt"] > deduped["frnd_ansr_cnt_qt"]) | \
              (deduped["qlty_high_stsf_cnt_qt"] > deduped["qlty_ansr_cnt_qt"])
    clean = deduped[~invalid]
    return clean, {"raw_rows": n, "duplicates_removed": n - len(deduped), "invalid_rows_removed": int(invalid.sum()),
                   "rows_kept": len(clean)}


class CsatCube:
    def __init__(self, csat: pd.DataFrame):
        self.start = csat["cal_dt"].min().normalize()
        self.end = csat["cal_dt"].max().normalize()
        self.days = pd.date_range(self.start, self.end, freq="D")
        self.glins = np.array(sorted(csat["glin"].unique()))
        self.g_index = {g: i for i, g in enumerate(self.glins)}
        self.k_index = {k: i for i, k in enumerate(CSAT_CHANNELS)}
        cube = np.zeros((len(self.glins), len(CSAT_CHANNELS), len(self.days), 4))
        gi = csat["glin"].map(self.g_index).to_numpy()
        di = ((csat["cal_dt"] - self.start).dt.days).to_numpy()
        vals = csat[["qlty_high_stsf_cnt_qt", "qlty_ansr_cnt_qt", "frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt"]].to_numpy(float)
        keys = {"all": np.ones(len(csat), bool)}
        for k in CSAT_CHANNELS[1:]:
            kind, label = k.split(":")
            col = "cust_surv_psnt_pos_chnl_ds" if kind == "psnt" else "cust_surv_plcd_pos_chnl_ds"
            if label == "In Restaurant":
                keys[k] = csat[col].isin(["Front Counter", "Table Service"]).to_numpy()
            else:
                keys[k] = (csat[col] == label).to_numpy()
        for k, mask in keys.items():
            np.add.at(cube[:, self.k_index[k]], (gi[mask], di[mask]), vals[mask])
        self.cum = np.concatenate([np.zeros((*cube.shape[:2], 1, 4)), cube.cumsum(axis=2)], axis=2)

    def window_sum(self, g: np.ndarray, k: int, d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
        """Sum over day index [d0, d1] inclusive (clipped to the data range). g < 0 means no CSAT."""
        d0 = np.clip(d0, 0, len(self.days))
        d1 = np.clip(d1 + 1, 0, len(self.days))
        out = np.zeros((len(g), 4))
        ok = (g >= 0) & (d1 > d0)
        out[ok] = self.cum[g[ok], k, d1[ok]] - self.cum[g[ok], k, d0[ok]]
        return out

    def day_index(self, dates: pd.Series) -> np.ndarray:
        return ((pd.to_datetime(dates) - self.start).dt.days).to_numpy()


def _calendar_bounds(dates: pd.Series, freq: str) -> tuple[pd.Series, pd.Series]:
    periods = pd.to_datetime(dates).dt.to_period(freq)
    return periods.dt.start_time, periods.dt.end_time.dt.normalize()


def align_time_windows(cube: CsatCube, visits: pd.DataFrame) -> pd.DataFrame:
    g = visits["glin"].map(cube.g_index).fillna(-1).astype(int).to_numpy()
    vday = cube.day_index(visits["visit_date"])
    frames = []
    for name, spec in WINDOWS.items():
        if isinstance(spec, tuple):
            d0, d1 = vday + spec[0], vday + spec[1]
        else:
            s, e = _calendar_bounds(visits["visit_date"], spec)
            d0, d1 = cube.day_index(s), cube.day_index(e)
        for key in CSAT_CHANNELS:
            sums = cube.window_sum(g, cube.k_index[key], d0, d1)
            frames.append(pd.DataFrame({"visit_id": visits["visit_id"].to_numpy(), "window": name,
                                        "csat_channel": key, **{m: sums[:, i] for i, m in enumerate(MEASURES)}}))
    return pd.concat(frames, ignore_index=True)


def period_label(dates: pd.Series, period: str) -> pd.Series:
    if period == "all":
        return pd.Series("all", index=dates.index)
    p = pd.to_datetime(dates).dt.to_period("Q" if period == "half" else PERIODS[period])
    if period == "half":
        return p.dt.year.astype(str) + "-H" + ((p.dt.quarter + 1) // 2).astype(str)
    return p.astype(str)


def restaurant_period_csat(cube: CsatCube, period: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = cube.days[(cube.days >= start) & (cube.days <= end)]
    labels = period_label(pd.Series(days), period)
    frames = []
    for label in labels.unique():
        sel = days[(labels == label).to_numpy()]
        d0, d1 = (sel.min() - cube.start).days, (sel.max() - cube.start).days
        g = np.arange(len(cube.glins))
        for key in CSAT_CHANNELS:
            sums = cube.window_sum(g, cube.k_index[key], np.full(len(g), d0), np.full(len(g), d1))
            frames.append(pd.DataFrame({"glin": cube.glins, "period": label, "csat_channel": key,
                                        **{m: sums[:, i] for i, m in enumerate(MEASURES)}}))
    return pd.concat(frames, ignore_index=True)


def csat_monthly(csat: pd.DataFrame) -> pd.DataFrame:
    frame = csat.assign(month=csat["cal_dt"].dt.to_period("M").astype(str),
                        weekday=csat["cal_dt"].dt.day_name())
    return (frame.groupby(["glin", "month", "cust_surv_psnt_pos_chnl_ds", "cust_surv_plcd_pos_chnl_ds"])
            [["qlty_high_stsf_cnt_qt", "qlty_ansr_cnt_qt", "frnd_high_stsf_cnt_qt", "frnd_ansr_cnt_qt"]]
            .sum().reset_index()
            .rename(columns={"qlty_high_stsf_cnt_qt": "taste_hi", "qlty_ansr_cnt_qt": "taste_n",
                             "frnd_high_stsf_cnt_qt": "hosp_hi", "frnd_ansr_cnt_qt": "hosp_n",
                             "cust_surv_psnt_pos_chnl_ds": "presented", "cust_surv_plcd_pos_chnl_ds": "placed"}))


def build_visit_scores(decisions: dict) -> tuple[pd.DataFrame, dict]:
    taste, taste_report = scoring.score_taste(decisions)
    hosp, hosp_report = scoring.score_hospitality(decisions)
    keys = ["visit_id", "glin", "visit_date", "survey_id"]
    visits = taste.merge(hosp, on=keys, how="outer")
    visits["survey_name"] = visits["survey_id"].map({204: "RGR", 203: "CFV"})
    dim = sources.restaurant_dim()
    visits = visits.merge(dim.drop(columns=["terr_cd"], errors="ignore"), on="glin", how="left")
    visits["market"] = visits["market"].fillna("Unknown")
    visits["market_group"] = visits["market_group"].fillna("Unknown")
    visits["visit_month"] = visits["visit_date"].dt.to_period("M").astype(str)
    visits["visit_year"] = visits["visit_date"].dt.year
    visits["month_of_year"] = visits["visit_date"].dt.month
    return visits, {"taste": taste_report, "hospitality": hosp_report}


def filter_scope(visits: pd.DataFrame, scope: dict) -> tuple[pd.DataFrame, dict]:
    period = scope["period"]
    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])
    mask = (visits["market"].isin(scope["markets"]) & visits["survey_name"].isin(scope.get("visit_types", ["RGR", "CFV"]))
            & (visits["visit_date"] >= start) & (visits["visit_date"] <= end))
    out = visits[mask].copy()
    return out, {"visits_before_filter": int(len(visits)), "visits_in_scope": int(len(out)),
                 "excluded_unknown_market": int((visits["market"] == "Unknown").sum())}


def gse_index(rp: pd.DataFrame, pace_weight: float = .5) -> pd.DataFrame:
    out = rp.copy()
    for m, short in (("taste", "taste"), ("hospitality", "hosp")):
        for s in ("prog", "cons"):
            out[f"{short}_index_{s}"] = pace_weight * out[f"{short}_{s}"] + (1 - pace_weight) * out[f"csat_{short}"]
    for s in ("prog", "cons"):
        out[f"gse_kpi_{s}"] = (out[f"taste_index_{s}"] + out[f"hosp_index_{s}"]) / 2
    return out


def market_scorecard(visits: pd.DataFrame, csat_clean: pd.DataFrame, scope: dict) -> pd.DataFrame:
    """Deck-style scorecard: PACE (visit pass rate / average) + CSAT top-two-box -> indexes."""
    start, end = pd.Timestamp(scope["period"]["start"]), pd.Timestamp(scope["period"]["end"])
    c = csat_clean[(csat_clean["cal_dt"] >= start) & (csat_clean["cal_dt"] <= end)]
    c = c.assign(market=c["terr_cd"].map(sources.MARKET_BY_TERR))
    rows = []
    for market, v in list(visits.groupby("market")) + [("All in scope", visits)]:
        cm = c[c["market"] == market] if market != "All in scope" else c[c["market"].isin(scope["markets"])]
        row = {"market": market, "pace_visits": int(len(v)),
               "csat_taste_responses": int(cm["qlty_ansr_cnt_qt"].sum()),
               "csat_hosp_responses": int(cm["frnd_ansr_cnt_qt"].sum()),
               "csat_taste": cm["qlty_high_stsf_cnt_qt"].sum() / max(cm["qlty_ansr_cnt_qt"].sum(), 1),
               "csat_hosp": cm["frnd_high_stsf_cnt_qt"].sum() / max(cm["frnd_ansr_cnt_qt"].sum(), 1)}
        for short in ("taste", "hosp"):
            row[f"pace_{short}_cons"] = v[f"{short}_cons"].mean()
            row[f"pace_{short}_prog"] = v[f"{short}_prog"].mean()
            for s in ("cons", "prog"):
                row[f"{short}_index_{s}"] = .5 * row[f"pace_{short}_{s}"] + .5 * row[f"csat_{short}"]
        for s in ("cons", "prog"):
            row[f"gse_kpi_{s}"] = (row[f"taste_index_{s}"] + row[f"hosp_index_{s}"]) / 2
        rows.append(row)
    return pd.DataFrame(rows)
