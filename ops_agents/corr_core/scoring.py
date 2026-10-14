"""PACE scoring as defined in the GSE KPI deck.

Taste:       product PASS -> visit (conservative all-pass / progressive share) -> restaurant-period.
Hospitality: question -> touchpoint -> channel journey PASS -> visit (conservative / progressive).

Scope decisions change the scoring, so every function takes the `decisions`
block from the scope (see corr_core.scope.DEFAULT_DECISIONS).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from corr_core import sources
from corr_core.sources import DIAGNOSTIC_CODES, TASTE_INDEX_CODES

CHANNEL_BY_AREA = {
    "General Drive Thru Service": "Drive Thru", "Drive Thru": "Drive Thru",
    "General In Restaurant Service": "In Restaurant", "In Restaurant": "In Restaurant",
    "Delivery": "Delivery", "Curbside": "Curbside", "Service": "General Service",
}
CHANNEL_KEY = {"Drive Thru": "dt", "In Restaurant": "ir", "Delivery": "dl", "Curbside": "cb"}
TOUCHPOINTS = ("order", "pay", "present", "care")
DIMENSIONS = ("appearance", "taste", "temperature", "texture")


def mapping_table(decisions: dict) -> dict[str, str | None]:
    """local code -> global code (None = drop) following the mapping decisions."""
    if not sources.available("question_mapping"):
        return {}
    table = sources.load("question_mapping").fillna("")
    out = {}
    for row in table.itertuples():
        if row.mapping_status == "confirmed" or (
                row.mapping_status == "proposed" and decisions.get("proposed_mappings") == "include"):
            out[row.local_question_code] = row.global_question_code or None
        else:
            out[row.local_question_code] = None
    return out


def apply_question_mapping(frame: pd.DataFrame, decisions: dict) -> tuple[pd.DataFrame, dict]:
    mapping = mapping_table(decisions)
    frame = frame.copy()
    frame["source_question_code"] = frame["question_code"]
    local = frame["question_code"].isin(mapping)
    frame.loc[local, "question_code"] = frame.loc[local, "question_code"].map(mapping)
    is_local_core = frame["source_question_code"].str.startswith("Q27a")
    report = {
        "local_rows": int(local.sum()),
        "mapped_rows": int((local & frame["question_code"].notna()).sum()),
        "dropped_unmapped_rows": int((local & frame["question_code"].isna() & ~is_local_core).sum()),
        "local_core_rows": int(is_local_core.sum()),
        "mapping_used": {k: v for k, v in mapping.items()},
    }
    return frame, report


def _reason_dimension(reason: str) -> str | None:
    match = re.search(r"standard for (\w+)", str(reason))
    return match.group(1) if match else ("other" if "Other" in str(reason) else None)


def score_taste(decisions: dict) -> tuple[pd.DataFrame, dict]:
    raw = sources.load("pace_product")
    n_raw = len(raw)
    raw = raw.drop_duplicates()
    mapped, mapping_report = apply_question_mapping(raw, decisions)
    local_core = mapped["source_question_code"].str.startswith("Q27a")
    mapped["category"] = mapped["question_code"].map({**TASTE_INDEX_CODES, **DIAGNOSTIC_CODES})
    mapped.loc[local_core, "category"] = "Local Core"
    mapped = mapped[mapped["category"].notna()]
    mapped["eligible"] = mapped["question_exception_flag"] >= 0
    mapped["passed"] = (mapped["question_exception_flag"] == 0).astype(float)
    mapped["fail_dimension"] = np.where(mapped["question_exception_flag"] == 1,
                                        mapped["reasons"].map(_reason_dimension), None)

    keys = ["visit_id", "glin", "visit_date", "survey_id"]
    index_rows = mapped[mapped["category"].isin(TASTE_INDEX_CODES.values()) & mapped["eligible"]]
    # Local core product replaces an unavailable global category and inherits its weight.
    if decisions.get("local_core_products", "replace_unavailable") == "replace_unavailable":
        core = mapped[(mapped["category"] == "Local Core") & mapped["eligible"]]
        n_index = index_rows.groupby("visit_id").size()
        short = n_index[n_index < len(TASTE_INDEX_CODES)].index
        index_rows = pd.concat([index_rows, core[core["visit_id"].isin(short)]])
    visit = index_rows.groupby(keys).agg(
        taste_eligible=("passed", "size"), taste_passed=("passed", "sum")).reset_index()
    visit["taste_prog"] = visit["taste_passed"] / visit["taste_eligible"]
    visit["taste_cons"] = (visit["taste_passed"] == visit["taste_eligible"]).astype(float)

    cat = mapped[mapped["eligible"]].pivot_table(index="visit_id", columns="category", values="passed", aggfunc="mean")
    cat.columns = [f"cat_{c.lower().replace(' ', '_')}_pass" for c in cat.columns]
    fails = mapped[mapped["question_exception_flag"] == 1]
    dims = pd.crosstab(fails["visit_id"], fails["fail_dimension"]) if len(fails) else pd.DataFrame()
    dims = dims.reindex(columns=list(DIMENSIONS) + ["other"], fill_value=0)
    dims.columns = [f"fail_{c}" for c in dims.columns]
    visit = visit.merge(cat, left_on="visit_id", right_index=True, how="left")
    visit = visit.merge(dims, left_on="visit_id", right_index=True, how="left")
    fail_cols = [c for c in visit.columns if c.startswith("fail_")]
    visit[fail_cols] = visit[fail_cols].fillna(0).astype(int)
    report = {
        "raw_rows": n_raw, "duplicate_rows_removed": n_raw - len(raw), "visits_scored": int(len(visit)),
        "mean_eligible_products": round(float(visit["taste_eligible"].mean()), 3),
        "visits_with_fewer_than_5_products": int((visit["taste_eligible"] < 5).sum()),
        "question_mapping": mapping_report,
    }
    return visit, report


def score_hospitality(decisions: dict) -> tuple[pd.DataFrame, dict]:
    raw = sources.load("pace_service")
    n_raw = len(raw)
    raw = raw.drop_duplicates()
    mapped, mapping_report = apply_question_mapping(raw, decisions)
    mapped = mapped[mapped["question_code"].notna()]
    mapped["channel"] = mapped["survey_area"].map(CHANNEL_BY_AREA)
    if decisions.get("general_service_questions", "exclude") == "include_as_in_restaurant":
        mapped.loc[mapped["channel"] == "General Service", "channel"] = "In Restaurant"
    assessed = mapped[(mapped["question_exception_flag"] >= 0) & (mapped["channel"] != "General Service")].copy()
    assessed["q_pass"] = (assessed["question_exception_flag"] == 0).astype(float)

    keys = ["visit_id", "glin", "visit_date", "survey_id", "survey_name"]
    journey = assessed.groupby(keys + ["channel"])["q_pass"].min().rename("journey_pass").reset_index()
    visit = journey.groupby(keys).agg(hosp_channels=("journey_pass", "size"),
                                      hosp_passed=("journey_pass", "sum")).reset_index()
    visit["hosp_prog"] = visit["hosp_passed"] / visit["hosp_channels"]
    visit["hosp_cons"] = (visit["hosp_passed"] == visit["hosp_channels"]).astype(float)
    tested = journey.groupby("visit_id")["channel"].agg(lambda s: s.iloc[0] if len(s) == 1 else "Multiple")
    visit["channel_tested"] = visit["visit_id"].map(tested)

    ch = journey.pivot_table(index="visit_id", columns="channel", values="journey_pass", aggfunc="mean")
    ch.columns = [f"ch_{CHANNEL_KEY[c]}_pass" for c in ch.columns]
    tp_rows = []
    for tp in TOUCHPOINTS:
        sub = assessed[assessed[f"{tp}_touchpoint"] == 1]
        tp_rows.append(sub.groupby("visit_id")["q_pass"].min().rename(f"tp_{tp}_pass"))
    tp = pd.concat(tp_rows, axis=1)
    general = mapped[(mapped["channel"] == "General Service") & (mapped["question_exception_flag"] >= 0)]
    gen = (general.assign(q=(general["question_exception_flag"] == 0).astype(float))
           .groupby("visit_id")["q"].mean().rename("general_service_rate"))
    visit = visit.merge(ch, left_on="visit_id", right_index=True, how="left")
    visit = visit.merge(tp, left_on="visit_id", right_index=True, how="left")
    visit = visit.merge(gen, left_on="visit_id", right_index=True, how="left")
    report = {
        "raw_rows": n_raw, "duplicate_rows_removed": n_raw - len(raw), "visits_scored": int(len(visit)),
        "general_service_treatment": decisions.get("general_service_questions", "exclude"),
        "channels_assessed": journey["channel"].value_counts().to_dict(),
        "question_mapping": mapping_report,
    }
    return visit, report


def question_level(decisions: dict) -> dict[str, pd.DataFrame]:
    """Mapped question-level rows for EDA (pass rates by product / question / touchpoint)."""
    prod, _ = apply_question_mapping(sources.load("pace_product").drop_duplicates(), decisions)
    prod["category"] = prod["question_code"].map({**TASTE_INDEX_CODES, **DIAGNOSTIC_CODES})
    prod.loc[prod["source_question_code"].str.startswith("Q27a"), "category"] = "Local Core"
    prod["fail_dimension"] = np.where(prod["question_exception_flag"] == 1, prod["reasons"].map(_reason_dimension), None)
    serv, _ = apply_question_mapping(sources.load("pace_service").drop_duplicates(), decisions)
    serv["channel"] = serv["survey_area"].map(CHANNEL_BY_AREA)
    serv["touchpoint"] = np.select([serv[f"{t}_touchpoint"] == 1 for t in TOUCHPOINTS], TOUCHPOINTS, "none")
    return {"product": prod, "service": serv}
