"""Scoring rules from the GSE KPI deck, checked on the synthetic sources, plus stats helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corr_core import features, scoring, stats  # noqa: E402
from corr_core.scope import DEFAULT_DECISIONS  # noqa: E402


@pytest.fixture(scope="module")
def taste():
    return scoring.score_taste(DEFAULT_DECISIONS)[0]


@pytest.fixture(scope="module")
def hosp():
    return scoring.score_hospitality(DEFAULT_DECISIONS)[0]


def test_taste_progressive_and_conservative(taste):
    assert taste["taste_prog"].between(0, 1).all()
    assert set(taste["taste_cons"].unique()) <= {0.0, 1.0}
    # Conservative passes exactly when every eligible product passes.
    assert ((taste["taste_cons"] == 1) == (taste["taste_prog"] == 1)).all()
    assert (taste["taste_eligible"] <= 5).all()


def test_local_core_only_replaces_unavailable_category():
    with_core = scoring.score_taste({**DEFAULT_DECISIONS, "local_core_products": "replace_unavailable"})[0]
    without = scoring.score_taste({**DEFAULT_DECISIONS, "local_core_products": "exclude"})[0]
    assert with_core["taste_eligible"].mean() > without["taste_eligible"].mean()
    assert with_core["taste_eligible"].max() == without["taste_eligible"].max() == 5


def test_hospitality_journeys(hosp):
    assert hosp["hosp_prog"].between(0, 1).all()
    assert ((hosp["hosp_cons"] == 1) == (hosp["hosp_prog"] == 1)).all()
    cfv = hosp[hosp["survey_name"] == "CFV"]
    assert (cfv["hosp_channels"] == 1).all(), "a CFV tests one channel"
    rgr = hosp[hosp["survey_name"] == "RGR"]
    assert rgr["hosp_channels"].mean() > 1.5, "an RGRV tests every channel"
    if "ch_dl_pass" in hosp:
        assert hosp.loc[hosp["ch_dl_pass"].notna(), "survey_name"].eq("RGR").all(), "Delivery is RGRV-only"
    if "ch_cb_pass" in hosp:
        assert hosp.loc[hosp["ch_cb_pass"].notna(), "survey_name"].eq("CFV").all(), "Curbside is CFV-only"


def test_general_service_questions_decision():
    excl = scoring.score_hospitality({**DEFAULT_DECISIONS, "general_service_questions": "exclude"})[0]
    incl = scoring.score_hospitality({**DEFAULT_DECISIONS, "general_service_questions": "include_as_in_restaurant"})[0]
    assert len(excl) == len(incl)
    assert not excl["hosp_prog"].equals(incl["hosp_prog"])


def test_question_mapping_decisions():
    excluded = scoring.mapping_table({**DEFAULT_DECISIONS, "proposed_mappings": "exclude"})
    included = scoring.mapping_table({**DEFAULT_DECISIONS, "proposed_mappings": "include"})
    assert excluded["Q9a-CA"] == "Q9a" and excluded["Q10a-UK"] is None
    assert included["Q10a-UK"] == "Q10a"


def test_windows():
    for name, w in features.WINDOWS.items():
        if name.startswith("pm"):
            assert w[0] == -w[1]
        elif name.startswith("pre"):
            assert w[1] == -1, "pre-visit windows exclude the visit day"
    assert {"month", "quarter", "year"} <= set(features.WINDOWS)


def test_stats_helpers():
    lo, hi = stats.fisher_ci(.3, 100)
    assert lo < .3 < hi
    q = stats.bh_fdr([.001, .01, .04, .5])
    assert all(a <= b for a, b in zip(q, q[1:])) and q[-1] == pytest.approx(.5)
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    assert stats.distance_corr(x, x ** 2) > .3 > abs(stats.corr(x, x ** 2)[0])
