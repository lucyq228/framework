"""End-to-end run through the skill scripts, standing in for the agents.

Drives every script the way the agents do (subprocess, JSON on stdout), checks
the guardrails refuse what they should, and checks the planted signals in the
synthetic data are recovered. Uses a temporary runs folder and restores
data/reference/ afterwards.

    .venv\\Scripts\\python.exe -m pytest tests/test_end_to_end.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".cursor" / "skills"
ATTRS = ROOT / "data" / "reference" / "restaurant_attributes.csv"


class Run:
    def __init__(self, runs_dir: Path):
        self.env = {**os.environ, "CORR_RUNS_DIR": str(runs_dir), "PYTHONIOENCODING": "utf-8"}
        self.tmp = runs_dir

    def __call__(self, script: str, *args: str, ok: bool = True) -> dict:
        proc = subprocess.run([sys.executable, str(SKILLS / script), *args], cwd=ROOT, env=self.env,
                              capture_output=True, text=True, encoding="utf-8")
        assert proc.stdout.strip(), f"{script} printed nothing. stderr:\n{proc.stderr}"
        out = json.loads(proc.stdout)
        if ok:
            assert proc.returncode == 0 and out.get("ok", True), f"{script} failed: {out}\n{proc.stderr[-2000:]}"
        else:
            assert proc.returncode != 0 and out.get("ok") is False, f"{script} should have refused: {out}"
        return out

    def file(self, name: str, payload) -> str:
        path = self.tmp / name
        path.write_text(json.dumps(payload), encoding="utf-8-sig")
        return str(path)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    existed = ATTRS.exists()
    if existed:
        ATTRS.unlink()
    yield Run(tmp_path_factory.mktemp("runs"))
    if ATTRS.exists() and not existed:
        ATTRS.unlink()


def result(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / "results" / f"{name}.json").read_text(encoding="utf-8"))


def est(res: dict, label: str) -> float:
    return next(f["estimate"] for f in res["findings"] if f["label"] == label)


def test_full_run(run):
    start = run("run-correlation-project/scripts/start_run.py", "--label", "e2e")
    run_dir = Path(start["run_dir"])
    assert "restaurant_attributes" in start["not_available"]

    run("data-quality-eda/scripts/data_quality.py", ok=False)
    base = run("data-quality-eda/scripts/baseline_eda.py")
    assert base["observations"]
    ctx = run("scope-hypotheses/scripts/read_context.py")
    assert ctx["sources"]["restaurant_attributes"]["acquirable_on_request"] is True

    bad_scope = {"project_name": "x", "hypotheses": [{"id": "H1", "pathway": [{"tool": "build-readout"}]}]}
    assert run("scope-hypotheses/scripts/save_scope_proposal.py", "--file", run.file("bad.json", bad_scope), ok=False)["issues"]
    example = json.loads((ROOT / "examples" / "example_scope_proposal.json").read_text(encoding="utf-8"))
    question = {"id": "Q1", "question": "Include 2025?", "options": ["yes", "no"], "recommended": "yes",
                "why_it_matters": "Doubles the visits per restaurant"}
    saved = run("scope-hypotheses/scripts/save_scope_proposal.py", "--file",
                run.file("scope.json", {**example, "user_questions": [question]}))
    assert saved["open_user_questions"] == ["Q1"]

    run("run-correlation-project/scripts/approve_scope.py", "--user-confirmation", "yes", ok=False)
    matrix = run("feasibility-check/scripts/check_feasibility.py")
    assert matrix["counts"]["blocked"] == 1 and matrix["data_requests"][0]["source"] == "restaurant_attributes"

    judgment = {"summary": "Most pathways run; restaurant-type cuts need the restaurant master.",
                "can_do": ["Market-level and visit-level correlations"], "cannot_do": ["Product-level taste"],
                "step_judgments": [{"step_id": "H6.3", "verdict": "partial", "reason": "Breakpoint p-values are naive"}],
                "recommendation": "Acquire restaurant_attributes, then approve."}
    relax = {**judgment, "step_judgments": [{"step_id": "H4.2", "verdict": "supported", "reason": "relax"}]}
    run("feasibility-check/scripts/save_feasibility_review.py", "--file", run.file("relax.json", relax), ok=False)
    run("feasibility-check/scripts/save_feasibility_review.py", "--file", run.file("a3.json", judgment))
    assert "Q1" in run("run-correlation-project/scripts/approve_scope.py", "--user-confirmation", "yes", ok=False)["error"]
    assert run("run-correlation-project/scripts/run_status.py")["pending"]["open_user_questions"][0]["id"] == "Q1"
    run("run-correlation-project/scripts/record_answers.py", "--answers-file",
        run.file("answers.json", [{"id": "Q1", "answer": "yes", "user_words": "yes include 2025"}]))
    assert "blocked" in run("run-correlation-project/scripts/approve_scope.py", "--user-confirmation", "yes", ok=False)["error"].lower()

    run("run-correlation-project/scripts/request_data.py", "--source", "restaurant_attributes",
        "--user-confirmation", "yes, get the restaurant master")
    status = run("run-correlation-project/scripts/run_status.py")
    assert "feasibility" in status["stale"]
    assert "inventory" in run("run-correlation-project/scripts/approve_scope.py", "--user-confirmation", "yes", ok=False)["error"]
    assert run("feasibility-check/scripts/check_feasibility.py")["counts"]["blocked"] == 0
    run("feasibility-check/scripts/save_feasibility_review.py", "--file", run.file("a3.json", judgment))
    lock = run("run-correlation-project/scripts/approve_scope.py", "--user-confirmation", "yes, approved")
    assert lock["partial_steps"] == ["H6.3"]
    run("scope-hypotheses/scripts/save_scope_proposal.py", "--file", str(ROOT / "examples" / "example_scope_proposal.json"), ok=False)

    run("correlation-shelf/scripts/run_plan.py", ok=False)
    dq = run("data-quality-eda/scripts/data_quality.py")
    assert dq["overall"] in {"warn", "info", "pass"}
    run("feature-creation/scripts/build_datasets.py")
    run("data-quality-eda/scripts/descriptive_eda.py")

    plan = run("correlation-shelf/scripts/run_plan.py")
    assert not plan["missing_steps"] and plan["ran"][-1]["tool"] == "multiple-testing"
    run("correlation-shelf/scripts/run_tool.py", "--tool", "pairwise-correlation", ok=False)
    run("correlation-shelf/scripts/run_tool.py", "--tool", "product-level-csat", "--followup-reason", "try", ok=False)
    fu = run("correlation-shelf/scripts/run_tool.py", "--tool", "between-within-correlation",
             "--followup-reason", "Check restaurant vs over-time association")
    assert fu["result_file"].startswith("results/F_")
    run("correlation-shelf/scripts/run_tool.py", "--tool", "rank-agreement", "--param", "grain=visit",
        "--followup-reason", "x", ok=False)

    # Planted signals.
    simpson = result(run_dir, "H1_3")
    for m in ("taste", "hospitality"):
        assert est(simpson, f"{m} within market_group") > est(simpson, f"{m} pooled") + .05
    lag = result(run_dir, "H2_1")
    assert est(lag, "hospitality window pm30") > est(lag, "hospitality window year")
    gap = result(run_dir, "H3_1")
    for m in ("taste", "hospitality"):
        assert est(gap, f"{m} RGRV-CFV gap") > 0
        assert est(gap, f"{m} CFV vs CSAT") > est(gap, f"{m} RGRV vs CSAT")

    checks = run("review-synthesis/scripts/review_checks.py")
    assert any("FDR table is older" in i for i in checks["run_issues"])
    run("correlation-shelf/scripts/run_plan.py", "--steps", "H8.4")
    checks = run("review-synthesis/scripts/review_checks.py")
    assert not any("FDR" in i for i in checks["run_issues"])

    hyps = [f"H{i}" for i in range(1, 9)]
    review = {"verdict": "pass_with_caveats", "summary": "Associations are positive and stronger within market.",
              "hypothesis_verdicts": [{"id": h, "verdict": "supported", "reason": "test", "evidence": [f"{h}_1"]} for h in hyps],
              "challenges": [], "reconciliations": [], "rerun_requests": []}
    run("review-synthesis/scripts/save_review.py", "--file",
        run.file("review_bad.json", {**review, "hypothesis_verdicts": review["hypothesis_verdicts"][:3]}), ok=False)
    run("build-deliverables/scripts/build_deliverables.py", ok=False)
    run("review-synthesis/scripts/save_review.py", "--file", run.file("review.json", review))
    insights = {"headline": "PACE and CSAT agree within market, not across markets.",
                "executive_summary": ["Within-market correlation is about 0.35."],
                "insights": [{"id": "I1", "title": "Market confounding", "tag": "diverges", "importance": "high",
                              "robustness": "high", "so_what": "Compare within market.",
                              "evidence": [{"statement": "Pooled r is lower than within-market r", "result": "H1_3"},
                                           {"statement": "Coverage", "result": "eda:eda-coverage"}]}],
                "limitations": ["Synthetic data."], "data_requests": [], "next_steps": []}
    run("review-synthesis/scripts/save_insights.py", "--file",
        run.file("ins_bad.json", {**insights, "insights": [{**insights["insights"][0], "evidence": [{"statement": "x", "result": "nope"}]}]}),
        ok=False)
    run("review-synthesis/scripts/save_insights.py", "--file", run.file("ins.json", insights))
    dl = run("build-deliverables/scripts/build_deliverables.py")
    assert any(f.endswith(".xlsx") for f in dl["files"]) and "insight_readout.html" in dl["files"]

    final = run("run-correlation-project/scripts/finalize_run.py")
    assert final["completed"], final["missing_milestones"]
    assert (run_dir / "live_run.log").exists()
