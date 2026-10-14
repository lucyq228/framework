"""Run the whole deterministic pipeline on the example scope, without agents.

Usage: python tests/smoke_pipeline.py [--keep]
Runs in a temporary runs folder (CORR_RUNS_DIR) and simulates acquiring the
restaurant attributes file when feasibility asks for it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CORR_RUNS_DIR", tempfile.mkdtemp(prefix="corr_smoke_"))

from corr_core import paths, registry, runner, sources, state  # noqa: E402
from corr_core import scope as sc  # noqa: E402

BASELINE = ["gcp-connector", "profile-sources", "confluence-mcp", "eda-coverage", "eda-pace-pass-rates",
            "eda-csat-rates", "eda-trends", "eda-failure-reasons", "eda-segment-profile", "eda-summary-pack"]
FEATURE = ["build-analysis-dataset", "apply-question-mapping", "score-pace-taste", "score-pace-hospitality",
           "aggregate-csat", "align-time-windows", "compute-gse-index"]
DESCRIPTIVE = ["eda-coverage", "eda-pace-pass-rates", "eda-csat-rates", "eda-trends", "eda-failure-reasons",
               "eda-segment-profile", "eda-summary-pack"]


def timed(name, fn):
    t = time.time()
    out = fn()
    print(f"\n== {name} ({time.time() - t:.1f}s): {out.get('summary', '') if isinstance(out, dict) else ''}")
    for e in (out or {}).get("errors", []) if isinstance(out, dict) else []:
        print("   ERROR", e)
        MILESTONE_ERRORS.append(e)
    return out


MILESTONE_ERRORS: list = []


def feasibility(run, proposal):
    chk = registry.check_scope(proposal)
    review = {**chk, "scope_sha256": state.sha(run / sc.SCOPE_PROPOSAL),
              "inventory_fingerprint": sources.inventory()["fingerprint"], "summary": str(chk["counts"])}
    state.write_artifact(run, sc.FEASIBILITY, review)
    return review


def main() -> int:
    acquired = ROOT / "data" / "reference" / "restaurant_attributes.csv"
    if acquired.exists():
        acquired.unlink()
    run = paths.new_run_dir("smoke")
    paths.set_current_run(run)
    print("run:", run)

    timed("baseline", lambda: runner.run_milestone(run, "baseline_eda.json", "smoke", BASELINE))
    proposal = json.loads((ROOT / "examples" / "example_scope_proposal.json").read_text(encoding="utf-8"))
    assert not sc.validate_scope(proposal), sc.validate_scope(proposal)
    state.write_artifact(run, sc.SCOPE_PROPOSAL, proposal)
    review = timed("feasibility (before acquisition)", lambda: feasibility(run, proposal))
    print("   requests:", [r["source"] for r in review["data_requests"]])

    shutil.copy(ROOT / "data" / "acquirable" / "restaurant_attributes.csv", acquired)
    review = timed("feasibility (after acquisition)", lambda: feasibility(run, proposal))
    print("   approve:", sc.approve(run, "smoke-test", "approved in smoke test"))

    timed("data quality", lambda: runner.run_milestone(run, "dq_report.json", "smoke", ["data-quality-check"]))
    timed("features", lambda: runner.run_milestone(run, "analysis_datasets.json", "smoke", FEATURE))
    timed("descriptive eda", lambda: runner.run_milestone(run, "eda_descriptives.json", "smoke", DESCRIPTIVE))

    scope = sc.load_approved_scope(run)
    index, errors = [], []
    t0 = time.time()
    for step in scope["plan"]:
        t = time.time()
        try:
            r = runner.execute(step["tool"], run, step.get("params"), actor="smoke", step=step)
            index.append({"step_id": step["step_id"], "tool": step["tool"], "status": r["status"],
                          "file": f"results/{step['step_id'].replace('.', '_')}.json", "summary": r["summary"]})
            print(f"   {step['step_id']:<6} {step['tool']:<32} {time.time() - t:5.1f}s {r['status']:<8} {r['summary'][:160]}")
        except Exception as exc:
            errors.append({"step_id": step["step_id"], "tool": step["tool"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"   {step['step_id']:<6} {step['tool']:<32} ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
    state.write_artifact(run, "analyses_index.json", {"summary": f"{len(index)} ran, {len(errors)} errors",
                                                       "results": index, "errors": errors})
    print(f"\n== analyses ({time.time() - t0:.1f}s): {len(index)} ok, {len(errors)} errors")

    if "--sweep" in sys.argv:
        print("\n== shelf sweep (follow-ups, not indexed)")
        for e in registry.entries():
            if e["phase"] != "analysis":
                continue
            variants = [{}]
            if "grain" in registry.allowed_params(e):
                variants.append({"grain": "restaurant_period"})
            for p in variants:
                label = f"sweep_{e['id']}_{p.get('grain', 'default')}"
                try:
                    r = runner.execute(e["id"], run, p, label=label, actor="smoke", followup_reason="smoke sweep")
                    print(f"   {e['id']:<32} {str(p):<30} {r['status']:<8} {r['summary'][:140]}")
                except runner.ToolBlocked as exc:
                    print(f"   {e['id']:<32} {str(p):<30} BLOCKED  {exc}")
                except Exception as exc:
                    errors.append({"tool": e["id"], "params": p, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"   {e['id']:<32} {str(p):<30} ERROR {type(exc).__name__}: {exc}")
                    traceback.print_exc(limit=2)
        for f in (run / "results").glob("sweep_*.json"):
            f.unlink()

    state.write_artifact(run, "review.json", {"verdict": "pass", "analyses_index_sha256": state.sha(run / "analyses_index.json"),
                                              "hypothesis_verdicts": [], "rerun_requests": []})
    state.write_artifact(run, "insights.json", {
        "review_sha256": state.sha(run / "review.json"),
        "headline": "Smoke-test headline", "executive_summary": ["Smoke test only."], "limitations": ["Synthetic data."],
        "insights": [
            {"id": f"I{i + 1}", "title": e["summary"][:80], "tag": "test", "importance": "medium", "robustness": "medium",
             "so_what": "smoke", "evidence": [{"statement": e["summary"][:120], "result": Path(e["file"]).stem}]}
            for i, e in enumerate(index[:5])]})
    timed("deliverables", lambda: runner.run_milestone(run, "deliverables.json", "smoke",
                                                       ["chart-pack", "build-excel-prototype", "build-readout"],
                                                       extra={"insights_sha256": state.sha(run / "insights.json")}))
    for s in state.milestone_status(run):
        print("  ", s["milestone"], "done" if s["done"] else "MISSING", s.get("stale", ""))
    acquired.unlink(missing_ok=True)
    return 1 if errors or MILESTONE_ERRORS else 0


if __name__ == "__main__":
    raise SystemExit(main())
