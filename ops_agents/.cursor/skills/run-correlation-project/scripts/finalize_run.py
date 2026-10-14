"""Close the run: write run_summary.json and autonomous_trace.json, and list the deliverables."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402
from corr_core.scope import load_approved_scope  # noqa: E402
from corr_core.state import milestone_status, read_artifact, write_artifact  # noqa: E402


def main() -> None:
    configure_stdout()
    run_dir = resolve(base_parser(__doc__).parse_args())
    scope = load_approved_scope(run_dir)
    status = milestone_status(run_dir)
    missing = [s["milestone"] for s in status if not s["done"]]
    review = read_artifact(run_dir, "review.json", {})
    insights = read_artifact(run_dir, "insights.json", {})
    deliverables = read_artifact(run_dir, "deliverables.json", {})
    completed = not missing
    events.emit(run_dir, "complete" if completed else "stopped", "Manager",
                "All milestones and deliverables exist." if completed else f"Run stopped with missing milestones: {missing}")
    trace = events.read_events(run_dir)
    write_artifact(run_dir, "autonomous_trace.json", {"events": trace})
    summary = {"run_id": run_dir.name, "project_name": scope["project_name"], "completed": completed,
               "missing_milestones": missing, "review_verdict": review.get("verdict"),
               "headline": insights.get("headline"), "insights": len(insights.get("insights", [])),
               "trace_events": len(trace)}
    write_artifact(run_dir, "run_summary.json", summary)
    files = list(deliverables.get("files", [])) + ["approved_scope.json", "feasibility_review.json", "review.json",
                                                    "insights.json", "live_run.log", "events.jsonl"]
    emit_json({**summary, "deliverables": {n: str(run_dir / n) for n in dict.fromkeys(files) if (run_dir / n).exists()}})


if __name__ == "__main__":
    main()
