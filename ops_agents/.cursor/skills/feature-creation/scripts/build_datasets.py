"""Build every analysis dataset from the locked scope. Writes analysis_datasets.json.

Applies question mapping, scores PACE Taste and Hospitality (progressive and
conservative), cleans and aggregates CSAT, aligns time windows around each
visit, and computes the GSE KPI indexes. Refuses when the DQ report failed,
unless --dq-override-reason records why the user accepted the risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core.runner import base_parser, configure_stdout, emit_json, resolve, run_milestone  # noqa: E402
from corr_core.state import milestone_done, read_artifact  # noqa: E402

TOOLS = ["build-analysis-dataset", "apply-question-mapping", "score-pace-taste", "score-pace-hospitality",
         "aggregate-csat", "align-time-windows", "compute-gse-index"]


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--dq-override-reason", help="Why the user accepted a failing DQ report.")
    args = parser.parse_args()
    run_dir = resolve(args)
    if not milestone_done(run_dir, "data_quality"):
        emit_json({"ok": False, "error": "dq_report.json is missing; A4 must run data quality first."})
        raise SystemExit(1)
    dq = read_artifact(run_dir, "dq_report.json")
    overall = next((s.get("overall") for s in dq.get("steps", []) if s.get("tool") == "data-quality-check"), None)
    if overall == "fail" and not args.dq_override_reason:
        emit_json({"ok": False, "error": "The DQ report has failing checks. The manager must resolve them with the user "
                                         "or pass --dq-override-reason."})
        raise SystemExit(1)
    out = run_milestone(run_dir, "analysis_datasets.json", args.actor, TOOLS,
                        extra={"dq_overall": overall, **({"dq_override_reason": args.dq_override_reason}
                                                         if args.dq_override_reason else {})})
    emit_json({"ok": not out["errors"], "summary": out["summary"],
               "steps": {s["tool"]: s["summary"] for s in out["steps"]}, "errors": out["errors"]})
    raise SystemExit(1 if out["errors"] else 0)


if __name__ == "__main__":
    main()
