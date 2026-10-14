"""Baseline EDA before scoping: inventory, profiles, context extraction, and raw-data EDA.

Runs on all markets and dates with default scoring decisions (no scope yet).
Writes baseline_eda.json and EDA charts. Prints summaries only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core.runner import base_parser, configure_stdout, emit_json, resolve, run_milestone  # noqa: E402

TOOLS = ["gcp-connector", "profile-sources", "confluence-mcp", "eda-coverage", "eda-pace-pass-rates",
         "eda-csat-rates", "eda-trends", "eda-failure-reasons", "eda-segment-profile", "eda-summary-pack"]


def main() -> None:
    configure_stdout()
    args = base_parser(__doc__).parse_args()
    run_dir = resolve(args)
    if (run_dir / "approved_scope.json").exists():
        emit_json({"ok": False, "error": "The scope is approved; run descriptive_eda.py instead."})
        raise SystemExit(1)
    out = run_milestone(run_dir, "baseline_eda.json", args.actor, TOOLS)
    obs = next((s.get("observations", []) for s in out["steps"] if s["tool"] == "eda-summary-pack"), [])
    emit_json({"ok": not out["errors"], "summary": out["summary"],
               "steps": {s["tool"]: s["summary"] for s in out["steps"]}, "observations": obs, "errors": out["errors"]})
    raise SystemExit(1 if out["errors"] else 0)


if __name__ == "__main__":
    main()
