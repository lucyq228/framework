"""Descriptive EDA on the scored, in-scope datasets. Writes eda_descriptives.json.

Pass rates by category, product, channel, touchpoint, visit type, market, and
month; score distributions; failure reasons; CSAT by channel and month; trends;
restaurant-type profiles when attributes exist. Becomes the Excel Descriptives tab.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core.runner import base_parser, configure_stdout, emit_json, resolve, run_milestone  # noqa: E402
from corr_core.state import milestone_done  # noqa: E402

TOOLS = ["eda-coverage", "eda-pace-pass-rates", "eda-csat-rates", "eda-trends", "eda-failure-reasons",
         "eda-segment-profile", "eda-summary-pack"]


def main() -> None:
    configure_stdout()
    args = base_parser(__doc__).parse_args()
    run_dir = resolve(args)
    if not milestone_done(run_dir, "analysis_datasets"):
        emit_json({"ok": False, "error": "analysis_datasets is missing; A5 must build the datasets first."})
        raise SystemExit(1)
    out = run_milestone(run_dir, "eda_descriptives.json", args.actor, TOOLS)
    obs = next((s.get("observations", []) for s in out["steps"] if s["tool"] == "eda-summary-pack"), [])
    emit_json({"ok": not out["errors"], "summary": out["summary"],
               "steps": {s["tool"]: s["summary"] for s in out["steps"]}, "observations": obs, "errors": out["errors"]})
    raise SystemExit(1 if out["errors"] else 0)


if __name__ == "__main__":
    main()
