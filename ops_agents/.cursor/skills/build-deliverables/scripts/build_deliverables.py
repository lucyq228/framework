"""Build the charts, the Excel prototype, and the insight readout. Writes deliverables.json.

Needs current insights (tied to the current review). Output files land in the run folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core.runner import base_parser, configure_stdout, emit_json, resolve, run_milestone  # noqa: E402
from corr_core.state import milestone_done, sha, write_artifact  # noqa: E402

TOOLS = ["chart-pack", "build-excel-prototype", "build-readout"]


def main() -> None:
    configure_stdout()
    args = base_parser(__doc__).parse_args()
    run_dir = resolve(args)
    if not milestone_done(run_dir, "insights"):
        emit_json({"ok": False, "error": "insights.json is missing or stale; A7 must save insights first."})
        raise SystemExit(1)
    files = []
    out = run_milestone(run_dir, "deliverables.json", args.actor, TOOLS,
                        extra={"insights_sha256": sha(run_dir / "insights.json")})
    for s in out["steps"]:
        files += ([s["file"]] if s.get("file") else []) + list(s.get("files") or [])
    out["files"] = files
    write_artifact(run_dir, "deliverables.json", out)
    emit_json({"ok": not out["errors"], "summary": out["summary"], "files": {f: str(run_dir / f) for f in files},
               "errors": out["errors"]})
    raise SystemExit(1 if out["errors"] else 0)


if __name__ == "__main__":
    main()
