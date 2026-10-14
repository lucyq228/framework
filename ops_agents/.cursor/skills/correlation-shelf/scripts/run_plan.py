"""Run the approved plan's analysis steps and rebuild analyses_index.json.

By default runs every planned step that has no result yet. --steps H1.2,H3.1
re-runs specific steps (for example after a review rerun request); --all re-runs
everything. multiple-testing steps always run last so they see every p-value.
One result file per step: results/<step_id>.json. Prints one line per step.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events  # noqa: E402
from corr_core.runner import (base_parser, configure_stdout, emit_json, execute, resolve,  # noqa: E402
                              write_analyses_index)
from corr_core.scope import ScopeLockError, load_approved_scope  # noqa: E402
from corr_core.state import milestone_done  # noqa: E402


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--steps", help="Comma-separated step ids to (re)run.")
    parser.add_argument("--all", action="store_true", help="Re-run every planned step.")
    args = parser.parse_args()
    run_dir = resolve(args)
    try:
        scope = load_approved_scope(run_dir)
    except ScopeLockError as exc:
        emit_json({"ok": False, "error": str(exc)})
        raise SystemExit(1)
    if not milestone_done(run_dir, "analysis_datasets"):
        emit_json({"ok": False, "error": "analysis_datasets is missing; A5 must build the datasets first."})
        raise SystemExit(1)
    plan = scope["plan"]
    wanted = {s.strip() for s in args.steps.split(",")} if args.steps else None
    if wanted and wanted - {s["step_id"] for s in plan}:
        emit_json({"ok": False, "error": f"Not in the approved plan: {sorted(wanted - {s['step_id'] for s in plan})}. "
                                         "Use run_tool.py with --followup-reason for exploratory work."})
        raise SystemExit(1)

    def done(step):
        return (run_dir / "results" / f"{step['step_id'].replace('.', '_')}.json").exists()

    todo = [s for s in plan if (wanted and s["step_id"] in wanted) or (not wanted and (args.all or not done(s)))]
    todo.sort(key=lambda s: s["tool"] == "multiple-testing")
    if any(s["tool"] != "multiple-testing" for s in todo):
        todo += [s for s in plan if s["tool"] == "multiple-testing" and s not in todo]
    lines, errors = [], []
    for step in todo:
        try:
            r = execute(step["tool"], run_dir, step.get("params"), actor=args.actor, step=step)
            lines.append({"step": step["step_id"], "tool": step["tool"], "status": r["status"], "summary": r["summary"]})
        except Exception as exc:
            err = {"step_id": step["step_id"], "tool": step["tool"], "error": f"{type(exc).__name__}: {exc}"}
            errors.append(err)
            lines.append({"step": step["step_id"], "tool": step["tool"], "status": "error", "summary": err["error"]})
    index = write_analyses_index(run_dir, errors)
    events.emit(run_dir, "milestone", args.actor, f"analyses_index.json: {index['summary']}; {len(errors)} errors")
    ok = not errors and not index["missing_steps"]
    emit_json({"ok": ok, "ran": lines, "index_summary": index["summary"],
               "missing_steps": index["missing_steps"], "errors": errors})
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
