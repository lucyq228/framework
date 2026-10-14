"""Record the user's answers to scope or feasibility questions (user_answers.json).

A2 reads these when it revises the proposal. Pass --answers as a JSON list of
{"id": ..., "answer": ..., "user_words": ...} or point --answers-file at one.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, read_json_file, resolve  # noqa: E402
from corr_core.state import read_artifact, write_artifact  # noqa: E402


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--answers", help="JSON list of answers.")
    g.add_argument("--answers-file", help="File with a JSON list of answers.")
    args = parser.parse_args()
    run_dir = resolve(args)
    if (run_dir / "approved_scope.json").exists():
        emit_json({"ok": False, "error": "The scope is already approved and locked; start a new run to change it."})
        raise SystemExit(1)
    if args.answers_file:
        new = read_json_file(args.answers_file)
    else:
        try:
            new = json.loads(args.answers)
        except json.JSONDecodeError as exc:
            emit_json({"ok": False, "error": f"--answers is not valid JSON: {exc}"})
            raise SystemExit(1)
    if not isinstance(new, list) or not all(isinstance(a, dict) and a.get("id") and "answer" in a for a in new):
        emit_json({"ok": False, "error": "Answers must be a JSON list of objects with 'id' and 'answer'."})
        raise SystemExit(1)
    doc = read_artifact(run_dir, "user_answers.json", {"answers": []})
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    doc["answers"] += [{**a, "recorded_at": stamp} for a in new]
    write_artifact(run_dir, "user_answers.json", doc)
    events.emit(run_dir, "user answers", "Manager", f"Recorded {len(new)} answers: {', '.join(a['id'] for a in new)}")
    emit_json({"ok": True, "recorded": len(new), "total_answers": len(doc["answers"]),
               "next": "Delegate to a2-scope to revise the proposal with these answers, then re-run a3-feasibility."})


if __name__ == "__main__":
    main()
