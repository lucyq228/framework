"""Validate a drafted scope and save it as runs/<run>/scope_proposal.json.

The previous proposal, if any, is kept as scope_proposal.v<N>.json. Saving a
new version makes the feasibility review stale, so A3 must check it again.
Nothing is written when validation fails.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events, registry, scope as sc  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, read_json_file, resolve  # noqa: E402
from corr_core.state import write_artifact  # noqa: E402

QUESTION_FIELDS = ("id", "question", "options", "recommended", "why_it_matters")


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--file", required=True, help="Drafted scope JSON.")
    args = parser.parse_args()
    run_dir = resolve(args)
    if (run_dir / sc.APPROVED_SCOPE).exists():
        emit_json({"ok": False, "error": "This run's scope is already approved and locked. Start a new run."})
        raise SystemExit(1)
    proposal = read_json_file(args.file)
    issues = sc.validate_scope(proposal)
    for q in proposal.get("user_questions", []) or []:
        missing = [f for f in QUESTION_FIELDS if not q.get(f)]
        if missing:
            issues.append(f"user question {q.get('id')!r} is missing {missing}.")
    if issues:
        emit_json({"ok": False, "issues": issues})
        raise SystemExit(1)

    current = run_dir / sc.SCOPE_PROPOSAL
    version = 1
    if current.exists():
        version = len(list(run_dir.glob("scope_proposal.v*.json"))) + 2
        shutil.copy(current, run_dir / f"scope_proposal.v{version - 1}.json")
    proposal["version"] = version
    write_artifact(run_dir, sc.SCOPE_PROPOSAL, proposal)
    steps = registry.plan_steps(proposal)
    open_q = [q["id"] for q in proposal.get("user_questions", []) if not str(q.get("answer") or "").strip()]
    events.emit(run_dir, "scope draft", "a2-scope",
                f"Scope proposal v{version}: {len(proposal['hypotheses'])} hypotheses, {len(steps)} pathway steps, "
                f"{len(open_q)} open questions", {"version": version})
    emit_json({"ok": True, "version": version, "hypotheses": [h["id"] for h in proposal["hypotheses"]],
               "pathway_steps": len(steps), "open_user_questions": open_q,
               "next": "a3-feasibility checks this version."})


if __name__ == "__main__":
    main()
