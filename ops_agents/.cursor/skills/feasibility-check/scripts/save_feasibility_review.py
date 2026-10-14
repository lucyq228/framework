"""Merge A3's judgment into the feasibility matrix and save feasibility_review.json.

A3 may make a step's verdict stricter (supported -> partial -> blocked) with a
reason, never more lenient. The matrix must be fresh: built from the current
proposal and the current data inventory.

Judgment file fields: summary, can_do [str], cannot_do [str],
step_judgments [{step_id, verdict, reason}], extra_data_requests
[{need, owner, why, unlocks}], user_questions [{id, question, options,
recommended, why_it_matters}], recommendation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events, scope as sc, sources  # noqa: E402
from corr_core.registry import VERDICT_ORDER  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, read_json_file, resolve  # noqa: E402
from corr_core.state import read_artifact, sha, write_artifact  # noqa: E402

REQUIRED = ("summary", "can_do", "cannot_do", "recommendation")


def fail(msg) -> None:
    emit_json({"ok": False, "error": msg})
    raise SystemExit(1)


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--file", required=True, help="A3 judgment JSON.")
    args = parser.parse_args()
    run_dir = resolve(args)
    if (run_dir / sc.APPROVED_SCOPE).exists():
        fail("The scope is already approved; feasibility cannot change now.")
    matrix = read_artifact(run_dir, "feasibility_matrix.json", {})
    if not matrix:
        fail("Run check_feasibility.py first.")
    if matrix.get("scope_sha256") != sha(run_dir / sc.SCOPE_PROPOSAL):
        fail("The proposal changed after the matrix was built. Re-run check_feasibility.py.")
    inventory_fp = sources.inventory()["fingerprint"]
    if matrix.get("inventory_fingerprint") != inventory_fp:
        fail("The data inventory changed after the matrix was built. Re-run check_feasibility.py.")
    judgment = read_json_file(args.file)
    missing = [f for f in REQUIRED if not judgment.get(f)]
    if missing:
        fail(f"Judgment is missing {missing}.")

    steps = {s["step_id"]: dict(s) for s in matrix["steps"]}
    problems = []
    for j in judgment.get("step_judgments", []) or []:
        s = steps.get(j.get("step_id"))
        if s is None:
            problems.append(f"unknown step_id {j.get('step_id')!r}")
            continue
        if j.get("verdict") not in VERDICT_ORDER or not str(j.get("reason") or "").strip():
            problems.append(f"{j['step_id']}: verdict must be one of {list(VERDICT_ORDER)} with a reason")
            continue
        if VERDICT_ORDER[j["verdict"]] < VERDICT_ORDER[s["verdict"]]:
            problems.append(f"{j['step_id']}: cannot relax '{s['verdict']}' to '{j['verdict']}'; change the scope instead")
            continue
        if VERDICT_ORDER[j["verdict"]] > VERDICT_ORDER[s["verdict"]]:
            s["verdict"] = j["verdict"]
        s["reasons"] = s["reasons"] + [f"A3: {j['reason']}"]
    if problems:
        fail(problems)

    step_list = [steps[s["step_id"]] for s in matrix["steps"]]
    hyps = []
    for h in matrix["hypotheses"]:
        hs = [s for s in step_list if s["hypothesis"] == h["hypothesis"]]
        hyps.append({**h, "verdict": max((s["verdict"] for s in hs), key=VERDICT_ORDER.get),
                     "steps": {s["step_id"]: s["verdict"] for s in hs}})
    counts = {v: sum(1 for s in step_list if s["verdict"] == v) for v in VERDICT_ORDER}
    review = {**matrix, "steps": step_list, "hypotheses": hyps, "counts": counts,
              "inventory_fingerprint": inventory_fp, "summary": judgment["summary"],
              "user_questions": judgment.get("user_questions", []),
              "a3": {k: judgment.get(k) for k in ("can_do", "cannot_do", "extra_data_requests", "recommendation")}}
    write_artifact(run_dir, sc.FEASIBILITY, review)
    events.emit(run_dir, "feasibility", "a3-feasibility", f"{judgment['summary']} {counts}")
    emit_json({"ok": True, "counts": counts,
               "blocked_steps": [s["step_id"] for s in step_list if s["verdict"] == "blocked"],
               "data_requests": [r["source"] for r in review["data_requests"]],
               "open_user_questions": [q.get("id") for q in review["user_questions"]]})


if __name__ == "__main__":
    main()
