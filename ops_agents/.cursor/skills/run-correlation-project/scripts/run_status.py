"""Show which milestones are done, what is stale, and who should act next.

Also surfaces what the manager must relay to the user: open questions from the
scope proposal and the feasibility review, data requests, the DQ verdict, and
the review verdict with any rerun requests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402
from corr_core.state import milestone_status, read_artifact  # noqa: E402


def _open_questions(run_dir: Path) -> list[dict]:
    if (run_dir / "approved_scope.json").exists():
        return []
    answered = {a["id"] for a in read_artifact(run_dir, "user_answers.json", {"answers": []})["answers"]}
    out = []
    for name, owner in (("scope_proposal.json", "a2-scope"), ("feasibility_review.json", "a3-feasibility")):
        doc = read_artifact(run_dir, name, {})
        for q in doc.get("user_questions", []) or []:
            if not str(q.get("answer") or "").strip() and q.get("id") not in answered:
                out.append({"from": owner, **{k: q.get(k) for k in ("id", "question", "options", "recommended", "why_it_matters")}})
    return out


def main() -> None:
    configure_stdout()
    args = base_parser(__doc__).parse_args()
    run_dir = resolve(args)
    status = milestone_status(run_dir)
    done = [s["milestone"] for s in status if s["done"]]
    missing = [s for s in status if not s["done"]]
    stale = {s["milestone"]: s["stale"] for s in status if s.get("stale")}

    nxt = None
    if missing:
        m = missing[0]
        if m["milestone"] == "approval":
            nxt = {"milestone": "approval", "action": "manager",
                   "instruction": "Present the scope proposal and the feasibility review to the user, resolve open "
                                  "questions and data requests, and lock the scope only after an explicit yes."}
        else:
            nxt = {"milestone": m["milestone"], "action": "delegate", "delegate_to": m["owner"],
                   **({"why": m["stale"]} if m.get("stale") else {})}

    feas = read_artifact(run_dir, "feasibility_review.json", {})
    dq = read_artifact(run_dir, "dq_report.json", {})
    review = read_artifact(run_dir, "review.json", {})
    answers = read_artifact(run_dir, "user_answers.json", {"answers": []})["answers"]
    pending = {
        "open_user_questions": _open_questions(run_dir),
        "recorded_user_answers": len(answers),
        "data_requests": [{k: r.get(k) for k in ("source", "description", "owner", "acquirable_now", "unlocks_steps")}
                          for r in feas.get("data_requests", [])],
        "feasibility_counts": feas.get("counts"),
        "dq_overall": next((s.get("overall") for s in dq.get("steps", []) if s.get("tool") == "data-quality-check"), None),
        "review_verdict": review.get("verdict"),
        "rerun_requests": review.get("rerun_requests", []),
    }
    emit_json({"ok": True, "run_id": run_dir.name, "completed": done, "missing": [m["milestone"] for m in missing],
               "stale": stale, "next": nxt, "pending": {k: v for k, v in pending.items() if v not in (None, [], {})}})


if __name__ == "__main__":
    main()
