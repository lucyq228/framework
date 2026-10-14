"""Validate A7's selected insights and save insights.json.

Insights file fields: headline, executive_summary [str, at most 5],
insights [{id, title, tag (aligned | diverges | inconclusive), importance
(high | medium | low), robustness (high | medium | low), so_what,
evidence [{statement, result}]}], limitations [str], data_requests [str],
next_steps [str]. Evidence `result` is a result id (results/<id>.json) or
"eda:<tool-id>" for a descriptive EDA observation. Needs a current review
whose verdict is not fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, read_json_file, resolve  # noqa: E402
from corr_core.state import milestone_done, read_artifact, sha, write_artifact  # noqa: E402

TAGS = {"aligned", "diverges", "inconclusive"}
LEVELS = {"high", "medium", "low"}


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    run_dir = resolve(args)
    problems = []
    if not milestone_done(run_dir, "review"):
        problems.append("review.json is missing or stale; save a current review first.")
    review = read_artifact(run_dir, "review.json", {})
    if review.get("verdict") == "fail":
        problems.append("The review verdict is fail. Resolve the reruns or report the failure to the user; "
                        "do not publish insights.")
    ins = read_json_file(args.file)
    results = {p.stem for p in (run_dir / "results").glob("*.json")}
    eda_tools = {s["tool"] for s in read_artifact(run_dir, "eda_descriptives.json", {}).get("steps", [])}
    if not str(ins.get("headline") or "").strip():
        problems.append("headline is required")
    if not ins.get("executive_summary") or len(ins["executive_summary"]) > 5:
        problems.append("executive_summary needs 1-5 bullets")
    if not ins.get("insights"):
        problems.append("at least one insight is required")
    ids = set()
    for i in ins.get("insights", []) or []:
        iid = i.get("id")
        if not iid or iid in ids:
            problems.append(f"insight ids must be unique (got {iid!r})")
        ids.add(iid)
        for f in ("title", "so_what"):
            if not str(i.get(f) or "").strip():
                problems.append(f"{iid}: {f} is required")
        if i.get("tag") not in TAGS:
            problems.append(f"{iid}: tag must be one of {sorted(TAGS)}")
        for f in ("importance", "robustness"):
            if i.get(f) not in LEVELS:
                problems.append(f"{iid}: {f} must be one of {sorted(LEVELS)}")
        if not i.get("evidence"):
            problems.append(f"{iid}: evidence is required")
        for e in i.get("evidence", []) or []:
            ref = str(e.get("result") or "")
            ok = ref in results or (ref.startswith("eda:") and ref[4:] in eda_tools)
            if not ok or not str(e.get("statement") or "").strip():
                problems.append(f"{iid}: evidence needs a statement and an existing result (got {ref!r})")
    if problems:
        emit_json({"ok": False, "issues": problems})
        raise SystemExit(1)
    ins["review_sha256"] = sha(run_dir / "review.json")
    write_artifact(run_dir, "insights.json", ins)
    events.emit(run_dir, "insights", "a7-review-synthesis", f"{len(ins['insights'])} insights: {ins['headline']}")
    emit_json({"ok": True, "insights": [i["id"] for i in ins["insights"]], "headline": ins["headline"]})


if __name__ == "__main__":
    main()
