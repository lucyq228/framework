"""Validate A7's review and save review.json.

Review file fields: verdict (pass | pass_with_caveats | fail), summary,
hypothesis_verdicts [{id, verdict (supported | partially_supported |
not_supported | inconclusive), reason, evidence [result ids]}], challenges
[str], reconciliations [str], rerun_requests [{step_id or tool, params, reason}].
Every hypothesis needs a verdict; evidence must name existing results; the
review checks must be current. The review is tied to analyses_index.json by hash.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, read_json_file, resolve  # noqa: E402
from corr_core.scope import load_approved_scope  # noqa: E402
from corr_core.state import read_artifact, sha, write_artifact  # noqa: E402

VERDICTS = {"pass", "pass_with_caveats", "fail"}
H_VERDICTS = {"supported", "partially_supported", "not_supported", "inconclusive"}


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    run_dir = resolve(args)
    scope = load_approved_scope(run_dir)
    checks = read_artifact(run_dir, "review_checks.json", {})
    index_sha = sha(run_dir / "analyses_index.json")
    problems = []
    if not checks or checks.get("analyses_index_sha256") != index_sha:
        problems.append("review_checks.json is missing or older than the analyses; run review_checks.py first.")
    review = read_json_file(args.file)
    if review.get("verdict") not in VERDICTS:
        problems.append(f"verdict must be one of {sorted(VERDICTS)}")
    if not str(review.get("summary") or "").strip():
        problems.append("summary is required")
    results = {p.stem for p in (run_dir / "results").glob("*.json")}
    got = {}
    for hv in review.get("hypothesis_verdicts", []) or []:
        got[hv.get("id")] = hv
        if hv.get("verdict") not in H_VERDICTS:
            problems.append(f"{hv.get('id')}: verdict must be one of {sorted(H_VERDICTS)}")
        if not str(hv.get("reason") or "").strip():
            problems.append(f"{hv.get('id')}: reason is required")
        bad = [e for e in hv.get("evidence", []) or [] if e not in results]
        if bad:
            problems.append(f"{hv.get('id')}: unknown evidence {bad}")
    missing = [h["id"] for h in scope["hypotheses"] if h["id"] not in got]
    if missing:
        problems.append(f"missing hypothesis verdicts: {missing}")
    for rr in review.get("rerun_requests", []) or []:
        if not (rr.get("step_id") or rr.get("tool")) or not rr.get("reason"):
            problems.append("each rerun request needs step_id or tool, and a reason")
    if problems:
        emit_json({"ok": False, "issues": problems})
        raise SystemExit(1)
    review["analyses_index_sha256"] = index_sha
    write_artifact(run_dir, "review.json", review)
    events.emit(run_dir, "review", "a7-review-synthesis",
                f"verdict={review['verdict']}; " + ", ".join(f"{k}={v['verdict']}" for k, v in got.items())
                + (f"; {len(review.get('rerun_requests', []))} rerun requests" if review.get("rerun_requests") else ""))
    emit_json({"ok": True, "verdict": review["verdict"],
               "hypotheses": {k: v["verdict"] for k, v in got.items()},
               "rerun_requests": review.get("rerun_requests", [])})


if __name__ == "__main__":
    main()
