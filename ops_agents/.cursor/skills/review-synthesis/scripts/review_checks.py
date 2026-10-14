"""Deterministic challenge checks over every indexed result. Writes review_checks.json.

Per result: sample size against the scope minimum, confidence intervals that
include zero, Benjamini-Hochberg q from the run's multiple-testing step, tool
flags (for example Simpson's paradox), partial feasibility, and sign
consistency across a result's association estimates. Run level: missing
planned steps, tool errors, hypotheses with no result, and an FDR table that
is older than some result. A7 reads this before writing review.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402
from corr_core.scope import load_approved_scope  # noqa: E402
from corr_core.state import milestone_done, read_artifact, sha, write_artifact  # noqa: E402

ASSOCIATION = {"pearson_r", "spearman_r", "kendall_r", "kendall_tau", "weighted_r", "partial_r", "phi",
               "point_biserial_r", "disattenuated_r", "distance_corr"}


def main() -> None:
    configure_stdout()
    run_dir = resolve(base_parser(__doc__).parse_args())
    scope = load_approved_scope(run_dir)
    if not (run_dir / "analyses_index.json").exists():
        emit_json({"ok": False, "error": "analyses_index.json is missing; A6 must run the plan first."})
        raise SystemExit(1)
    index = read_artifact(run_dir, "analyses_index.json")
    min_n = scope["min_sample"]["min_n_correlation"]
    results = {r["result"]: json.loads((run_dir / "results" / f"{r['result']}.json").read_text(encoding="utf-8"))
               for r in index["results"]}

    fdr_res = [res for res in results.values() if res.get("tool") == "multiple-testing" and res.get("step_id")]
    fdr = fdr_res[-1] if fdr_res else None
    q_of = {(t["result"], t["label"]): t["q_bh"] for t in ((fdr or {}).get("tables") or {}).get("fdr", [])}
    run_issues = []
    if index.get("missing_steps"):
        run_issues.append(f"Planned steps without results: {index['missing_steps']}")
    if index.get("errors"):
        run_issues.append(f"Tool errors in the last plan run: {[e['step_id'] for e in index['errors']]}")
    if fdr is None:
        run_issues.append("No multiple-testing step in the plan: p-values are uncorrected.")
    else:
        newer = [k for k, r in results.items() if r.get("tool") != "multiple-testing"
                 and str(r.get("created_at")) > str(fdr.get("created_at"))]
        if newer:
            run_issues.append(f"FDR table is older than {newer}; re-run the multiple-testing step.")
    covered = {r.get("hypothesis") for r in results.values()}
    for h in scope["hypotheses"]:
        if h["id"] not in covered:
            run_issues.append(f"{h['id']} has no results.")

    per_result = []
    for key, res in results.items():
        issues, notes = [], []
        if res.get("status") == "partial" or res.get("feasibility") == "partial":
            notes.append("partial: " + "; ".join(res.get("notes", [])[:2]) if res.get("notes") else "partial by design")
        for f in res.get("flags", []) or []:
            issues.append(f"flag: {f}")
        signs = {}
        for f in res.get("findings", []) or []:
            if f.get("n") is not None and f["n"] < min_n:
                issues.append(f"{f['label']}: n={f['n']} below {min_n}")
            if f.get("ci_low") is not None and f.get("ci_high") is not None and f["ci_low"] <= 0 <= f["ci_high"] \
                    and f["metric"] in ASSOCIATION:
                issues.append(f"{f['label']}: CI [{f['ci_low']}, {f['ci_high']}] includes 0")
            q = q_of.get((key, f["label"]))
            if q is not None and q >= .05:
                issues.append(f"{f['label']}: q={q:.3f} does not survive FDR")
            if f["metric"] in ASSOCIATION and f.get("estimate") is not None and abs(f["estimate"]) >= .05:
                signs.setdefault(f.get("measure"), set()).add(f["estimate"] > 0)
        mixed = [m for m, s in signs.items() if len(s) > 1]
        if mixed:
            issues.append(f"direction differs across estimates for {mixed}")
        per_result.append({"result": key, "tool": res.get("tool"), "hypothesis": res.get("hypothesis"),
                           "kind": "planned" if res.get("step_id") else "follow-up", "summary": res.get("summary"),
                           "issues": issues, "notes": notes,
                           "n_findings": len(res.get("findings", []) or [])})

    payload = {"analyses_index_sha256": sha(run_dir / "analyses_index.json"), "run_issues": run_issues,
               "results": per_result, "descriptive_eda_available": milestone_done(run_dir, "descriptive_eda"),
               "summary": f"{len(per_result)} results checked; {sum(bool(r['issues']) for r in per_result)} with issues; "
                          f"{len(run_issues)} run-level issues"}
    write_artifact(run_dir, "review_checks.json", payload)
    emit_json({"ok": True, **payload})


if __name__ == "__main__":
    main()
