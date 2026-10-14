"""File-based run state: each milestone is proven by an artifact on disk.

Some milestones also go stale: feasibility when the scope or the data inventory
changes, review when analyses are re-run, insights when the review changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MILESTONES = [
    # (key, artifact, owner)
    ("baseline_eda", "baseline_eda.json", "a4-data-quality-eda"),
    ("scope_draft", "scope_proposal.json", "a2-scope"),
    ("feasibility", "feasibility_review.json", "a3-feasibility"),
    ("approval", "approved_scope.json", "manager (explicit user approval)"),
    ("data_quality", "dq_report.json", "a4-data-quality-eda"),
    ("analysis_datasets", "analysis_datasets.json", "a5-feature-creation"),
    ("descriptive_eda", "eda_descriptives.json", "a4-data-quality-eda"),
    ("analyses", "analyses_index.json", "a6-analysis-execution"),
    ("review", "review.json", "a7-review-synthesis"),
    ("insights", "insights.json", "a7-review-synthesis"),
    ("deliverables", "deliverables.json", "a7-review-synthesis"),
]
ARTIFACT = {k: a for k, a, _ in MILESTONES}
ORDER = [k for k, _, _ in MILESTONES]


def write_artifact(run_dir: Path, name: str, payload: dict) -> Path:
    path = run_dir / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_artifact(run_dir: Path, name: str, default=None):
    path = run_dir / name
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Required artifact is missing: {name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _stale_reason(run_dir: Path, key: str) -> str | None:
    from corr_core import sources

    errors = read_artifact(run_dir, ARTIFACT[key]).get("errors") if key != "approval" else None
    if errors:
        return f"{len(errors)} tool errors recorded: {[e.get('tool') or e.get('step_id') for e in errors]}"
    if key == "feasibility":
        feas = read_artifact(run_dir, "feasibility_review.json")
        if feas.get("scope_sha256") != sha(run_dir / "scope_proposal.json"):
            return "scope_proposal.json changed after the feasibility review"
        if not (run_dir / "approved_scope.json").exists() and feas.get("inventory_fingerprint") != sources.inventory()["fingerprint"]:
            return "data inventory changed (new data acquired) after the feasibility review"
    if key == "analyses":
        missing = read_artifact(run_dir, "analyses_index.json").get("missing_steps") or []
        if missing:
            return f"planned steps without a result: {missing}"
    if key == "review":
        rev = read_artifact(run_dir, "review.json")
        if rev.get("analyses_index_sha256") != sha(run_dir / "analyses_index.json"):
            return "analyses were re-run after the review"
    if key == "insights":
        ins = read_artifact(run_dir, "insights.json")
        if ins.get("review_sha256") != sha(run_dir / "review.json"):
            return "review changed after insights were written"
    if key == "deliverables":
        dl = read_artifact(run_dir, "deliverables.json")
        if dl.get("insights_sha256") != sha(run_dir / "insights.json"):
            return "insights changed after the deliverables were built"
    return None


def milestone_status(run_dir: Path) -> list[dict]:
    out = []
    for key, artifact, owner in MILESTONES:
        exists = (run_dir / artifact).exists()
        stale = _stale_reason(run_dir, key) if exists else None
        out.append({"milestone": key, "done": exists and not stale, "artifact": artifact, "owner": owner,
                    **({"stale": stale} if stale else {})})
    return out


def milestone_done(run_dir: Path, key: str) -> bool:
    return next(s["done"] for s in milestone_status(run_dir) if s["milestone"] == key)
