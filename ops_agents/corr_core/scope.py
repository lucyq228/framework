"""Scope proposal schema, validation, and the approval lock.

A2 writes scope_proposal.json; A3 checks it (feasibility_review.json); the
user approves; approve() writes approved_scope.json with the flattened plan
and a SHA-256 fingerprint in scope.lock.json. load_approved_scope() refuses to
return a scope whose file no longer matches the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

SCOPE_PROPOSAL = "scope_proposal.json"
FEASIBILITY = "feasibility_review.json"
APPROVED_SCOPE = "approved_scope.json"
SCOPE_LOCK = "scope.lock.json"

DEFAULT_DECISIONS = {
    "general_service_questions": "exclude",
    "local_core_products": "replace_unavailable",
    "proposed_mappings": "exclude",
}
DECISION_OPTIONS = {
    "general_service_questions": {"exclude", "include_as_in_restaurant"},
    "local_core_products": {"replace_unavailable", "exclude"},
    "proposed_mappings": {"include", "exclude"},
}
MARKETS = {"Canada", "United Kingdom", "Ireland"}
GRAINS = {"visit", "restaurant_period"}
PERIODS = {"year", "half", "quarter", "all"}
SCORING = {"progressive", "conservative"}
REQUIRED_TEXT = ("project_name", "business_context", "decision_supported")


class ScopeLockError(RuntimeError):
    pass


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_scope(scope: dict) -> list[str]:
    from corr_core import features, registry

    issues = []
    for f in REQUIRED_TEXT:
        if not str(scope.get(f) or "").strip():
            issues.append(f"{f} is required.")
    markets = scope.get("markets") or []
    if not markets or set(markets) - MARKETS:
        issues.append(f"markets must be a non-empty subset of {sorted(MARKETS)}.")
    period = scope.get("period") or {}
    if not period.get("start") or not period.get("end") or str(period["start"]) > str(period["end"]):
        issues.append("period.start and period.end (YYYY-MM-DD, start <= end) are required.")
    if scope.get("primary_scoring") not in SCORING:
        issues.append(f"primary_scoring must be one of {sorted(SCORING)}.")
    if scope.get("primary_grain") not in GRAINS:
        issues.append(f"primary_grain must be one of {sorted(GRAINS)}.")
    if scope.get("restaurant_period", "year") not in PERIODS:
        issues.append(f"restaurant_period must be one of {sorted(PERIODS)}.")
    if scope.get("primary_csat_window") not in features.WINDOWS:
        issues.append(f"primary_csat_window must be one of {list(features.WINDOWS)}.")
    if set(scope.get("visit_types", ["RGR", "CFV"])) - {"RGR", "CFV"}:
        issues.append("visit_types may only contain RGR and CFV.")
    decisions = scope.get("decisions") or {}
    for k, opts in DECISION_OPTIONS.items():
        if decisions.get(k) not in opts:
            issues.append(f"decisions.{k} must be one of {sorted(opts)}.")
    ms = scope.get("min_sample") or {}
    if not isinstance(ms.get("min_n_correlation"), int) or ms["min_n_correlation"] < 10:
        issues.append("min_sample.min_n_correlation must be an integer >= 10.")
    if not isinstance(ms.get("min_csat_responses"), int) or ms["min_csat_responses"] < 1:
        issues.append("min_sample.min_csat_responses must be a positive integer.")

    hyps = scope.get("hypotheses") or []
    if not hyps:
        issues.append("At least one hypothesis with a pathway is required.")
    seen = set()
    reg = registry.by_id()
    for h in hyps:
        hid = h.get("id")
        if not hid or hid in seen:
            issues.append(f"Hypothesis ids must be unique and non-empty (got {hid!r}).")
        seen.add(hid)
        for f in ("statement", "rationale", "support_criteria"):
            if not str(h.get(f) or "").strip():
                issues.append(f"{hid}: {f} is required.")
        if not h.get("pathway"):
            issues.append(f"{hid}: pathway must list at least one tool step.")
        for s in h.get("pathway") or []:
            entry = reg.get(s.get("tool"))
            if not entry:
                issues.append(f"{hid}: unknown tool '{s.get('tool')}'.")
                continue
            if entry["phase"] != "analysis":
                issues.append(f"{hid}: '{s['tool']}' is a {entry['phase']} tool; pathways use analysis tools only.")
            bad = set(s.get("params") or {}) - registry.allowed_params(entry)
            if bad:
                issues.append(f"{hid}: {s['tool']} does not accept params {sorted(bad)}.")
            if not str(s.get("purpose") or "").strip():
                issues.append(f"{hid}: every pathway step needs a purpose.")
    return issues


def approve(run_dir: Path, approved_by: str, confirmation: str) -> dict:
    from corr_core import registry, sources

    if (run_dir / APPROVED_SCOPE).exists():
        raise ScopeLockError("This run already has an approved scope. Start a new run to change it.")
    proposal_path = run_dir / SCOPE_PROPOSAL
    feas_path = run_dir / FEASIBILITY
    if not proposal_path.exists() or not feas_path.exists():
        raise ScopeLockError("scope_proposal.json and feasibility_review.json must both exist before approval.")
    scope = json.loads(proposal_path.read_text(encoding="utf-8"))
    feas = json.loads(feas_path.read_text(encoding="utf-8"))
    if feas.get("scope_sha256") != fingerprint(proposal_path):
        raise ScopeLockError("The scope changed after the feasibility review. Re-run feasibility on the current proposal.")
    if feas.get("inventory_fingerprint") != sources.inventory()["fingerprint"]:
        raise ScopeLockError("The data inventory changed after the feasibility review (new data acquired?). Re-run feasibility.")
    issues = validate_scope(scope)
    if issues:
        raise ScopeLockError("Scope is not valid: " + " ".join(issues))
    answers_path = run_dir / "user_answers.json"
    answered = {a["id"] for a in json.loads(answers_path.read_text(encoding="utf-8-sig"))["answers"]} \
        if answers_path.exists() else set()
    open_q = [q.get("id") for q in (scope.get("user_questions") or []) + (feas.get("user_questions") or [])
              if not str(q.get("answer") or "").strip() and q.get("id") not in answered]
    if open_q:
        raise ScopeLockError(f"Open user questions {open_q}: ask the user, record the answers, and have A2 revise the scope.")
    blocked = [s["step_id"] for s in feas["steps"] if s["verdict"] == "blocked"]
    if blocked:
        raise ScopeLockError(f"Blocked steps remain: {blocked}. Remove them, acquire the data, or change parameters, "
                             "then re-run feasibility.")
    verdicts = {s["step_id"]: s for s in feas["steps"]}
    plan = []
    for step in registry.plan_steps(scope):
        v = verdicts[step["step_id"]]
        plan.append({**step, "feasibility": v["verdict"], "feasibility_reasons": v["reasons"]})
    approved = {**scope, "plan": plan,
                "approval": {"approved_by": approved_by, "user_confirmation": confirmation,
                             "approved_at": datetime.now().astimezone().isoformat(timespec="seconds")}}
    path = run_dir / APPROVED_SCOPE
    path.write_text(json.dumps(approved, indent=2, default=str), encoding="utf-8")
    lock = {"sha256": fingerprint(path), "approved_at": approved["approval"]["approved_at"],
            "approved_by": approved_by, "inventory_fingerprint": feas["inventory_fingerprint"]}
    (run_dir / SCOPE_LOCK).write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return {"plan_steps": len(plan), "partial_steps": [p["step_id"] for p in plan if p["feasibility"] == "partial"], **lock}


def load_approved_scope(run_dir: Path) -> dict:
    path, lock_path = run_dir / APPROVED_SCOPE, run_dir / SCOPE_LOCK
    if not path.exists() or not lock_path.exists():
        raise ScopeLockError("No approved scope for this run. Finish scoping and get explicit user approval first.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if fingerprint(path) != lock.get("sha256"):
        raise ScopeLockError("approved_scope.json changed after approval. Start a new run instead of editing it.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_working_scope(run_dir: Path) -> dict | None:
    """Approved scope if it exists, otherwise the current proposal (used before approval)."""
    if (run_dir / APPROVED_SCOPE).exists():
        return load_approved_scope(run_dir)
    p = run_dir / SCOPE_PROPOSAL
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
