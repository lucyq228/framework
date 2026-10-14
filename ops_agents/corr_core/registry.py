"""Tool registry (tool_registry.yaml) and the feasibility matcher.

check_scope() compares every pathway step of a scope proposal with the data
inventory and the sample sizes the scope implies, and returns one verdict per
step: supported, partial (runs with a stated compromise), or blocked.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yaml

from corr_core import features, sources
from corr_core.paths import REGISTRY_PATH

VERDICT_ORDER = {"supported": 0, "partial": 1, "blocked": 2}
ATTRIBUTE_SEGMENTS = {"region", "restaurant_type", "ownership_type", "volume_band", "has_drive_thru",
                      "has_table_service", "has_mccafe"}
ALWAYS_SEGMENTS = {"market", "market_group", "survey_name", "channel_tested", "csat_channel"}


@lru_cache(maxsize=1)
def _load() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def entries() -> list[dict]:
    return _load()["tools"]


def by_id() -> dict[str, dict]:
    return {e["id"]: e for e in entries()}


def get(tool_id: str) -> dict:
    reg = by_id()
    if tool_id not in reg:
        raise KeyError(f"Unknown tool '{tool_id}'. List tools with list_tools.py.")
    return reg[tool_id]


def allowed_params(entry: dict) -> set[str]:
    common = set()
    if entry["phase"] == "analysis" and entry.get("common_params", True):
        common = set(_load()["common_analysis_params"]) - ({"grain"} if entry.get("fixed_grain") else set())
    return common | set(entry.get("params") or [])


def scope_facts(scope: dict) -> dict:
    """Sample sizes implied by a scope (computed from the raw sources with the scope's decisions)."""
    visits, _ = features.build_visit_scores(scope.get("decisions", {}))
    visits, _ = features.filter_scope(visits, scope)
    per_rest = visits.groupby("glin").size()
    facts = {
        "visits": int(len(visits)),
        "restaurants": int(visits["glin"].nunique()),
        "visits_by_market_group": visits.groupby("market_group").size().to_dict(),
        "visits_by_market": visits.groupby("market").size().to_dict(),
        "visits_by_type": visits.groupby("survey_name").size().to_dict(),
        "share_restaurants_with_repeat_visits": round(float((per_rest >= 2).mean()), 3) if len(per_rest) else 0.0,
        "restaurant_periods": {},
        "mean_visits_per_restaurant_period": {},
    }
    for p in features.PERIODS:
        lab = features.period_label(visits["visit_date"], p)
        grp = visits.assign(_p=lab).groupby(["glin", "_p"]).size()
        facts["restaurant_periods"][p] = int(len(grp))
        facts["mean_visits_per_restaurant_period"][p] = round(float(grp.mean()), 2) if len(grp) else 0.0
    return facts


def _estimated_n(step_params: dict, scope: dict, facts: dict, visits_by_type_share: float) -> int:
    grain = step_params.get("grain") or scope.get("primary_grain", "visit")
    if grain == "restaurant_period":
        period = step_params.get("period") or scope.get("restaurant_period", "year")
        n = facts["restaurant_periods"].get(period, 0)
    else:
        n = facts["visits"]
    if step_params.get("market_group"):
        n = int(n * facts["visits_by_market_group"].get(step_params["market_group"], 0) / max(facts["visits"], 1))
    if step_params.get("market"):
        n = int(n * facts["visits_by_market"].get(step_params["market"], 0) / max(facts["visits"], 1))
    if step_params.get("visit_types"):
        share = sum(facts["visits_by_type"].get(t, 0) for t in step_params["visit_types"]) / max(facts["visits"], 1)
        n = int(n * share)
    return n


def check_step(step: dict, scope: dict, facts: dict) -> dict:
    tool_id = step["tool"]
    params = step.get("params") or {}
    out = {"step_id": step.get("step_id"), "hypothesis": step.get("hypothesis"), "tool": tool_id,
           "verdict": "supported", "reasons": [], "data_requests": []}

    def worsen(verdict: str, reason: str):
        if VERDICT_ORDER[verdict] > VERDICT_ORDER[out["verdict"]]:
            out["verdict"] = verdict
        out["reasons"].append(reason)

    try:
        entry = get(tool_id)
    except KeyError as exc:
        worsen("blocked", str(exc))
        return out
    if entry["phase"] != "analysis":
        worsen("blocked", f"{tool_id} is a {entry['phase']} tool; pathways may only use analysis tools")
        return out
    unknown = set(params) - allowed_params(entry)
    if unknown:
        worsen("blocked", f"Unknown params for {tool_id}: {sorted(unknown)}")

    req = entry.get("requires") or {}
    for src in req.get("sources") or []:
        if not sources.available(src):
            worsen("blocked", f"needs source '{src}', which is not available")
            out["data_requests"].append(src)
    if entry.get("status") == "blocked" and out["verdict"] != "blocked":
        worsen("blocked", "tool is registered as blocked")
    if entry.get("status") == "partial":
        worsen("partial", entry.get("limitation", "partial by design"))
        out["data_requests"] += [s for s in entry.get("unlocks_with", []) if not sources.available(s)]

    optional = req.get("optional") or {}
    seg_key = entry.get("segment_param")
    if seg_key and params.get(seg_key):
        wanted = set(params[seg_key])
        missing = {s for s in wanted if s in ATTRIBUTE_SEGMENTS and not sources.available("restaurant_attributes")}
        invalid = wanted - ATTRIBUTE_SEGMENTS - ALWAYS_SEGMENTS
        if invalid:
            worsen("blocked", f"unknown segments {sorted(invalid)}")
        if missing:
            worsen("blocked" if missing == wanted else "partial",
                   f"segments {sorted(missing)} need restaurant_attributes")
            out["data_requests"].append("restaurant_attributes")
    if params.get("controls"):
        missing = {c for c in params["controls"] if c in ATTRIBUTE_SEGMENTS and not sources.available("restaurant_attributes")}
        if missing:
            worsen("partial", f"controls {sorted(missing)} need restaurant_attributes and will be skipped")
            out["data_requests"].append("restaurant_attributes")
    for src, effect in optional.items():
        if not sources.available(src):
            out["reasons"].append(f"optional source '{src}' not available: {effect}")

    window = params.get("window")
    if window and window not in features.WINDOWS:
        worsen("blocked", f"unknown window '{window}'")
    for w in params.get("windows") or []:
        if w not in features.WINDOWS:
            worsen("blocked", f"unknown window '{w}'")
    channel = params.get("csat_channel")
    grain = params.get("grain") or scope.get("primary_grain", "visit")
    if channel and channel not in features.CSAT_CHANNELS + ["matched"]:
        worsen("blocked", f"unknown csat_channel '{channel}'")
    if channel == "matched" and grain != "visit":
        worsen("blocked", "csat_channel=matched only exists at visit grain")
    if params.get("grain") and params["grain"] not in ("visit", "restaurant_period"):
        worsen("blocked", f"unknown grain '{params['grain']}'")

    n = _estimated_n(params, scope, facts, 1.0)
    min_n = int(params.get("min_n") or req.get("min_n") or scope.get("min_sample", {}).get("min_n_correlation", 30))
    out["estimated_n"] = n
    if req.get("min_n") or entry["family"] != "Reporting":
        if n < min_n:
            worsen("blocked", f"estimated n={n} below the minimum {min_n}")
        elif n < 2 * min_n:
            worsen("partial", f"estimated n={n} is under 2x the minimum {min_n}: low power")
    if req.get("needs_repeat_visits") and facts["share_restaurants_with_repeat_visits"] < .3:
        worsen("partial", f"only {facts['share_restaurants_with_repeat_visits']:.0%} of restaurants have repeat visits")
    if grain == "restaurant_period":
        period = params.get("period") or scope.get("restaurant_period", "year")
        mv = facts["mean_visits_per_restaurant_period"].get(period, 0)
        if mv < 1.5:
            worsen("partial", f"restaurant-{period} rows average {mv} visits: most periods rest on a single visit")
    out["data_requests"] = sorted(set(out["data_requests"]))
    return out


def plan_steps(scope: dict) -> list[dict]:
    steps = []
    for h in scope.get("hypotheses", []):
        for i, s in enumerate(h.get("pathway", []), 1):
            steps.append({"step_id": f"{h['id']}.{i}", "hypothesis": h["id"], "tool": s["tool"],
                          "params": s.get("params") or {}, "purpose": s.get("purpose", "")})
    return steps


def check_scope(scope: dict) -> dict:
    facts = scope_facts(scope)
    steps = [check_step(s, scope, facts) for s in plan_steps(scope)]
    hyps = []
    for h in scope.get("hypotheses", []):
        hs = [s for s in steps if s["hypothesis"] == h["id"]]
        worst = max((s["verdict"] for s in hs), key=VERDICT_ORDER.get, default="blocked")
        hyps.append({"hypothesis": h["id"], "statement": h.get("statement"), "verdict": worst,
                     "steps": {s["step_id"]: s["verdict"] for s in hs}})
    requests = {}
    for s in steps:
        for src in s["data_requests"]:
            meta = sources.SOURCES.get(src, {})
            r = requests.setdefault(src, {"source": src, "description": meta.get("description"), "owner": meta.get("owner"),
                                          "acquirable_now": sources.acquirable(src), "unlocks_steps": []})
            r["unlocks_steps"].append(s["step_id"])
    counts = {v: sum(1 for s in steps if s["verdict"] == v) for v in VERDICT_ORDER}
    return {"facts": facts, "steps": steps, "hypotheses": hyps, "data_requests": list(requests.values()),
            "counts": counts, "inventory_fingerprint": sources.inventory()["fingerprint"]}


def catalogue(family: str | None = None, phase: str | None = None) -> list[dict]:
    out = []
    for e in entries():
        if family and family.lower() not in e["family"].lower():
            continue
        if phase and e["phase"] != phase:
            continue
        out.append(e)
    return out


def shelf_status() -> pd.DataFrame:
    rows = []
    for e in entries():
        missing = [s for s in (e.get("requires") or {}).get("sources") or [] if not sources.available(s)]
        rows.append({"id": e["id"], "family": e["family"], "phase": e["phase"], "status": e["status"],
                     "missing_sources": ", ".join(missing)})
    return pd.DataFrame(rows)
