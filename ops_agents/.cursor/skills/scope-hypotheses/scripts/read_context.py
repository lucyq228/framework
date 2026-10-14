"""Gather what A2 needs before drafting a scope, without exposing raw rows.

Prints: the context files to read, baseline EDA observations and headline
numbers, which sources exist (and which could be acquired), the valid values
for scope fields, recorded user answers, and feedback on the previous proposal
(feasibility verdicts) if there is one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import features, registry, scope as sc, sources  # noqa: E402
from corr_core.paths import CONTEXT_DIR, PROJECT_ROOT, REGISTRY_PATH  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402
from corr_core.state import read_artifact  # noqa: E402


def main() -> None:
    configure_stdout()
    run_dir = resolve(base_parser(__doc__).parse_args())
    if (run_dir / sc.APPROVED_SCOPE).exists():
        emit_json({"ok": False, "error": "This run's scope is already approved and locked."})
        raise SystemExit(1)
    baseline = read_artifact(run_dir, "baseline_eda.json", {})
    if not baseline:
        emit_json({"ok": False, "error": "baseline_eda.json is missing; A4 must run baseline EDA first."})
        raise SystemExit(1)
    steps = {s["tool"]: s for s in baseline.get("steps", [])}
    context_text = run_dir / "context_text.md"
    files = {
        "read_first": [str(CONTEXT_DIR / "PACE_CSAT_project_brief.md"), str(CONTEXT_DIR / "data_dictionary.md")],
        "extracted_context": str(context_text) if context_text.exists() else None,
        "deck_image_previews": [str(p) for p in sorted((CONTEXT_DIR / "_jpg").glob("*.jpg"))
                                if not p.stem.startswith("data_")],
        "tool_registry": str(REGISTRY_PATH),
        "example_scope": str(PROJECT_ROOT / "examples" / "example_scope_proposal.json"),
    }
    inv = sources.inventory()["sources"]
    previous = read_artifact(run_dir, sc.SCOPE_PROPOSAL, {})
    feas = read_artifact(run_dir, sc.FEASIBILITY, {})
    emit_json({
        "ok": True,
        "files": files,
        "baseline_observations": steps.get("eda-summary-pack", {}).get("observations", []),
        "baseline_headlines": {t: s.get("summary") for t, s in steps.items() if t.startswith("eda-")},
        "sources": {k: {"available": v["available"], "acquirable_on_request": v["acquirable"],
                        "description": v["description"], "owner": v["owner"]} for k, v in inv.items()},
        "valid_values": {
            "markets": sorted(sc.MARKETS), "grains": sorted(sc.GRAINS), "restaurant_period": sorted(sc.PERIODS),
            "scoring": sorted(sc.SCORING), "csat_windows": list(features.WINDOWS),
            "csat_channels": features.CSAT_CHANNELS + ["matched (visit grain only)"],
            "decisions": {k: sorted(v) for k, v in sc.DECISION_OPTIONS.items()},
            "default_decisions": sc.DEFAULT_DECISIONS,
            "segments_always": sorted(registry.ALWAYS_SEGMENTS),
            "segments_need_restaurant_attributes": sorted(registry.ATTRIBUTE_SEGMENTS),
            "analysis_tools": [e["id"] for e in registry.catalogue(phase="analysis")],
        },
        "user_answers": read_artifact(run_dir, "user_answers.json", {"answers": []})["answers"],
        "previous_proposal": ({"hypotheses": [h["id"] for h in previous.get("hypotheses", [])],
                               "open_questions": [q.get("id") for q in previous.get("user_questions", [])
                                                  if not str(q.get("answer") or "").strip()]} if previous else None),
        "feasibility_feedback": ({"counts": feas.get("counts"),
                                  "not_supported": [{k: s[k] for k in ("step_id", "tool", "verdict", "reasons")}
                                                    for s in feas.get("steps", []) if s["verdict"] != "supported"],
                                  "data_requests": [r["source"] for r in feas.get("data_requests", [])],
                                  "a3_recommendation": (feas.get("a3") or {}).get("recommendation")} if feas else None),
    })


if __name__ == "__main__":
    main()
