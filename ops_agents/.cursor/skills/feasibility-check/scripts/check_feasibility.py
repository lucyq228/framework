"""Deterministic feasibility matrix for the current scope proposal.

Matches every pathway step against tool_registry.yaml requirements and the data
inventory: sources, tool status, segments, controls, windows, CSAT channel,
grain, estimated n against min_n, repeat visits, and visits per period.
Writes feasibility_matrix.json. A3 adds its judgment with save_feasibility_review.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events, registry, scope as sc, sources  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402
from corr_core.state import read_artifact, sha, write_artifact  # noqa: E402


def main() -> None:
    configure_stdout()
    run_dir = resolve(base_parser(__doc__).parse_args())
    if (run_dir / sc.APPROVED_SCOPE).exists():
        emit_json({"ok": False, "error": "The scope is already approved; feasibility cannot change now."})
        raise SystemExit(1)
    proposal = read_artifact(run_dir, sc.SCOPE_PROPOSAL, {})
    if not proposal:
        emit_json({"ok": False, "error": "scope_proposal.json is missing; A2 must save a proposal first."})
        raise SystemExit(1)
    matrix = registry.check_scope(proposal)
    matrix["scope_sha256"] = sha(run_dir / sc.SCOPE_PROPOSAL)
    matrix["scope_version"] = proposal.get("version")
    write_artifact(run_dir, "feasibility_matrix.json", matrix)
    events.emit(run_dir, "feasibility matrix", "a3-feasibility",
                f"v{proposal.get('version')}: {matrix['counts']}; data requests: "
                f"{[r['source'] for r in matrix['data_requests']] or 'none'}")
    emit_json({"ok": True, "scope_version": proposal.get("version"), "counts": matrix["counts"],
               "facts": matrix["facts"],
               "hypotheses": [{k: h[k] for k in ("hypothesis", "verdict")} for h in matrix["hypotheses"]],
               "not_supported": [{k: s.get(k) for k in ("step_id", "tool", "verdict", "reasons", "estimated_n")}
                                 for s in matrix["steps"] if s["verdict"] != "supported"],
               "data_requests": matrix["data_requests"],
               "not_available_anywhere": [k for k, v in sources.inventory()["sources"].items()
                                          if not v["available"] and not v["acquirable"]]})


if __name__ == "__main__":
    main()
