"""Browse the tool shelf (tool_registry.yaml) with availability against current data.

--id shows one tool in full. Otherwise prints one line per tool, filtered by
--family (substring) and --phase. --available-only hides tools whose sources
are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import registry, sources  # noqa: E402
from corr_core.runner import configure_stdout, emit_json  # noqa: E402


def main() -> None:
    configure_stdout()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id")
    p.add_argument("--family")
    p.add_argument("--phase", choices=["intake", "eda", "quality", "feature", "analysis", "reporting"])
    p.add_argument("--available-only", action="store_true")
    args = p.parse_args()
    if args.id:
        try:
            e = registry.get(args.id)
        except KeyError as exc:
            emit_json({"ok": False, "error": str(exc)})
            raise SystemExit(1)
        emit_json({"ok": True, **e, "accepted_params": sorted(registry.allowed_params(e)),
                   "missing_sources": [s for s in (e.get("requires") or {}).get("sources") or [] if not sources.available(s)]})
        return
    rows = []
    for e in registry.catalogue(args.family, args.phase):
        missing = [s for s in (e.get("requires") or {}).get("sources") or [] if not sources.available(s)]
        if args.available_only and (missing or e["status"] == "blocked"):
            continue
        rows.append({"id": e["id"], "family": e["family"], "phase": e["phase"], "status": e["status"],
                     "question": e["question"], **({"missing_sources": missing} if missing else {}),
                     **({"fixed_grain": True} if e.get("fixed_grain") else {})})
    emit_json({"ok": True, "count": len(rows), "tools": rows})


if __name__ == "__main__":
    main()
