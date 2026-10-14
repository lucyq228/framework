"""Run one shelf tool outside the plan as a recorded follow-up.

Analysis tools need --followup-reason and still pass the feasibility check;
the result is written to results/F_<label>.json and indexed as a follow-up.
Params come from --params '{"grain": "restaurant_period"}', --params-file, or
repeated --param key=value (values parsed as JSON when possible).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import registry  # noqa: E402
from corr_core.runner import (ToolBlocked, base_parser, compact, configure_stdout, emit_json, execute,  # noqa: E402
                              parse_params, resolve, write_analyses_index)
from corr_core.scope import ScopeLockError  # noqa: E402


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--label", help="Short label for the result file (default: the tool id).")
    parser.add_argument("--followup-reason", help="Why this exploratory run is needed (required for analysis tools).")
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    parser.add_argument("--param", action="append")
    args = parser.parse_args()
    run_dir = resolve(args)
    try:
        entry = registry.get(args.tool)
    except KeyError as exc:
        emit_json({"ok": False, "error": str(exc)})
        raise SystemExit(1)
    if entry["phase"] == "analysis" and not args.followup_reason:
        emit_json({"ok": False, "error": "Analysis tools outside the plan need --followup-reason."})
        raise SystemExit(1)
    label = None
    if entry["phase"] == "analysis":
        base = re.sub(r"[^A-Za-z0-9]+", "_", args.label or args.tool).strip("_")
        label, n = f"F_{base}", 2
        while (run_dir / "results" / f"{label}.json").exists():
            label, n = f"F_{base}_{n}", n + 1
    try:
        result = execute(args.tool, run_dir, parse_params(args), label=label, actor=args.actor,
                         followup_reason=args.followup_reason)
    except (ToolBlocked, ScopeLockError) as exc:
        emit_json({"ok": False, "blocked": True, "error": str(exc)})
        raise SystemExit(1)
    except Exception as exc:
        emit_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise SystemExit(1)
    out = {"ok": True, **compact(result)}
    if entry["phase"] == "analysis":
        out["result_file"] = f"results/{label}.json"
        out["index_summary"] = write_analyses_index(run_dir)["summary"]
    emit_json(out)


if __name__ == "__main__":
    main()
