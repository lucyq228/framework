"""Data-quality checks under the locked scope's decisions. Writes dq_report.json.

Overall status is the worst check: pass < info < warn < fail. A fail stops
feature creation unless the manager records an override reason with the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core.runner import base_parser, configure_stdout, emit_json, resolve, run_milestone  # noqa: E402
from corr_core.state import milestone_done  # noqa: E402


def main() -> None:
    configure_stdout()
    args = base_parser(__doc__).parse_args()
    run_dir = resolve(args)
    if not milestone_done(run_dir, "approval"):
        emit_json({"ok": False, "error": "No approved scope yet. Data quality runs after the user approves the scope."})
        raise SystemExit(1)
    out = run_milestone(run_dir, "dq_report.json", args.actor, ["data-quality-check"])
    dq = out["steps"][0] if out["steps"] else {}
    emit_json({"ok": not out["errors"], "overall": dq.get("overall"), "summary": dq.get("summary"),
               "checks": [{k: c.get(k) for k in ("check", "status", "count", "rate", "handling")}
                          for c in dq.get("checks", []) if c["status"] != "pass"],
               "errors": out["errors"]})
    raise SystemExit(1 if out["errors"] else 0)


if __name__ == "__main__":
    main()
