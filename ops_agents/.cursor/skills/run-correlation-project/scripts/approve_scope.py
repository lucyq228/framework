"""Lock the scope after the user's explicit approval (manager only).

Refuses when the feasibility review is stale (scope or data changed after it),
when blocked steps remain, or when the scope is invalid. Writes
approved_scope.json (with the flattened plan) and scope.lock.json (SHA-256).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402
from corr_core.scope import ScopeLockError, approve  # noqa: E402


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--user-confirmation", required=True, help="The user's exact words approving the scope.")
    args = parser.parse_args()
    run_dir = resolve(args)
    words = args.user_confirmation.strip()
    if len(words) < 2:
        emit_json({"ok": False, "error": "--user-confirmation must quote the user's approval."})
        raise SystemExit(1)
    try:
        lock = approve(run_dir, "user (via manager)", words)
    except ScopeLockError as exc:
        emit_json({"ok": False, "error": str(exc)})
        raise SystemExit(1)
    events.emit(run_dir, "approval", "Manager", f"Scope approved and locked ({lock['plan_steps']} plan steps): \"{words}\"",
                {"sha256": lock["sha256"]})
    emit_json({"ok": True, **lock, "next": "Delegate data_quality to a4-data-quality-eda."})


if __name__ == "__main__":
    main()
