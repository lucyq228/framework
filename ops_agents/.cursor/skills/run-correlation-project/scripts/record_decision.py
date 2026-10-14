"""Record a manager decision in the run trace."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402

ACTIONS = ["delegate", "retry", "rerun", "ask_user", "acquire_data", "stop", "finish"]


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--agent", help="Subagent name when delegating, e.g. a2-scope.")
    parser.add_argument("--objective", default="", help="What should happen now.")
    parser.add_argument("--reason", required=True, help="Why this is the next useful step.")
    args = parser.parse_args()
    if args.action in {"delegate", "retry", "rerun"} and not args.agent:
        parser.error("--agent is required for delegate, retry, and rerun")
    run_dir = resolve(args)
    target = f" -> {args.agent}" if args.agent else ""
    record = events.emit(run_dir, "decision", "Manager",
                         f"{args.action}{target}: {args.objective or args.reason} (reason: {args.reason})",
                         {"action": args.action, "agent": args.agent, "objective": args.objective, "reason": args.reason})
    emit_json({"ok": True, "recorded": record})


if __name__ == "__main__":
    main()
