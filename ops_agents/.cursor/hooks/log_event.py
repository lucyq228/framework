"""Cursor hook: copy subagent and shell activity into the active run's log.

Registered in .cursor/hooks.json for subagentStart, subagentStop, and
postToolUse (Shell). Uses only the standard library so it works with any
Python on PATH. It never blocks the agent: subagentStart always answers
{"permission": "allow"}, and every failure is swallowed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MAX_TEXT = 400


def _short(value, limit: int = MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _exit_code(tool_output) -> int | None:
    try:
        return json.loads(tool_output).get("exitCode")
    except Exception:
        return None


def _record(payload: dict) -> None:
    from corr_core import events
    from corr_core.paths import current_run_dir

    run_dir = current_run_dir()
    if run_dir is None:
        return
    name = payload.get("hook_event_name", "")
    if name == "subagentStart":
        agent = payload.get("subagent_type") or "subagent"
        events.emit(run_dir, "subagent start", agent,
                    f"model={payload.get('subagent_model') or 'inherit'}; task={_short(payload.get('task'), 240)}",
                    {"subagent_id": payload.get("subagent_id"), "model": payload.get("subagent_model")})
    elif name == "subagentStop":
        agent = payload.get("subagent_type") or "subagent"
        seconds = (payload.get("duration_ms") or 0) / 1000
        events.emit(run_dir, "subagent stop", agent,
                    f"status={payload.get('status')}; {seconds:.0f}s; tool calls={payload.get('tool_call_count')}; "
                    f"summary={_short(payload.get('summary'), 300)}",
                    {"status": payload.get("status"), "modified_files": payload.get("modified_files")})
    elif name == "postToolUse" and payload.get("tool_name") == "Shell":
        command = (payload.get("tool_input") or {}).get("command", "")
        if ".cursor" not in command and "corr_core" not in command:
            return
        events.emit(run_dir, "shell", "agent", f"exit={_exit_code(payload.get('tool_output'))}; {_short(command, 300)}",
                    {"duration_ms": payload.get("duration")})


def main() -> None:
    name = ""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        name = payload.get("hook_event_name", "")
        _record(payload)
    except Exception:
        pass
    print(json.dumps({"permission": "allow"} if name == "subagentStart" else {}))


if __name__ == "__main__":
    main()
