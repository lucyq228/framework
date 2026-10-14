"""Append-only run log shared by skill scripts, the manager, and Cursor hooks.

Every event goes to two files in the run folder:
- live_run.log  for people  (time | elapsed | EVENT | actor | message)
- events.jsonl  for machines (one JSON object per line)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _elapsed_seconds(run_dir: Path, now: datetime) -> float:
    try:
        started = datetime.fromisoformat(json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["started_at"])
        return (now - started).total_seconds()
    except Exception:
        return 0.0


def emit(run_dir: Path, event: str, actor: str, message: str, details: dict | None = None) -> dict:
    now = datetime.now().astimezone()
    elapsed = _elapsed_seconds(run_dir, now)
    safe_message = " ".join(str(message).split())
    line = f"{now.strftime('%H:%M:%S')} | +{elapsed:7.1f}s | {event.upper():<16} | {actor:<22} | {safe_message}"
    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "event": event,
        "actor": actor,
        "message": safe_message,
        "details": details or {},
    }
    with (run_dir / "live_run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record


def read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
