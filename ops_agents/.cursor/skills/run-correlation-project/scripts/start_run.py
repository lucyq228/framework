"""Create a run folder, point runs/CURRENT at it, and print the data inventory.

Writes runs/<run-id>/run.json. The context documents stay in context/; A2 reads
them through the scope-hypotheses skill.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events, sources  # noqa: E402
from corr_core.paths import CONTEXT_DIR, new_run_dir, set_current_run  # noqa: E402
from corr_core.runner import configure_stdout, emit_json  # noqa: E402
from corr_core.state import write_artifact  # noqa: E402


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="pace-csat", help="Short run label, e.g. 'pace-csat'.")
    parser.add_argument("--request", default="", help="The user's request in their own words.")
    args = parser.parse_args()

    label = re.sub(r"[^a-z0-9]+", "-", args.label.lower()).strip("-")[:40] or "pace-csat"
    run_dir = new_run_dir(label)
    inv = sources.inventory()
    write_artifact(run_dir, "run.json", {"run_id": run_dir.name, "label": label, "user_request": args.request,
                                         "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                         "inventory_at_start": inv})
    set_current_run(run_dir)
    events.emit(run_dir, "start", "Manager", f"Started run {run_dir.name}.", {"request": args.request})
    docs = sorted(p.name for p in CONTEXT_DIR.iterdir() if p.is_file())
    emit_json({"ok": True, "run_id": run_dir.name, "run_dir": str(run_dir), "context_documents": docs,
               "available_sources": [k for k, v in inv["sources"].items() if v["available"]],
               "not_available": {k: v["description"] for k, v in inv["sources"].items() if not v["available"]},
               "next": "Delegate baseline_eda to a4-data-quality-eda."})


if __name__ == "__main__":
    main()
