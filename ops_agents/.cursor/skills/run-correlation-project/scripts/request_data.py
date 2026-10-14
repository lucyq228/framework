"""Acquire a data source the user agreed to obtain (manager only, after user approval).

In this demo, acquisition copies the file from data/acquirable/ into
data/reference/. That changes the data inventory, so feasibility goes stale and
A3 must re-run it before the scope can be approved. Sources with no acquirable
file are logged as an open request for the named owner.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from corr_core import events, sources  # noqa: E402
from corr_core.runner import base_parser, configure_stdout, emit_json, resolve  # noqa: E402
from corr_core.state import read_artifact, write_artifact  # noqa: E402


def main() -> None:
    configure_stdout()
    parser = base_parser(__doc__)
    parser.add_argument("--source", required=True, choices=sorted(sources.SOURCES))
    parser.add_argument("--user-confirmation", required=True, help="The user's exact words agreeing to acquire it.")
    args = parser.parse_args()
    run_dir = resolve(args)
    if (run_dir / "approved_scope.json").exists():
        emit_json({"ok": False, "error": "The scope is already locked. New data needs a new run."})
        raise SystemExit(1)
    if not args.user_confirmation.strip():
        emit_json({"ok": False, "error": "--user-confirmation must quote the user."})
        raise SystemExit(1)
    meta = sources.SOURCES[args.source]
    log = read_artifact(run_dir, "data_requests_log.json", {"requests": []})
    if sources.available(args.source):
        emit_json({"ok": True, "source": args.source, "status": "already available"})
        return
    if sources.acquirable(args.source):
        dest = Path(meta["file"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(meta["acquirable_file"], dest)
        status = "acquired"
        message = f"Acquired {args.source} from {meta['owner']} into data/reference/. Feasibility is now stale."
    else:
        status = "requested"
        message = (f"{args.source} does not exist in this demo. Logged a request to {meta['owner']}; "
                   "steps that need it stay blocked.")
    log["requests"].append({"source": args.source, "status": status, "owner": meta["owner"],
                            "user_confirmation": args.user_confirmation})
    write_artifact(run_dir, "data_requests_log.json", log)
    events.emit(run_dir, "data request", "Manager", message, {"source": args.source, "status": status})
    emit_json({"ok": True, "source": args.source, "status": status, "message": message,
               **({"next": "Delegate a3-feasibility to re-run feasibility on the new inventory."} if status == "acquired" else {})})


if __name__ == "__main__":
    main()
