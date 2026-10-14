"""Project locations and the active run folder."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = PROJECT_ROOT / "context"
DATA_DIR = Path(os.environ.get("CORR_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
ACQUIRABLE_DIR = DATA_DIR / "acquirable"
REGISTRY_PATH = PROJECT_ROOT / "tool_registry.yaml"
# Tests point this elsewhere so they never touch real runs or runs/CURRENT.
RUNS_DIR = Path(os.environ.get("CORR_RUNS_DIR", PROJECT_ROOT / "runs"))
CURRENT_POINTER = RUNS_DIR / "CURRENT"


def new_run_dir(label: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / (f"{stamp}-{label}" if label else stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "results").mkdir()
    (run_dir / "datasets").mkdir()
    (run_dir / "charts").mkdir()
    (run_dir / "drafts").mkdir()
    return run_dir


def set_current_run(run_dir: Path) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_POINTER.write_text(run_dir.name, encoding="utf-8")


def current_run_dir() -> Path | None:
    if not CURRENT_POINTER.exists():
        return None
    name = CURRENT_POINTER.read_text(encoding="utf-8").strip()
    run_dir = RUNS_DIR / name
    return run_dir if name and run_dir.is_dir() else None


def resolve_run_dir(explicit: str | None) -> Path:
    """Use --run when given, otherwise the run named in runs/CURRENT."""
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            by_name = RUNS_DIR / explicit
            candidate = by_name if by_name.is_dir() else (Path.cwd() / candidate)
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Run folder not found: {candidate}")
        return candidate
    run_dir = current_run_dir()
    if run_dir is None:
        raise FileNotFoundError("No active run. Start one with run-correlation-project (start_run.py).")
    return run_dir
