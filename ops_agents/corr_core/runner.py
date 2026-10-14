"""Guarded execution of shelf tools. Every script calls execute() or cli_main().

Guardrails, by phase:
- intake / eda (baseline): need a run; no scope required.
- quality: approved scope if present, otherwise default decisions.
- feature: approved, fingerprint-checked scope.
- analysis: approved scope + analysis_datasets; the step must be in the approved
  plan, or be an explicit follow-up whose feasibility is not blocked.
- reporting: approved scope + insights.json.
Only compact JSON summaries are printed, never raw rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from corr_core import events, registry, tools
from corr_core.datasets import Ctx
from corr_core.paths import resolve_run_dir
from corr_core.scope import ScopeLockError, load_approved_scope
from corr_core.state import milestone_done
from corr_core.tools.common import ToolContext


class ToolBlocked(RuntimeError):
    pass


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def emit_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _scope_for(entry: dict, run_dir: Path) -> dict | None:
    phase = entry["phase"]
    if phase in ("intake",):
        return None
    if phase in ("eda", "quality"):
        try:
            return load_approved_scope(run_dir)
        except ScopeLockError:
            return None
    return load_approved_scope(run_dir)


def execute(tool_id: str, run_dir: Path, params: dict | None = None, label: str | None = None, actor: str = "agent",
            step: dict | None = None, followup_reason: str | None = None, write: bool = True) -> dict:
    entry = registry.get(tool_id)
    params = dict(params or {})
    unknown = set(params) - registry.allowed_params(entry)
    if unknown:
        raise ToolBlocked(f"{tool_id} does not accept params {sorted(unknown)}")
    scope = _scope_for(entry, run_dir)
    phase = entry["phase"]
    data = None
    if phase == "analysis":
        if not milestone_done(run_dir, "analysis_datasets"):
            raise ToolBlocked("analysis_datasets milestone is missing; A5 must build the datasets first.")
        if step is None:
            if not followup_reason:
                raise ToolBlocked(f"{tool_id} is not a step of the approved plan. Pass --followup-reason to run it as a "
                                  "recorded exploratory follow-up.")
            check = registry.check_step({"tool": tool_id, "params": params, "step_id": label}, scope,
                                        registry.scope_facts(scope))
            if check["verdict"] == "blocked":
                raise ToolBlocked(f"Follow-up is blocked by feasibility: {'; '.join(check['reasons'])}")
        data = Ctx(run_dir, scope)
    elif phase == "feature":
        data = Ctx(run_dir, scope)
        if tool_id != "build-analysis-dataset" and not (run_dir / "datasets" / "visit_scores.parquet").exists():
            raise ToolBlocked("Run build-analysis-dataset first.")
    elif phase == "eda" and scope is not None and (run_dir / "datasets" / "visit_scores.parquet").exists():
        data = Ctx(run_dir, scope)
    elif phase == "reporting":
        if not (run_dir / "insights.json").exists():
            raise ToolBlocked("insights.json is missing; A7 must save insights first.")

    label = label or (step or {}).get("step_id") or tool_id
    tc = ToolContext(run_dir=run_dir, scope=scope, entry=entry, label=label.replace(".", "_"), data=data)
    events.emit(run_dir, "tool start", actor, f"{tool_id} [{label}]", {"tool": tool_id, "params": params})
    try:
        result = tools.get(tool_id)(tc, params)
    except Exception as exc:
        events.emit(run_dir, "tool error", actor, f"{tool_id} [{label}]: {type(exc).__name__}: {exc}",
                    {"tool": tool_id, "traceback": traceback.format_exc(limit=4)})
        raise
    result = {"tool": tool_id, "family": entry["family"], "phase": phase, "label": label, "status": result.pop("status", "ok"),
              "params": params, **({"step_id": step["step_id"], "hypothesis": step["hypothesis"], "purpose": step.get("purpose"),
                                    "feasibility": step.get("feasibility")} if step else {}),
              **({"followup_reason": followup_reason} if followup_reason else {}),
              "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "guardrails": entry.get("guardrails", []), **result}
    if write and phase == "analysis":
        (run_dir / "results" / f"{tc.label}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    events.emit(run_dir, "tool done", actor, f"{tool_id} [{label}]: {result['summary']}", {"tool": tool_id, "status": result["status"]})
    return result


def write_analyses_index(run_dir: Path, errors: list[dict] | None = None) -> dict:
    """Index every result file; planned steps without a result are listed as missing."""
    from corr_core.state import write_artifact

    scope = load_approved_scope(run_dir)
    rows = []
    for path in sorted((run_dir / "results").glob("*.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"result": path.stem, "tool": res.get("tool"), "step_id": res.get("step_id"),
                     "hypothesis": res.get("hypothesis"), "status": res.get("status"),
                     "kind": "planned" if res.get("step_id") else "follow-up",
                     **({"followup_reason": res["followup_reason"]} if res.get("followup_reason") else {}),
                     "summary": res.get("summary")})
    present = {r["result"] for r in rows}
    missing = [s["step_id"] for s in scope["plan"] if s["step_id"].replace(".", "_") not in present]
    payload = {"summary": f"{len(rows)} results ({sum(r['kind'] == 'planned' for r in rows)} planned, "
                          f"{sum(r['kind'] == 'follow-up' for r in rows)} follow-up); {len(missing)} planned steps missing",
               "results": rows, "missing_steps": missing, "errors": errors or []}
    write_artifact(run_dir, "analyses_index.json", payload)
    return payload


def compact(result: dict) -> dict:
    keep = ("tool", "label", "status", "summary", "notes", "flags", "observations", "charts", "file", "files")
    return {k: result[k] for k in keep if k in result}


def read_json_file(path: str | Path):
    """Read a JSON file an agent wrote; tolerates a UTF-8 BOM. Exits with a JSON error if unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        emit_json({"ok": False, "error": f"File not found: {path}"})
    except json.JSONDecodeError as exc:
        emit_json({"ok": False, "error": f"{path} is not valid JSON: {exc}"})
    raise SystemExit(1)


def parse_params(args) -> dict:
    params = {}
    if getattr(args, "params_file", None):
        params.update(read_json_file(args.params_file))
    if getattr(args, "params", None):
        params.update(json.loads(args.params))
    for item in getattr(args, "param", None) or []:
        key, _, raw = item.partition("=")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run", help="Run folder name or path. Defaults to runs/CURRENT.")
    p.add_argument("--actor", default="agent", help="Agent name recorded in the run log.")
    return p


def resolve(args) -> Path:
    try:
        return resolve_run_dir(args.run)
    except FileNotFoundError as exc:
        emit_json({"ok": False, "error": str(exc)})
        raise SystemExit(1)


def run_milestone(run_dir: Path, artifact: str, actor: str, tool_ids: list[str], extra: dict | None = None,
                  params: dict | None = None) -> dict:
    """Run a list of tools and write their combined results as one milestone artifact."""
    from corr_core.state import write_artifact

    steps, errors = [], []
    for tid in tool_ids:
        try:
            steps.append(execute(tid, run_dir, (params or {}).get(tid), actor=actor, write=False))
        except Exception as exc:  # keep going so one failure does not hide the rest
            errors.append({"tool": tid, "error": f"{type(exc).__name__}: {exc}"})
    payload = {"summary": f"{len(steps)} tools ran, {len(errors)} errors", "steps": steps, "errors": errors, **(extra or {})}
    write_artifact(run_dir, artifact, payload)
    events.emit(run_dir, "milestone", actor, f"{artifact}: {payload['summary']}")
    return payload
