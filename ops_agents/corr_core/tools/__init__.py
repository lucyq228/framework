"""The correlation tool shelf.

Every tool is a function `fn(tc: ToolContext, params: dict) -> dict` registered
under the id used in tool_registry.yaml. The result dict always has `summary`
and may have `findings` (standard testable statistics), `tables`, `charts`,
and `notes`. corr_core.runner wraps every call with the guardrails.
"""

from __future__ import annotations

import importlib

TOOLS: dict = {}
_MODULES = ("platform", "eda", "feature", "distributions", "association", "components", "panel",
            "segmentation", "temporal", "consistency", "nonlinear", "regression", "importance",
            "reporting", "blocked")


def tool(tool_id: str):
    def register(fn):
        TOOLS[tool_id] = fn
        return fn
    return register


def get(tool_id: str):
    load_all()
    if tool_id not in TOOLS:
        raise KeyError(f"No implementation registered for tool '{tool_id}'.")
    return TOOLS[tool_id]


def load_all() -> dict:
    for name in _MODULES:
        importlib.import_module(f"corr_core.tools.{name}")
    return TOOLS
