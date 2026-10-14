"""Registered tools whose data does not exist yet. They are on the shelf so feasibility can
name them and issue data requests; running one returns status=blocked."""

from __future__ import annotations

from corr_core import sources
from corr_core.tools import tool
from corr_core.tools.common import ToolContext


def _blocked(tool_id: str, needs: list[str]) -> dict:
    missing = [n for n in needs if not sources.available(n)]
    return {"summary": f"{tool_id} is blocked: needs {', '.join(missing)}", "status": "blocked",
            "data_requests": [{"source": n, "description": sources.SOURCES[n]["description"],
                               "owner": sources.SOURCES[n]["owner"]} for n in missing]}


@tool("product-level-csat")
def product_level_csat(tc: ToolContext, params: dict) -> dict:
    return _blocked("product-level-csat", ["csat_product_level"])


@tool("daypart-alignment")
def daypart_alignment(tc: ToolContext, params: dict) -> dict:
    return _blocked("daypart-alignment", ["visit_timestamp"])


@tool("operational-drivers")
def operational_drivers(tc: ToolContext, params: dict) -> dict:
    return _blocked("operational-drivers", ["ops_data"])
