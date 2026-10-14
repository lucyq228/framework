"""Connectors and platform tools: gcp-connector, confluence-mcp, github-mcp, profile-sources, data-quality-check."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from corr_core import paths, scoring, sources
from corr_core.tools import tool
from corr_core.tools.common import ToolContext

TEXT_SUFFIXES = {".md", ".txt"}


@tool("gcp-connector")
def gcp_connector(tc: ToolContext, params: dict) -> dict:
    backend = params.get("backend", "local")
    if backend != "local":
        return {"summary": f"Backend '{backend}' is not configured in this demo; use backend=local.",
                "status": "blocked",
                "notes": ["A BigQuery adapter needs a service account and the table names for the three extracts."]}
    inv = sources.inventory()
    avail = {k: v for k, v in inv["sources"].items() if v["available"]}
    ranges = {}
    for name in avail:
        col = sources.SOURCES[name]["date_col"]
        if col:
            f = sources.load(name)
            ranges[name] = [str(f[col].min().date()), str(f[col].max().date())]
    return {"summary": f"{len(avail)} sources available locally: {', '.join(avail)}",
            "inventory": inv, "date_ranges": ranges}


@tool("confluence-mcp")
def confluence_mcp(tc: ToolContext, params: dict) -> dict:
    docs, chunks = [], []
    for path in sorted(paths.CONTEXT_DIR.rglob("*")):
        if not path.is_file() or "_jpg" in path.parts:
            continue
        kind = path.suffix.lower()
        entry = {"file": str(path.relative_to(paths.PROJECT_ROOT)), "type": kind.lstrip(".")}
        if kind in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif kind == ".docx":
            import docx
            text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        else:
            preview = paths.CONTEXT_DIR / "_jpg" / f"{path.stem}.jpg"
            entry["image_preview"] = str(preview.relative_to(paths.PROJECT_ROOT)) if preview.exists() else None
            docs.append(entry)
            continue
        entry["characters"] = len(text)
        docs.append(entry)
        chunks.append(f"\n\n===== {entry['file']} =====\n\n{text}")
    out = tc.run_dir / "context_text.md"
    out.write_text("".join(chunks), encoding="utf-8")
    images = [d for d in docs if "image_preview" in d]
    return {"summary": f"{len(docs) - len(images)} text documents extracted to context_text.md; "
                       f"{len(images)} images listed with JPG previews to view",
            "documents": docs, "context_text": str(out.name)}


@tool("github-mcp")
def github_mcp(tc: ToolContext, params: dict) -> dict:
    gaps_path = tc.run_dir / "tool_gaps.json"
    gaps = json.loads(gaps_path.read_text(encoding="utf-8")) if gaps_path.exists() else []
    if params.get("title"):
        gaps.append({"title": params["title"], "description": params.get("description", ""),
                     "requested_by": params.get("requested_by", "agent"),
                     "created_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        gaps_path.write_text(json.dumps(gaps, indent=2), encoding="utf-8")
    return {"summary": f"{len(gaps)} tool-gap request(s) recorded for the data science team", "tool_gaps": gaps}


@tool("profile-sources")
def profile_sources(tc: ToolContext, params: dict) -> dict:
    grains = {"csat_daily": ["glin", "cal_dt", "cust_surv_psnt_pos_chnl_ds", "cust_surv_plcd_pos_chnl_ds"],
              "pace_product": ["visit_id", "question_code", "product_id"],
              "pace_service": ["visit_id", "question_id", "question_code"],
              "restaurant_attributes": ["glin"], "question_mapping": ["local_question_code"]}
    profiles = {}
    for name, meta in sources.SOURCES.items():
        if not sources.available(name):
            profiles[name] = {"available": False, "acquirable": sources.acquirable(name),
                              "description": meta["description"], "owner": meta["owner"]}
            continue
        f = sources.load(name)
        key = grains.get(name, [])
        cols = [{"name": c, "dtype": str(f[c].dtype), "null_rate": round(float(f[c].isna().mean()), 4),
                 "distinct": int(f[c].nunique())} for c in f.columns]
        prof = {"available": True, "rows": int(len(f)), "columns": cols, "declared_grain": key,
                "grain_unique": bool(not f.duplicated(subset=key).any()) if key else None,
                "duplicate_grain_rows": int(f.duplicated(subset=key).sum()) if key else None}
        if "glin" in f.columns:
            prof["restaurants"] = int(f["glin"].nunique())
        if meta["date_col"]:
            prof["date_range"] = [str(f[meta["date_col"]].min().date()), str(f[meta["date_col"]].max().date())]
        if "visit_id" in f.columns:
            prof["visits"] = int(f["visit_id"].nunique())
        profiles[name] = prof
    avail = [k for k, v in profiles.items() if v["available"]]
    return {"summary": f"Profiled {len(avail)} available sources; "
                       f"{sum(1 for v in profiles.values() if not v['available'])} catalogued sources are not available",
            "profiles": profiles}


def _check(name, count, total, severity_if_any, handling, detail=None):
    rate = count / total if total else 0.0
    status = "pass" if count == 0 else severity_if_any
    return {"check": name, "status": status, "count": int(count), "rate": round(rate, 5),
            "handling": handling, **({"detail": detail} if detail else {})}


@tool("data-quality-check")
def data_quality_check(tc: ToolContext, params: dict) -> dict:
    decisions = (tc.scope or {}).get("decisions", {})
    csat = sources.load("csat_daily")
    prod = sources.load("pace_product")
    serv = sources.load("pace_service")
    checks = []
    checks.append(_check("csat_duplicate_rows", csat.duplicated().sum(), len(csat), "warn",
                         "Drop exact duplicates before aggregation"))
    bad = (csat["frnd_high_stsf_cnt_qt"] > csat["frnd_ansr_cnt_qt"]) | (csat["qlty_high_stsf_cnt_qt"] > csat["qlty_ansr_cnt_qt"])
    checks.append(_check("csat_top_box_exceeds_answered", bad.sum(), len(csat), "warn",
                         "Drop the row (impossible count); do not cap"))
    zero = (csat["frnd_ansr_cnt_qt"] == 0) & (csat["qlty_ansr_cnt_qt"] == 0)
    checks.append(_check("csat_rows_with_no_answers", zero.sum(), len(csat), "info", "Harmless; contributes nothing"))
    checks.append(_check("product_duplicate_rows", prod.duplicated().sum(), len(prod), "warn",
                         "Drop exact duplicates before scoring"))
    checks.append(_check("service_duplicate_rows", serv.duplicated().sum(), len(serv), "warn",
                         "Drop exact duplicates before scoring"))

    pace_glins = set(prod["glin"]) | set(serv["glin"])
    csat_glins = set(csat["glin"])
    no_csat = sorted(pace_glins - csat_glins)
    checks.append(_check("pace_restaurants_without_csat", len(no_csat), len(pace_glins), "warn",
                         "Exclude from correlation; market unknown without the restaurant master", detail=no_csat))
    checks.append(_check("csat_restaurants_without_pace", len(csat_glins - pace_glins), len(csat_glins), "info",
                         "Exclude; no execution measure"))

    mapping = sources.load("question_mapping") if sources.available("question_mapping") else pd.DataFrame()
    local_codes = sorted({c for c in set(prod["question_code"]) | set(serv["question_code"]) if c.endswith(("-CA", "-UK"))})
    status_of = dict(zip(mapping.get("local_question_code", []), mapping.get("mapping_status", [])))
    unresolved = [c for c in local_codes if status_of.get(c) != "confirmed"]
    checks.append(_check("local_questions_not_confirmed", len(unresolved), len(local_codes), "warn",
                         "Follow the scope decisions for proposed and unmapped codes",
                         detail={c: status_of.get(c, "missing from mapping") for c in unresolved}))

    visits = prod.drop_duplicates("visit_id")
    dessert_na = prod[(prod["question_code"] == "Q20a") & (prod["question_exception_flag"] == -1)]["visit_id"].nunique()
    checks.append(_check("dessert_unavailable_visits", dessert_na, len(visits), "info",
                         "Category is not eligible; local core may replace it per decisions"))
    taste, _ = scoring.score_taste(decisions)
    short = int((taste["taste_eligible"] < 5).sum())
    checks.append(_check("visits_with_fewer_than_5_index_products", short, len(taste), "info",
                         "Progressive score uses eligible products as the denominator"))
    hosp, _ = scoring.score_hospitality(decisions)
    cfv = hosp[hosp["survey_name"] == "CFV"]
    checks.append(_check("cfv_not_exactly_one_channel", int((cfv["hosp_channels"] != 1).sum()), len(cfv), "fail",
                         "Investigate; a CFV should test one channel"))
    general = serv[serv["survey_area"] == "Service"]["visit_id"].nunique()
    checks.append(_check("visits_with_general_service_questions", general, serv["visit_id"].nunique(), "info",
                         f"Treatment per scope: {decisions.get('general_service_questions', 'exclude')}"))

    vy = hosp.assign(year=hosp["visit_date"].dt.year).groupby(["glin", "year", "survey_name"]).size().unstack(fill_value=0)
    per_year = vy.groupby(level="year").mean().round(2).to_dict("index")
    checks.append({"check": "visits_per_restaurant_per_year", "status": "info", "detail": per_year,
                   "handling": "Small PACE samples: restaurant-month analysis is not viable"})
    c = csat.assign(month=csat["cal_dt"].dt.to_period("M"))
    months = c.groupby("glin")["month"].nunique()
    full = c["month"].nunique()
    partial = months[months < full]
    checks.append(_check("restaurants_with_partial_csat_history", len(partial), len(months), "info",
                         "Likely new openings; windows before opening have no CSAT",
                         detail={g: int(m) for g, m in partial.items()}))
    try:
        terr_ok = csat["terr_cd"].isin(list(sources.MARKET_BY_TERR)).all()
    except Exception:
        terr_ok = False
    checks.append({"check": "territory_codes_recognised", "status": "pass" if terr_ok else "fail",
                   "handling": "Map 124 Canada, 826 United Kingdom, 372 Ireland"})

    order = {"fail": 3, "warn": 2, "info": 1, "pass": 0}
    worst = max((ch["status"] for ch in checks), key=lambda s: order[s])
    counts = {s: sum(1 for ch in checks if ch["status"] == s) for s in order}
    return {"summary": f"{len(checks)} checks: {counts['fail']} fail, {counts['warn']} warn, "
                       f"{counts['info']} info, {counts['pass']} pass (worst: {worst})",
            "overall": worst, "checks": checks}
