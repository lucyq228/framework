"""Source catalogue, loaders, and the data inventory used for feasibility.

A source is "available" when its file exists in data/raw or data/reference.
Sources listed with no file describe data the project could acquire (or that
does not exist anywhere yet); feasibility reports them as data requests.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from corr_core.paths import ACQUIRABLE_DIR, RAW_DIR, REFERENCE_DIR

SOURCES = {
    "csat_daily": dict(
        file=RAW_DIR / "csat_daily.csv", date_col="cal_dt",
        description="CSAT top-two-box counts per restaurant, day, presented and placed channel",
        owner="Customer Insights (CSAT platform)",
    ),
    "pace_product": dict(
        file=RAW_DIR / "pace_product_responses.csv", date_col="visit_date",
        description="PACE Taste product checks, one row per product question per visit",
        owner="Operations Excellence (PACE)",
    ),
    "pace_service": dict(
        file=RAW_DIR / "pace_service_responses.csv", date_col="visit_date",
        description="PACE Hospitality service checks, one row per service question per visit",
        owner="Operations Excellence (PACE)",
    ),
    "question_mapping": dict(
        file=REFERENCE_DIR / "question_mapping.csv", date_col=None,
        description="Local to global PACE question mapping with confirmation status",
        owner="Market PACE leads",
    ),
    "restaurant_attributes": dict(
        file=REFERENCE_DIR / "restaurant_attributes.csv", date_col=None,
        acquirable_file=ACQUIRABLE_DIR / "restaurant_attributes.csv",
        description="Restaurant master: region, restaurant type, ownership, volume band, channel flags",
        owner="Restaurant master data (Real Estate / Development)",
    ),
    "csat_product_level": dict(
        file=None, date_col=None,
        description="CSAT responses tagged to the product ordered, or item-level taste questions",
        owner="Customer Insights (CSAT platform)",
    ),
    "pace_dimension_results": dict(
        file=None, date_col=None,
        description="PACE product results per dimension (appearance, taste, temperature, texture) on every check",
        owner="Operations Excellence (PACE)",
    ),
    "visit_timestamp": dict(
        file=None, date_col=None,
        description="PACE visit start time and CSAT response hour or daypart",
        owner="Operations Excellence (PACE) and Customer Insights",
    ),
    "ops_data": dict(
        file=None, date_col=None,
        description="Speed of service, staffing levels, crew tenure by restaurant and day",
        owner="Restaurant Operations / Workforce Management",
    ),
    "delivery_csat": dict(
        file=None, date_col=None,
        description="Delivery aggregator satisfaction feed by restaurant",
        owner="Delivery partnerships",
    ),
}

DTYPES = {"glin": str}
TASTE_INDEX_CODES = {"Q9a": "Beef", "Q10a": "Chicken", "Q15a": "Fries", "Q19a": "Beverage", "Q20a": "Dessert"}
DIAGNOSTIC_CODES = {"Q21a": "Hot Coffee", "Q11a-MC": "McCafe Food", "Q13a-MC": "McCafe Drink"}
MARKET_BY_TERR = {124: "Canada", 826: "United Kingdom", 372: "Ireland"}
MARKET_GROUP = {"Canada": "Canada", "United Kingdom": "UK&I", "Ireland": "UK&I"}


def available(name: str) -> bool:
    f = SOURCES[name]["file"]
    return bool(f) and Path(f).exists()


def acquirable(name: str) -> bool:
    f = SOURCES[name].get("acquirable_file")
    return bool(f) and Path(f).exists() and not available(name)


@lru_cache(maxsize=None)
def _read(name: str, mtime: float) -> pd.DataFrame:
    meta = SOURCES[name]
    frame = pd.read_csv(meta["file"], dtype=DTYPES)
    if meta["date_col"]:
        frame[meta["date_col"]] = pd.to_datetime(frame[meta["date_col"]], format="%m/%d/%y")
    return frame


def load(name: str) -> pd.DataFrame:
    if not available(name):
        raise FileNotFoundError(f"Source '{name}' is not available. Feasibility should list it as a data request.")
    path = Path(SOURCES[name]["file"])
    return _read(name, path.stat().st_mtime).copy()


def restaurant_dim() -> pd.DataFrame:
    """Restaurant-level dimensions: always market; full attributes when acquired."""
    if available("restaurant_attributes"):
        dim = load("restaurant_attributes")
        dim["terr_cd"] = dim["terr_cd"].astype(int)
        return dim
    csat = load("csat_daily")[["glin", "terr_cd"]].drop_duplicates("glin")
    glins = set(load("pace_product")["glin"]) | set(load("pace_service")["glin"])
    missing = sorted(glins - set(csat["glin"]))
    dim = csat.copy()
    if missing:
        # Restaurants without CSAT: the market is still known from the GLIN range used in the extract.
        dim = pd.concat([dim, pd.DataFrame({"glin": missing, "terr_cd": pd.NA})], ignore_index=True)
    dim["market"] = dim["terr_cd"].map(lambda t: MARKET_BY_TERR.get(int(t)) if pd.notna(t) else "Unknown")
    dim["market_group"] = dim["market"].map(MARKET_GROUP).fillna("Unknown")
    return dim


def segment_columns() -> list[str]:
    cols = ["market", "market_group"]
    if available("restaurant_attributes"):
        cols += ["region", "restaurant_type", "ownership_type", "volume_band",
                 "has_drive_thru", "has_table_service", "has_mccafe"]
    return cols


def inventory() -> dict:
    """What exists right now: per source availability, rows, columns, and a fingerprint."""
    out = {}
    for name, meta in SOURCES.items():
        entry = {"available": available(name), "acquirable": acquirable(name),
                 "description": meta["description"], "owner": meta["owner"]}
        if entry["available"]:
            path = Path(meta["file"])
            header = pd.read_csv(path, nrows=0).columns.tolist()
            with path.open("rb") as handle:
                rows = sum(1 for _ in handle) - 1
            entry.update(file=str(path.name), rows=rows, columns=header,
                         modified=path.stat().st_mtime)
        out[name] = entry
    fingerprint = hashlib.sha256(json.dumps(
        {k: (v["available"], v.get("rows")) for k, v in out.items()}, sort_keys=True).encode()).hexdigest()[:16]
    return {"sources": out, "fingerprint": fingerprint}
