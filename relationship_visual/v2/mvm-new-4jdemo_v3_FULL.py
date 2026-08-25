"""Build the MVM v3 relationship graph as a standalone interactive HTML file.

The input schema is:
    gc_decomp_bucket, gc_decomp_bucket_l2, driver_group,
    KPI, Drill-down KPI, State

External Context, Growth Momentum, and Foundation Protection are equal peer
pillars. Status colors are applied to KPI and drill-down nodes; hierarchy nodes
use blue so mixed-status categories are not misrepresented. Pass
``--include-png`` only when a static PNG is also needed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # Pillow is only required for --include-png.
    Image = ImageDraw = ImageFont = None


STATUS_COLORS = {
    "Answerable": "#69B578",
    "Blocked": "#B8C0CC",
    "Directional": "#F4C95D",
}
TYPE_COLORS = {
    "Bucket": "#234A73",
    "L2": "#3E6F9E",
    "Driver": "#DCEAF5",
}
TYPE_RADIUS = {
    "Bucket": 92,
    "L2": 82,
    "Driver": 78,
    "KPI": 70,
    "Drill-down KPI": 62,
}
TYPE_FONT_SIZE = {
    "Bucket": 30,
    "L2": 27,
    "Driver": 24,
    "KPI": 22,
    "Drill-down KPI": 19,
}
TYPE_MAX_LINES = {
    "Bucket": 3,
    "L2": 3,
    "Driver": 5,
    "KPI": 4,
    "Drill-down KPI": 4,
}
EDGE_COLOR = "#9AA8B7"
PEER_EDGE_COLOR = "#4B6F95"
TEXT_COLOR = "#17212B"
MUTED_TEXT = "#5E6B78"
WHITE = "#FFFFFF"

BUCKET_ORDER = [
    "External Context",
    "Growth Momentum",
    "Foundation Protection",
]


def clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def safe_id(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return key or "blank"


def node_id(kind: str, *parts: str) -> str:
    return kind + "|" + "|".join(safe_id(part) for part in parts)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [{key: clean(value) for key, value in row.items()} for row in csv.DictReader(handle)]

    expected = {
        "gc_decomp_bucket",
        "gc_decomp_bucket_l2",
        "driver_group",
        "KPI",
        "Drill-down KPI",
        "State",
    }
    if not rows:
        raise ValueError("The input CSV is empty.")
    missing = expected - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    usable = []
    for row in rows:
        if not row["gc_decomp_bucket"] and not row["State"]:
            continue
        if not all(row[field] for field in ("gc_decomp_bucket", "gc_decomp_bucket_l2", "driver_group", "KPI", "State")):
            raise ValueError(f"Incomplete hierarchy row: {row}")
        if row["State"] not in STATUS_COLORS:
            raise ValueError(f"Unexpected State value: {row['State']!r}")
        usable.append(row)
    return usable


def build_graph(rows: list[dict[str, str]]) -> tuple[dict[str, dict], list[dict]]:
    nodes: dict[str, dict] = {}
    edge_keys: set[tuple[str, str, str]] = set()
    edges: list[dict] = []

    def add_node(identifier: str, name: str, kind: str, bucket: str, branch: str = "", state: str = "") -> None:
        node = nodes.setdefault(
            identifier,
            {
                "id": identifier,
                "name": name,
                "type": kind,
                "bucket": bucket,
                "branch": branch or bucket,
                "state": state,
                "states": Counter(),
                "children": [],
            },
        )
        if state:
            if node["state"] and node["state"] != state:
                node["state"] = "Mixed"
            elif not node["state"]:
                node["state"] = state

    def add_edge(source: str, target: str, relation: str) -> None:
        key = (source, target, relation)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({"source": source, "target": target, "relation": relation})
        nodes[source]["children"].append(target)

    for row in rows:
        bucket = row["gc_decomp_bucket"]
        l2 = row["gc_decomp_bucket_l2"]
        driver = row["driver_group"]
        kpi = row["KPI"]
        drill = row["Drill-down KPI"]
        state = row["State"]

        bucket_key = node_id("bucket", bucket)
        # The v3 file repeats the bucket name as L2 for External Context and
        # Growth Momentum. Collapse those duplicate labels into one visual node.
        l2_key = bucket_key if l2.casefold() == bucket.casefold() else node_id("l2", bucket, l2)
        driver_key = node_id("driver", bucket, l2, driver)
        kpi_key = node_id("kpi", bucket, l2, driver, kpi)

        add_node(bucket_key, bucket, "Bucket", bucket, branch=bucket)
        if l2_key != bucket_key:
            add_node(l2_key, l2, "L2", bucket, branch=l2)
        add_node(driver_key, driver, "Driver", bucket, branch=l2)
        add_node(kpi_key, kpi, "KPI", bucket, branch=l2, state=state)
        if l2_key != bucket_key:
            add_edge(bucket_key, l2_key, "HAS_L2")
        add_edge(l2_key, driver_key, "HAS_DRIVER")
        add_edge(driver_key, kpi_key, "HAS_KPI")

        for key in dict.fromkeys((bucket_key, l2_key, driver_key, kpi_key)):
            nodes[key]["states"][state] += 1

        if drill:
            drill_key = node_id("drill", bucket, l2, driver, kpi, drill)
            add_node(drill_key, drill, "Drill-down KPI", bucket, branch=l2, state=state)
            nodes[drill_key]["states"][state] += 1
            add_edge(kpi_key, drill_key, "HAS_DRILLDOWN")

    for node in nodes.values():
        node["status_counts"] = dict(node.pop("states"))

    return nodes, edges


def bucket_sort_key(name: str) -> tuple[int, str]:
    try:
        return BUCKET_ORDER.index(name), name
    except ValueError:
        return len(BUCKET_ORDER), name


def build_layout(nodes: dict[str, dict], edges: list[dict]) -> tuple[dict[str, tuple[float, float]], int, int, list[dict]]:
    """Create a compact, column-based tidy tree.

    The three bucket nodes are independent peer roots. Every sibling set is
    stacked vertically, and its parent is centered on that vertical group.
    """

    status_rank = {"Answerable": 0, "Directional": 1, "Blocked": 2}

    def node_rank(node: dict) -> tuple[int, str]:
        if node["state"] in status_rank:
            rank = status_rank[node["state"]]
        else:
            present = [status_rank[state] for state, count in node["status_counts"].items() if count]
            rank = min(present, default=3)
        return rank, node["name"].casefold()

    for node in nodes.values():
        node["children"].sort(key=lambda child_id: node_rank(nodes[child_id]))

    roots = sorted(
        (node for node in nodes.values() if node["type"] == "Bucket"),
        key=lambda node: bucket_sort_key(node["name"]),
    )
    positions: dict[str, tuple[float, float]] = {}
    bands: list[dict] = []
    cursor_y = 270.0
    leaf_gap = 190.0
    root_gap = 340.0
    origin_x = 360.0
    level_gap = 470.0

    def place(identifier: str, depth: int) -> float:
        nonlocal cursor_y
        node = nodes[identifier]
        children = node["children"]
        if children:
            child_ys = [place(child_id, depth + 1) for child_id in children]
            y = (child_ys[0] + child_ys[-1]) / 2
        else:
            y = cursor_y
            cursor_y += leaf_gap
        positions[identifier] = (origin_x + depth * level_gap, y)
        return y

    for root in roots:
        band_top = cursor_y - leaf_gap / 2
        root_y = place(root["id"], 0)
        band_bottom = cursor_y - leaf_gap / 2
        bands.append({"bucket": root["name"], "top": band_top, "bottom": band_bottom, "center": root_y})
        cursor_y += root_gap

    max_x = max(x + TYPE_RADIUS[nodes[node_id_value]["type"]] for node_id_value, (x, _) in positions.items())
    width = int(math.ceil(max_x + 220))
    height = int(math.ceil(cursor_y + 120))
    return positions, width, height, bands


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def node_fill(node: dict) -> str:
    if node["type"] in TYPE_COLORS:
        return TYPE_COLORS[node["type"]]
    return STATUS_COLORS.get(node["state"], STATUS_COLORS["Blocked"])


def text_fill(node: dict) -> str:
    return WHITE if node["type"] in {"Bucket", "L2"} else TEXT_COLOR


def wrap_for_circle(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, radius: int, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    max_width = radius * 1.58
    for word in words:
        candidate = word if not current else current + " " + word
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        final = lines[-1]
        while final and draw.textbbox((0, 0), final + "…", font=font)[2] > max_width:
            final = final[:-1]
        lines[-1] = final.rstrip() + "…"
    return lines


def edge_endpoints(a: tuple[float, float], b: tuple[float, float], ra: float, rb: float) -> tuple[tuple[float, float], tuple[float, float]]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    return (a[0] + ux * ra, a[1] + uy * ra), (b[0] - ux * rb, b[1] - uy * rb)


def bezier_points(start: tuple[float, float], end: tuple[float, float], samples: int = 28) -> list[tuple[float, float]]:
    x1, y1 = start
    x2, y2 = end
    control_x = (x1 + x2) / 2
    result = []
    for index in range(samples + 1):
        t = index / samples
        u = 1 - t
        x = u**3 * x1 + 3 * u**2 * t * control_x + 3 * u * t**2 * control_x + t**3 * x2
        y = u**3 * y1 + 3 * u**2 * t * y1 + 3 * u * t**2 * y2 + t**3 * y2
        result.append((x, y))
    return result


def draw_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    x2, y2 = points[-1]
    x1, y1 = points[-3]
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 14 + width
    spread = 0.48
    left = (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread))
    right = (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread))
    draw.polygon([(x2, y2), left, right], fill=color)


def render_png(nodes: dict[str, dict], edges: list[dict], positions: dict[str, tuple[float, float]], width: int, height: int, bands: list[dict], destination: Path, source_name: str) -> None:
    scale = 1
    image = Image.new("RGB", (width * scale, height * scale), WHITE)
    draw = ImageDraw.Draw(image)

    title_font = find_font(42, bold=True)
    subtitle_font = find_font(22)
    legend_font = find_font(20)
    band_font = find_font(20, bold=True)

    draw.text((90, 50), "MVM Relationship Network", font=title_font, fill=TEXT_COLOR)
    draw.text((90, 112), f"Three equal pillars • {len(nodes)} nodes • {len(edges)} relationships • Source: {source_name}", font=subtitle_font, fill=MUTED_TEXT)

    legend_x = width - 1680
    legend_y = 68
    legend = [
        (TYPE_COLORS["Bucket"], "Hierarchy"),
        (STATUS_COLORS["Answerable"], "Answerable"),
        (STATUS_COLORS["Blocked"], "Blocked"),
        (STATUS_COLORS["Directional"], "Directional"),
    ]
    for color, label in legend:
        draw.ellipse((legend_x, legend_y, legend_x + 25, legend_y + 25), fill=color, outline="#5F6B76", width=1)
        draw.text((legend_x + 38, legend_y - 1), label, font=legend_font, fill=TEXT_COLOR)
        legend_x += 390

    for index, band in enumerate(bands):
        if index:
            y = int(band["top"] - 90)
            draw.line((70, y, width - 70, y), fill="#E7EBF0", width=2)
        draw.text((85, int(band["top"])), band["bucket"].upper(), font=band_font, fill="#7B8794")

    for edge in edges:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        a = positions[edge["source"]]
        b = positions[edge["target"]]
        start, end = edge_endpoints(a, b, TYPE_RADIUS[source["type"]], TYPE_RADIUS[target["type"]])
        color = EDGE_COLOR
        line_width = 3
        draw_arrow(draw, bezier_points(start, end), color, line_width)

    node_order = {"Bucket": 0, "L2": 1, "Driver": 2, "KPI": 3, "Drill-down KPI": 4}
    for node in sorted(nodes.values(), key=lambda item: node_order[item["type"]]):
        x, y = positions[node["id"]]
        radius = TYPE_RADIUS[node["type"]]
        fill = node_fill(node)
        outline = "#2E4053" if node["type"] in {"Bucket", "L2"} else "#748290"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=3)

        if node["type"] in {"Bucket", "L2", "Driver"}:
            counts = node["status_counts"]
            total = sum(counts.values())
            if total:
                angle = -90.0
                ring_box = (x - radius - 7, y - radius - 7, x + radius + 7, y + radius + 7)
                for state in ("Answerable", "Directional", "Blocked"):
                    count = counts.get(state, 0)
                    if not count:
                        continue
                    sweep = count / total * 360
                    draw.arc(ring_box, angle, angle + sweep, fill=STATUS_COLORS[state], width=7)
                    angle += sweep

        font = find_font(TYPE_FONT_SIZE[node["type"]], bold=node["type"] in {"Bucket", "L2"})
        lines = wrap_for_circle(draw, node["name"], font, radius, TYPE_MAX_LINES[node["type"]])
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        heights = [box[3] - box[1] for box in boxes]
        line_gap = 3
        total_height = sum(heights) + line_gap * max(0, len(lines) - 1)
        cursor_y = y - total_height / 2
        for line, box, line_height in zip(lines, boxes, heights):
            line_width = box[2] - box[0]
            draw.text((x - line_width / 2, cursor_y), line, font=font, fill=text_fill(node))
            cursor_y += line_height + line_gap

    image.save(destination, format="PNG", optimize=True)


def render_html(nodes: dict[str, dict], edges: list[dict], positions: dict[str, tuple[float, float]], width: int, height: int, destination: Path, source_name: str) -> None:
    html_radius = {"Bucket": 50, "L2": 39, "Driver": 33, "KPI": 27, "Drill-down KPI": 23}
    html_fill = {"Bucket": "#5F6B76", "L2": "#2F80B9", "Driver": "#58AEE2"}

    def hierarchy_availability(node: dict) -> tuple[str, str]:
        states = {state for state, count in node["status_counts"].items() if count}
        if states == {"Answerable"}:
            return "Answerable", STATUS_COLORS["Answerable"]
        if states == {"Blocked"}:
            return "Blocked", STATUS_COLORS["Blocked"]
        return "Partial", STATUS_COLORS["Directional"]

    browser_nodes = []
    for node in nodes.values():
        availability, outline = hierarchy_availability(node) if node["type"] in {"Bucket", "L2", "Driver"} else ("", "#FFFFFF")
        browser_nodes.append(
            {
                "id": node["id"], "name": node["name"], "type": node["type"],
                "bucket": node["bucket"], "branch": node["branch"], "state": node["state"],
                "statusCounts": node["status_counts"], "children": node["children"],
                "r": html_radius[node["type"]],
                "fill": html_fill.get(node["type"], STATUS_COLORS.get(node["state"], STATUS_COLORS["Blocked"])),
                "availability": availability, "outline": outline,
                "textFill": "#FFFFFF" if node["type"] in {"Bucket", "L2", "Driver"} else "#17212B",
            }
        )

    bucket_ids = [node_id("bucket", name) for name in BUCKET_ORDER if node_id("bucket", name) in nodes]
    browser_edges = list(edges)
    if len(bucket_ids) == 3:
        browser_edges.extend(
            [
                {"source": bucket_ids[0], "target": bucket_ids[1], "relation": "PEER"},
                {"source": bucket_ids[1], "target": bucket_ids[2], "relation": "PEER"},
                {"source": bucket_ids[2], "target": bucket_ids[0], "relation": "PEER"},
            ]
        )

    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MVM Relationship Network v3</title>
<style>
:root{--ink:#17212b;--muted:#5e6b78;--line:#a5b1bd;--border:#d9e0e7;--soft:#f5f7fa;--green:#69b578;--gray:#b8c0cc;--yellow:#f4c95d;--blue:#58aee2}
*{box-sizing:border-box}html,body{height:100%;margin:0}body{font-family:"Segoe UI",Arial,sans-serif;color:var(--ink);background:#fff;overflow:hidden}
.bar{height:88px;display:flex;align-items:center;gap:10px;padding:12px 18px;background:rgba(255,255,255,.97);border-bottom:1px solid var(--border);position:relative;z-index:3}.heading{margin-right:auto;min-width:300px}.heading h1{font-size:20px;line-height:1.2;margin:0 0 4px;font-weight:600}.heading p{font-size:12px;color:var(--muted);margin:0}
button{font:inherit;font-size:13px;color:var(--ink);background:#fff;border:1px solid var(--border);border-radius:7px;padding:8px 11px;cursor:pointer}button:hover{background:var(--soft)}button[aria-pressed="true"]{background:#425466;color:#fff;border-color:#425466}button:focus-visible{outline:3px solid rgba(47,128,185,.22);outline-offset:1px}.legend{display:flex;gap:11px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted)}.legend span{display:inline-flex;gap:5px;align-items:center}.swatch{width:12px;height:12px;border-radius:50%;border:1px solid rgba(23,33,43,.25)}
#viewport{height:calc(100% - 88px);width:100%;overflow:hidden;background:linear-gradient(180deg,#fbfcfd,#f7f9fb);cursor:grab;touch-action:none}#viewport.dragging{cursor:grabbing}svg{display:block;width:100%;height:100%;user-select:none}.edge{stroke:var(--line);stroke-width:1.15;vector-effect:non-scaling-stroke}.edge.peer{stroke:#657482;stroke-width:1.6;stroke-dasharray:7 5}.edge.dim{opacity:.08}
.node{cursor:grab}.node:active{cursor:grabbing}.node circle{stroke:#fff;stroke-width:2;vector-effect:non-scaling-stroke}.node text{pointer-events:none;text-anchor:middle;dominant-baseline:middle;font-weight:500}.node.dim{opacity:.1}.node.match circle{stroke:#183a57;stroke-width:4}.tooltip{position:fixed;z-index:6;display:none;pointer-events:none;max-width:320px;padding:9px 11px;background:#16202a;color:#fff;border-radius:7px;font-size:12px;box-shadow:0 8px 22px rgba(0,0,0,.18)}.tooltip strong{display:block;font-size:13px;margin-bottom:3px}.tooltip .muted{color:#cbd4dd}
@media(max-width:900px){.bar{height:auto;min-height:112px;align-items:flex-start;flex-wrap:wrap}.heading{width:100%}#viewport{height:calc(100% - 112px)}.legend{display:none}}
</style></head><body>
<header class="bar"><div class="heading"><h1>MVM Relationship Network</h1><p>Neo4j-style force view · __NODE_COUNT__ nodes · __EDGE_COUNT__ relationships · __SOURCE__</p></div>
<button id="showAll" type="button" aria-pressed="true">All hierarchy</button><button id="showTop" type="button" aria-pressed="false">Top hierarchy</button><button id="reheat" type="button">Restart layout</button><button id="fit" type="button">Fit network</button>
<div class="legend" aria-label="Color legend"><span><i class="swatch" style="background:#58aee2"></i>Hierarchy fill</span><span><i class="swatch" style="background:var(--green)"></i>Answerable / available</span><span><i class="swatch" style="background:var(--yellow)"></i>Partial / directional</span><span><i class="swatch" style="background:var(--gray)"></i>Blocked</span></div></header>
<main id="viewport" aria-label="Interactive force-directed relationship graph. Drag nodes, wheel to zoom, and drag the background to pan."><svg id="graph" viewBox="0 0 2300 1450" role="img" aria-labelledby="svgTitle svgDesc"><title id="svgTitle">MVM force-directed relationship network</title><desc id="svgDesc">External Context, Growth Momentum, and Foundation Protection are three equal peer hubs. Hierarchy nodes surround their parent hubs, and leaf metrics expand outward from their hierarchy nodes.</desc></svg></main>
<div id="tooltip" class="tooltip" role="tooltip"></div>
<script>
const nodes=__NODES__,edges=__EDGES__,W=2300,H=1450,NS='http://www.w3.org/2000/svg';
const svg=document.getElementById('graph'),viewport=document.getElementById('viewport'),tip=document.getElementById('tooltip'),nodeById=new Map(nodes.map(n=>[n.id,n]));
const anchors={'External Context':{x:440,y:720},'Growth Momentum':{x:1050,y:320},'Foundation Protection':{x:1490,y:720}},branchCenters={'Restaurant Foundation':{x:1270,y:1060},'Commercial BAU':{x:1840,y:870}},statusRank={Answerable:0,Directional:1,Blocked:2};
let vb={x:0,y:0,w:W,h:H},alpha=1,raf=0,dragNode=null,dragStart=null,pan=null,moved=false;
function el(tag,attrs={}){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e}
function hash(text){let h=2166136261;for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
function centerFor(node){return node.bucket==='Foundation Protection'&&branchCenters[node.branch]?branchCenters[node.branch]:(anchors[node.bucket]||{x:W/2,y:H/2})}
for(const node of nodes){node.x=W/2;node.y=H/2;node.vx=0;node.vy=0;node.visible=true}
const links=edges.map((edge,index)=>({...edge,index,sourceNode:nodeById.get(edge.source),targetNode:nodeById.get(edge.target)}));
const childrenById=new Map(),parentById=new Map(),layoutTargets=new Map();
for(const link of links){if(link.relation==='PEER')continue;parentById.set(link.target,link.sourceNode);if(!childrenById.has(link.source))childrenById.set(link.source,[]);childrenById.get(link.source).push(link.targetNode)}
function hierarchyOwner(node){let current=node;while(current&&current.type!=='Driver')current=parentById.get(current.id);return current?.id||node.id}for(const node of nodes)node.clusterId=hierarchyOwner(node);
function activeNodes(){return nodes.filter(n=>n.visible)}function activeLinks(){return links.filter(link=>link.sourceNode.visible&&link.targetNode.visible)}
function orderedChildren(parent){return(childrenById.get(parent.id)||[]).filter(n=>n.visible).sort((a,b)=>a.name.localeCompare(b.name)||a.id.localeCompare(b.id))}
function tierDistance(parent,child){if(child.type==='L2')return 330;if(child.type==='Driver')return 165;if(child.type==='KPI')return 105;return 78}
function putTarget(node,x,y){layoutTargets.set(node.id,{x:Math.max(node.r+45,Math.min(W-node.r-45,x)),y:Math.max(node.r+45,Math.min(H-node.r-45,y))})}
function rebuildTargets(){layoutTargets.clear();for(const node of nodes.filter(n=>n.type==='Bucket'&&n.visible)){const a=anchors[node.bucket];putTarget(node,a.x,a.y)}
  const visit=parent=>{const p=layoutTargets.get(parent.id),kids=orderedChildren(parent);if(!p||!kids.length)return;const fullRing=['Bucket','L2','Driver'].includes(parent.type),owner=layoutTargets.get(parent.clusterId),cluster=owner||centerFor(parent),outward=Math.hypot(p.x-cluster.x,p.y-cluster.y)>20?Math.atan2(p.y-cluster.y,p.x-cluster.x):(hash(parent.id)%6283)/1000,spread=Math.min(1.8,.7*Math.max(1,kids.length-1)),offset=(hash(parent.id)%6283)/1000;
    kids.forEach((child,index)=>{if(child.type==='L2'&&child.bucket==='Foundation Protection'&&branchCenters[child.branch]){const a=branchCenters[child.branch];putTarget(child,a.x,a.y)}else{const angle=fullRing?offset+Math.PI*2*index/Math.max(1,kids.length):outward+(kids.length===1?0:(index-(kids.length-1)/2)*spread/(kids.length-1)),distance=tierDistance(parent,child);putTarget(child,p.x+Math.cos(angle)*distance,p.y+Math.sin(angle)*distance)}visit(child)})};
  for(const root of nodes.filter(n=>n.type==='Bucket'&&n.visible))visit(root);for(const node of activeNodes())if(!layoutTargets.has(node.id)){const a=centerFor(node);putTarget(node,a.x,a.y)}}
const defs=el('defs'),marker=el('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:3.5,markerHeight:3.5,orient:'auto'});marker.appendChild(el('path',{d:'M0 0 L10 5 L0 10 z',fill:'#9aa8b7'}));defs.appendChild(marker);svg.appendChild(defs);
const edgeLayer=el('g'),nodeLayer=el('g');svg.append(edgeLayer,nodeLayer);
for(const link of links){link.line=el('line',{class:'edge '+(link.relation==='PEER'?'peer':''),'data-source':link.source,'data-target':link.target,'data-relation':link.relation});if(link.relation!=='PEER')link.line.setAttribute('marker-end','url(#arrow)');edgeLayer.appendChild(link.line)}
function wrap(text,max){const words=text.split(/\s+/),lines=[];let line='';for(const word of words){const next=line?line+' '+word:word;if(next.length<=max)line=next;else{if(line)lines.push(line);line=word}}if(line)lines.push(line);if(lines.length>3){lines.length=3;lines[2]=lines[2].slice(0,Math.max(1,max-1))+'…'}return lines}
for(const node of nodes){const g=el('g',{class:'node','data-id':node.id,'data-type':node.type,'data-cluster':node.clusterId});node.g=g;g.appendChild(el('circle',{r:node.r,fill:node.fill,stroke:node.outline,'stroke-width':node.availability?4:2}));const max=node.type==='Bucket'?13:node.type==='L2'?11:node.type==='Driver'?10:node.type==='KPI'?9:8,font=node.type==='Bucket'?13:node.type==='L2'?12:node.type==='Driver'?11:10,lines=wrap(node.name,max),start=-(lines.length-1)*font*.55;lines.forEach((line,i)=>{const t=el('text',{x:0,y:start+i*font*1.05,'font-size':font,fill:node.textFill});t.textContent=line;g.appendChild(t)});g.addEventListener('mouseenter',e=>showTip(node));g.addEventListener('mousemove',moveTip);g.addEventListener('mouseleave',hideTip);g.addEventListener('pointerdown',e=>startNodeDrag(e,node));nodeLayer.appendChild(g)}
function desired(link){if(link.relation==='PEER')return 720;if(link.relation==='HAS_L2')return 330;if(link.relation==='HAS_DRIVER')return 165;if(link.relation==='HAS_KPI')return 105;return 78}
function simulate(){const strength=alpha,shownNodes=activeNodes(),shownLinks=activeLinks();for(const link of shownLinks){const a=link.sourceNode,b=link.targetNode,dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,delta=(d-desired(link))*.04*strength,ux=dx/d,uy=dy/d;if(a.fx==null){a.vx+=ux*delta;a.vy+=uy*delta}if(b.fx==null){b.vx-=ux*delta;b.vy-=uy*delta}}
  for(let i=0;i<shownNodes.length;i++){const a=shownNodes[i];for(let j=i+1;j<shownNodes.length;j++){const b=shownNodes[j],dx=b.x-a.x,dy=b.y-a.y,d2=Math.max(36,dx*dx+dy*dy),d=Math.sqrt(d2),connected=parentById.get(a.id)?.id===b.id||parentById.get(b.id)?.id===a.id,sameCluster=a.clusterId===b.clusterId,min=a.r+b.r+(connected||sameCluster?14:48),ux=dx/d,uy=dy/d,charge=(connected||sameCluster?750:2300)*strength/d2;if(a.fx==null){a.vx-=ux*charge;a.vy-=uy*charge}if(b.fx==null){b.vx+=ux*charge;b.vy+=uy*charge}if(d<min){const push=(min-d)*((connected||sameCluster) ? 0.10 : 0.14)*strength;if(a.fx==null){a.vx-=ux*push;a.vy-=uy*push}if(b.fx==null){b.vx+=ux*push;b.vy+=uy*push}}}}
  for(const n of shownNodes){const target=layoutTargets.get(n.id);if(n.fx==null&&target){n.vx+=(target.x-n.x)*.012*strength;n.vy+=(target.y-n.y)*.012*strength}if(n.fx!=null){n.x=n.fx;n.y=n.fy;n.vx=n.vy=0}else{n.vx*=.8;n.vy*=.8;n.x=Math.max(n.r+18,Math.min(W-n.r-18,n.x+n.vx));n.y=Math.max(n.r+18,Math.min(H-n.r-18,n.y+n.vy))}}
  alpha*=.975;update();if(alpha>.015)raf=requestAnimationFrame(simulate);else raf=0}
function update(){for(const link of activeLinks()){const a=link.sourceNode,b=link.targetNode,dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,ux=dx/d,uy=dy/d,x1=a.x+ux*a.r,y1=a.y+uy*a.r,x2=b.x-ux*b.r,y2=b.y-uy*b.r;link.line.setAttribute('x1',x1);link.line.setAttribute('y1',y1);link.line.setAttribute('x2',x2);link.line.setAttribute('y2',y2)}for(const n of activeNodes())n.g.setAttribute('transform',`translate(${n.x},${n.y})`)}
function restart(){cancelAnimationFrame(raf);rebuildTargets();for(const n of activeNodes()){const target=layoutTargets.get(n.id),jitter=n.type==='Bucket'||n.type==='L2'?0:(hash(n.id)%13)-6;n.vx=n.vy=0;n.x=target.x+jitter;n.y=target.y-jitter;if(n.type==='Bucket'||(n.type==='L2'&&n.bucket==='Foundation Protection')){n.fx=target.x;n.fy=target.y}else n.fx=n.fy=null}alpha=1;clearFocus();if(matchMedia('(prefers-reduced-motion: reduce)').matches){for(let i=0;i<260;i++)simulateStepOnly();update()}else simulate()}
function simulateStepOnly(){const old=raf;raf=0;simulate();cancelAnimationFrame(raf);raf=old}
function worldPoint(e){const rect=svg.getBoundingClientRect();return{x:vb.x+(e.clientX-rect.left)/rect.width*vb.w,y:vb.y+(e.clientY-rect.top)/rect.height*vb.h}}
function startNodeDrag(e,node){e.stopPropagation();moved=false;dragNode=node;dragStart={x:e.clientX,y:e.clientY};node.g.setPointerCapture(e.pointerId);const p=worldPoint(e);node.fx=p.x;node.fy=p.y;alpha=Math.max(alpha,.35);if(!raf)simulate()}
svg.addEventListener('pointermove',e=>{if(dragNode){const p=worldPoint(e);moved=moved||Math.hypot(e.clientX-dragStart.x,e.clientY-dragStart.y)>4;if(moved){dragNode.fx=p.x;dragNode.fy=p.y;if(dragNode.type==='Bucket'){anchors[dragNode.bucket].x=p.x;anchors[dragNode.bucket].y=p.y;rebuildTargets()}else if(dragNode.type==='L2'&&dragNode.bucket==='Foundation Protection'&&branchCenters[dragNode.branch]){branchCenters[dragNode.branch].x=p.x;branchCenters[dragNode.branch].y=p.y;rebuildTargets()}}return}if(pan){moved=true;const rect=svg.getBoundingClientRect();vb.x=pan.vx-(e.clientX-pan.x)/rect.width*vb.w;vb.y=pan.vy-(e.clientY-pan.y)/rect.height*vb.h;setViewBox()}});svg.addEventListener('pointerup',e=>{if(dragNode){const selected=dragNode;dragNode.g.releasePointerCapture(e.pointerId);dragNode=null;dragStart=null;if(!moved)focusNode(selected.id)}pan=null;viewport.classList.remove('dragging')});svg.addEventListener('pointerdown',e=>{if(e.target.closest('.node'))return;moved=false;pan={x:e.clientX,y:e.clientY,vx:vb.x,vy:vb.y};svg.setPointerCapture(e.pointerId);viewport.classList.add('dragging');clearFocus()});
viewport.addEventListener('wheel',e=>{e.preventDefault();const rect=svg.getBoundingClientRect(),mx=vb.x+(e.clientX-rect.left)/rect.width*vb.w,my=vb.y+(e.clientY-rect.top)/rect.height*vb.h,f=e.deltaY>0?1.13:.87,nw=Math.min(W*2.4,Math.max(420,vb.w*f)),nh=nw*rect.height/rect.width;vb={x:mx-(mx-vb.x)*nw/vb.w,y:my-(my-vb.y)*nh/vb.h,w:nw,h:nh};setViewBox()},{passive:false});
function setViewBox(){svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`)}function fit(){const shown=activeNodes();if(!shown.length){vb={x:0,y:0,w:W,h:H};return setViewBox()}let left=Math.min(...shown.map(n=>n.x-n.r))-85,right=Math.max(...shown.map(n=>n.x+n.r))+85,top=Math.min(...shown.map(n=>n.y-n.r))-85,bottom=Math.max(...shown.map(n=>n.y+n.r))+85,w=right-left,h=bottom-top;const rect=svg.getBoundingClientRect(),aspect=rect.width/Math.max(1,rect.height);if(w/h<aspect){const nw=h*aspect;left-=(nw-w)/2;w=nw}else{const nh=w/aspect;top-=(nh-h)/2;h=nh}vb={x:left,y:top,w,h};setViewBox()}
function focusNode(id){const related=new Set([id]);for(const e of activeLinks()){if(e.source===id)related.add(e.target);if(e.target===id)related.add(e.source)}for(const n of activeNodes()){n.g.classList.toggle('match',n.id===id);n.g.classList.toggle('dim',!related.has(n.id))}for(const e of activeLinks()){const dim=e.source!==id&&e.target!==id;e.line.classList.toggle('dim',dim)}}function clearFocus(){svg.querySelectorAll('.dim,.match').forEach(e=>e.classList.remove('dim','match'))}
function showTip(node){const counts=Object.entries(node.statusCounts).filter(([,v])=>v).map(([k,v])=>`${k}: ${v}`).join(' · '),availability=node.availability?`<br>Availability: ${escapeHtml(node.availability)}`:'';tip.innerHTML=`<strong>${escapeHtml(node.name)}</strong><span class="muted">${escapeHtml(node.type)}${node.state?` · ${escapeHtml(node.state)}`:''}</span>${availability}${counts?`<br>${escapeHtml(counts)}`:''}`;tip.style.display='block'}function moveTip(e){tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px'}function hideTip(){tip.style.display='none'}function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function applyHierarchyMode(mode){const all=mode==='all';for(const n of nodes){n.visible=all||['Bucket','L2','Driver'].includes(n.type);n.g.style.display=n.visible?'':'none'}for(const link of links){const visible=link.sourceNode.visible&&link.targetNode.visible;link.line.style.display=visible?'':'none'}document.getElementById('showAll').setAttribute('aria-pressed',String(all));document.getElementById('showTop').setAttribute('aria-pressed',String(!all));restart();fit()}
document.getElementById('fit').addEventListener('click',fit);document.getElementById('reheat').addEventListener('click',restart);document.getElementById('showAll').addEventListener('click',()=>applyHierarchyMode('all'));document.getElementById('showTop').addEventListener('click',()=>applyHierarchyMode('top'));window.addEventListener('resize',fit);fit();applyHierarchyMode('all');
</script></body></html>'''
    output = (
        template.replace("__NODES__", json.dumps(browser_nodes, ensure_ascii=False, separators=(",", ":")))
        .replace("__EDGES__", json.dumps(browser_edges, ensure_ascii=False, separators=(",", ":")))
        .replace("__NODE_COUNT__", str(len(nodes)))
        .replace("__EDGE_COUNT__", str(len(browser_edges)))
        .replace("__SOURCE__", html.escape(source_name))
    )
    destination.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("mvm_sample_demo_v3.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--include-png", action="store_true", help="Also render the optional static PNG.")
    args = parser.parse_args()

    rows = load_rows(args.input)
    nodes, edges = build_graph(rows)
    if args.include_png:
        positions, width, height, bands = build_layout(nodes, edges)
    else:
        positions, width, height, bands = {}, 0, 0, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "mvm_relationship_graph_demo_v3.png"
    html_path = args.output_dir / "mvm_relationship_graph_demo_v3.html"

    if args.include_png:
        if Image is None:
            raise SystemExit("PNG output requires Pillow. Install it with: pip install Pillow")
        render_png(nodes, edges, positions, width, height, bands, png_path, args.input.name)
    render_html(nodes, edges, positions, width, height, html_path, args.input.name)
    print(f"Rows: {len(rows)}")
    print(f"Nodes: {len(nodes)}")
    print(f"Relationships: {len(edges)}")
    if args.include_png:
        print(f"Canvas: {width} x {height}")
        print(f"PNG: {png_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
