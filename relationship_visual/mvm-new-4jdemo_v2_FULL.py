
"""
MVM Neo4j Demo v2

Reads:
    mvm_sample_demo_v2.csv

Generates:
    mvm_neo4j_nodes_demo_v2.csv
    mvm_neo4j_relationships_demo_v2.csv
    mvm_relationship_graph_demo_v2.png
    mvm_relationship_graph_demo_v2.html

Expected source columns:
    record_id
    gc_decomp_bucket
    gc_decomp_bucket_l2
    driver_group
    Topline KPI
    Sub KPI
    Drill-down KPI
    State

Color rules:
    Directional = yellow
    Answerable  = green
    Bucket / L2 = blue

Install:
    pip install graphviz

Optional for Neo4j loading:
    pip install neo4j
"""

import csv
import json
import re
from pathlib import Path

from graphviz import Digraph


BASE = Path(__file__).resolve().parent

INPUT_CSV = BASE / "mvm_sample_demo_v2.csv"
NODES_CSV = BASE / "mvm_neo4j_nodes_demo_v2.csv"
RELS_CSV = BASE / "mvm_neo4j_relationships_demo_v2.csv"
PNG_FILE = BASE / "mvm_relationship_graph_demo_v2.png"
HTML_FILE = BASE / "mvm_relationship_graph_demo_v2.html"

STATE_COLOR = {
    "Directional": "#F2C94C",
    "Answerable": "#6FB24A",
}

NEUTRAL_COLOR = "#4B78C2"
EDGE_COLOR = "#5B6B7A"


def safe_id(text):
    text = (text or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower() or "blank"


def wrap_label(text, width=19):
    words = (text or "").split()
    lines = []
    current = []
    n = 0

    for word in words:
        extra = len(word) + (1 if current else 0)

        if current and n + extra > width:
            lines.append(" ".join(current))
            current = [word]
            n = len(word)
        else:
            current.append(word)
            n += extra

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def build_neo4j_csvs():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_CSV}\n"
            "Put mvm_sample_demo_v2.csv in the same folder as this script."
        )

    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input CSV is empty.")

    required = {
        "record_id",
        "gc_decomp_bucket",
        "gc_decomp_bucket_l2",
        "driver_group",
        "Topline KPI",
        "Sub KPI",
        "Drill-down KPI",
        "State",
    }

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    nodes = {}
    rels = {}

    def add_node(node_id, name, node_type, state="", color="", record_id="", level=0):
        if node_id not in nodes:
            nodes[node_id] = {
                "node_id:ID": node_id,
                "name": name,
                "node_type:LABEL": node_type,
                "state": state,
                "color": color,
                "record_id": record_id,
                "level": level,
            }

    def add_rel(start_id, end_id, rel_type):
        rels[(start_id, end_id, rel_type)] = {
            ":START_ID": start_id,
            ":END_ID": end_id,
            ":TYPE": rel_type,
        }

    for row in rows:
        record_id = row["record_id"].strip()
        bucket = row["gc_decomp_bucket"].strip()
        l2 = row["gc_decomp_bucket_l2"].strip()
        driver = row["driver_group"].strip()
        top = row["Topline KPI"].strip()
        sub = row["Sub KPI"].strip()
        drill = row["Drill-down KPI"].strip()
        state = row["State"].strip()

        color = STATE_COLOR.get(state, "#BDBDBD")

        bucket_id = f"bucket|{safe_id(bucket)}"
        l2_id = f"l2|{safe_id(bucket)}|{safe_id(l2)}"
        driver_id = f"driver|{safe_id(bucket)}|{safe_id(l2)}|{safe_id(driver)}"
        top_id = (
            f"topline_kpi|{safe_id(bucket)}|{safe_id(l2)}|"
            f"{safe_id(driver)}|{safe_id(top)}"
        )
        sub_id = (
            f"sub_kpi|{safe_id(bucket)}|{safe_id(l2)}|"
            f"{safe_id(driver)}|{safe_id(top)}|{safe_id(sub)}"
        )

        add_node(bucket_id, bucket, "GCDecompBucket", color=NEUTRAL_COLOR, level=0)
        add_node(l2_id, l2, "GCDecompBucketL2", color=NEUTRAL_COLOR, level=1)
        add_node(driver_id, driver, "DriverGroup", state, color, record_id, 2)
        add_node(top_id, top, "ToplineKPI", state, color, record_id, 3)
        add_node(sub_id, sub, "SubKPI", state, color, record_id, 4)

        add_rel(bucket_id, l2_id, "HAS_L2")
        add_rel(l2_id, driver_id, "HAS_DRIVER_GROUP")
        add_rel(driver_id, top_id, "HAS_TOPLINE_KPI")
        add_rel(top_id, sub_id, "HAS_SUB_KPI")

        if drill:
            drill_id = (
                f"drill_down_kpi|{safe_id(bucket)}|{safe_id(l2)}|"
                f"{safe_id(driver)}|{safe_id(top)}|"
                f"{safe_id(sub)}|{safe_id(drill)}"
            )

            add_node(
                drill_id,
                drill,
                "DrillDownKPI",
                state,
                color,
                record_id,
                5,
            )

            add_rel(
                sub_id,
                drill_id,
                "HAS_DRILLDOWN_KPI",
            )

    with NODES_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_id:ID",
                "name",
                "node_type:LABEL",
                "state",
                "color",
                "record_id",
                "level",
            ],
        )
        writer.writeheader()
        writer.writerows(nodes.values())

    with RELS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                ":START_ID",
                ":END_ID",
                ":TYPE",
            ],
        )
        writer.writeheader()
        writer.writerows(rels.values())

    return list(nodes.values()), list(rels.values())


def render_png(nodes, relationships):
    dot = Digraph("MVM_Demo", format="png")

    dot.attr(
        rankdir="LR",
        bgcolor="white",
        splines="spline",
        overlap="false",
        nodesep="0.85",
        ranksep="1.35",
        pad="0.4",
        dpi="180",
    )

    dot.attr(
        "node",
        shape="circle",
        style="filled",
        fontname="Arial",
        fontsize="11",
        margin="0.15",
        penwidth="1.25",
    )

    dot.attr(
        "edge",
        color=EDGE_COLOR,
        penwidth="1.1",
        arrowsize="0.75",
    )

    for node in nodes:
        neutral = node["node_type:LABEL"] in {
            "GCDecompBucket",
            "GCDecompBucketL2",
        }

        dot.node(
            node["node_id:ID"],
            label=wrap_label(node["name"]),
            fillcolor=node["color"],
            color="#333333",
            fontcolor="white" if neutral else "#111111",
        )

    for rel in relationships:
        dot.edge(
            rel[":START_ID"],
            rel[":END_ID"],
        )

    dot.render(
        str(PNG_FILE.with_suffix("")),
        cleanup=True,
    )


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MVM Relationship Graph Demo</title>
<style>
:root {
  --blue:#4B78C2;
  --yellow:#F2C94C;
  --green:#6FB24A;
  --edge:#5B6B7A;
  --text:#17202A;
  --muted:#667085;
  --panel:#F8FAFC;
  --border:#D0D5DD;
}
* { box-sizing:border-box; }
body {
  margin:0;
  font-family:Arial,Helvetica,sans-serif;
  color:var(--text);
  background:#fff;
}
.topbar {
  position:sticky;
  top:0;
  z-index:5;
  display:flex;
  flex-wrap:wrap;
  gap:10px 16px;
  align-items:center;
  padding:12px 16px;
  border-bottom:1px solid var(--border);
  background:rgba(255,255,255,.97);
}
.title { font-weight:700; margin-right:auto; }
.hint { color:var(--muted); font-size:13px; }
button {
  border:1px solid var(--border);
  background:#fff;
  border-radius:8px;
  padding:7px 11px;
  cursor:pointer;
  font-size:13px;
}
button:hover { background:var(--panel); }
.legend {
  display:flex;
  gap:14px;
  flex-wrap:wrap;
  align-items:center;
  font-size:13px;
}
.legend-item {
  display:inline-flex;
  gap:6px;
  align-items:center;
}
.dot {
  width:13px;
  height:13px;
  border-radius:50%;
  display:inline-block;
  border:1px solid #333;
}
#viewport {
  width:100%;
  height:calc(100vh - 96px);
  overflow:auto;
  position:relative;
  background:
    linear-gradient(#fff,#fff) padding-box,
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 39px,
      rgba(75,120,194,.04) 40px
    );
}
#graph { display:block; background:#fff; }
.edge {
  fill:none;
  stroke:var(--edge);
  stroke-width:1.6;
}
.node circle {
  stroke:#333;
  stroke-width:1.4;
  cursor:pointer;
}
.node text {
  text-anchor:middle;
  dominant-baseline:middle;
  pointer-events:none;
  font-size:13px;
  font-weight:600;
}
.node .type-label {
  font-size:10px;
  font-weight:400;
  fill:#667085;
}
.node:hover circle { stroke-width:2.5; }
.tooltip {
  position:fixed;
  display:none;
  z-index:20;
  max-width:320px;
  padding:8px 10px;
  border-radius:8px;
  background:#111827;
  color:white;
  font-size:12px;
  pointer-events:none;
}
</style>
</head>
<body>

<div class="topbar">
  <div class="title">MVM Relationship Graph Demo</div>

  <button id="collapseBtn">Collapse to Driver Groups</button>
  <button id="expandBtn">Expand All</button>
  <button id="fitBtn">Fit Screen</button>
  <button id="actualBtn">Actual Size</button>

  <div class="legend">
    <span class="legend-item">
      <span class="dot" style="background:var(--blue)"></span>
      Hierarchy
    </span>
    <span class="legend-item">
      <span class="dot" style="background:var(--yellow)"></span>
      Directional
    </span>
    <span class="legend-item">
      <span class="dot" style="background:var(--green)"></span>
      Answerable
    </span>
  </div>

  <div class="hint">
    Double-click a node to expand/collapse its next level.
  </div>
</div>

<div id="viewport">
  <svg id="graph" role="img" aria-label="Interactive MVM relationship graph"></svg>
</div>

<div id="tooltip" class="tooltip"></div>

<script>
const allNodes = __NODES_JSON__;
const allEdges = __EDGES_JSON__;

const nodeMap = new Map(allNodes.map(n => [n.id, n]));
const children = new Map();
const parent = new Map();

for (const edge of allEdges) {
  if (!children.has(edge.from)) {
    children.set(edge.from, []);
  }
  children.get(edge.from).push(edge.to);
  parent.set(edge.to, edge.from);
}

const roots = allNodes
  .filter(n => !parent.has(n.id))
  .map(n => n.id);

// Show through Driver Group initially.
const expanded = new Set(
  allNodes
    .filter(n => n.level <= 1)
    .map(n => n.id)
);

const svg = document.getElementById("graph");
const viewport = document.getElementById("viewport");
const tooltip = document.getElementById("tooltip");
const NS = "http://www.w3.org/2000/svg";

const NODE_RADIUS = 58;
const LEVEL_GAP = 290;
const LEAF_GAP = 145;
const MARGIN_X = 95;
const MARGIN_Y = 85;

let scaleMode = "actual";


function visibleNodeIds() {
  const visible = new Set();

  function walk(id) {
    visible.add(id);

    if (!expanded.has(id)) {
      return;
    }

    for (const child of (children.get(id) || [])) {
      walk(child);
    }
  }

  for (const root of roots) {
    walk(root);
  }

  return visible;
}


function visibleChildren(id, visible) {
  return (children.get(id) || [])
    .filter(child => visible.has(child));
}


function layoutVisibleTree(visible) {
  let leafIndex = 0;
  const positions = new Map();

  function assign(id, depth) {
    const kids = visibleChildren(id, visible);
    let y;

    if (kids.length === 0) {
      y = MARGIN_Y + leafIndex * LEAF_GAP;
      leafIndex += 1;
    } else {
      const childYs = kids.map(
        child => assign(child, depth + 1)
      );

      y = childYs.reduce((a, b) => a + b, 0) / childYs.length;
    }

    positions.set(id, {
      x: MARGIN_X + depth * LEVEL_GAP,
      y: y
    });

    return y;
  }

  for (const root of roots) {
    assign(root, 0);
  }

  const maxDepth = Math.max(
    ...[...visible].map(
      id => nodeMap.get(id).level
    )
  );

  const width =
    MARGIN_X * 2
    + maxDepth * LEVEL_GAP
    + NODE_RADIUS * 2;

  const height = Math.max(
    620,
    MARGIN_Y * 2
    + Math.max(1, leafIndex - 1) * LEAF_GAP
    + NODE_RADIUS * 2
  );

  return {positions, width, height};
}


function svgEl(tag, attrs={}) {
  const el = document.createElementNS(NS, tag);

  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }

  return el;
}


function wrapText(text, maxChars=18) {
  const words = text.split(/\s+/);
  const lines = [];
  let line = "";

  for (const word of words) {
    if (!line) {
      line = word;
    } else if ((line + " " + word).length <= maxChars) {
      line += " " + word;
    } else {
      lines.push(line);
      line = word;
    }
  }

  if (line) {
    lines.push(line);
  }

  return lines.slice(0, 5);
}


function createMarkerDefs() {
  const defs = svgEl("defs");

  const marker = svgEl("marker", {
    id: "arrow",
    markerWidth: 8,
    markerHeight: 8,
    refX: 7,
    refY: 4,
    orient: "auto",
    markerUnits: "strokeWidth"
  });

  const path = svgEl("path", {
    d: "M0,0 L8,4 L0,8 z",
    fill: "#5B6B7A"
  });

  marker.appendChild(path);
  defs.appendChild(marker);
  svg.appendChild(defs);
}


function edgePath(a, b) {
  const x1 = a.x + NODE_RADIUS;
  const y1 = a.y;
  const x2 = b.x - NODE_RADIUS;
  const y2 = b.y;
  const midX = (x1 + x2) / 2;

  return (
    `M ${x1} ${y1} `
    + `C ${midX} ${y1}, `
    + `${midX} ${y2}, `
    + `${x2} ${y2}`
  );
}


function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}


function centerOnNode(id) {
  requestAnimationFrame(() => {
    const visible = visibleNodeIds();
    const {positions} = layoutVisibleTree(visible);
    const pos = positions.get(id);

    if (!pos) {
      return;
    }

    viewport.scrollTo({
      left: Math.max(
        0,
        pos.x - viewport.clientWidth * 0.35
      ),
      top: Math.max(
        0,
        pos.y - viewport.clientHeight * 0.5
      ),
      behavior: "smooth"
    });
  });
}


function applyScaleMode() {
  svg.style.transformOrigin = "top left";
  svg.style.transform = "none";

  if (scaleMode === "fit") {
    const svgWidth = parseFloat(svg.getAttribute("width"));
    const svgHeight = parseFloat(svg.getAttribute("height"));

    const scaleX =
      (viewport.clientWidth - 30) / svgWidth;

    const scaleY =
      (viewport.clientHeight - 30) / svgHeight;

    const scale = Math.max(
      0.60,
      Math.min(
        1,
        scaleX,
        scaleY
      )
    );

    svg.style.transform = `scale(${scale})`;
  }
}


function render() {
  const visible = visibleNodeIds();
  const {positions, width, height} =
    layoutVisibleTree(visible);

  svg.innerHTML = "";
  createMarkerDefs();

  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute(
    "viewBox",
    `0 0 ${width} ${height}`
  );

  // Draw relationships first.
  for (const edge of allEdges) {
    if (
      !visible.has(edge.from)
      || !visible.has(edge.to)
    ) {
      continue;
    }

    const a = positions.get(edge.from);
    const b = positions.get(edge.to);

    svg.appendChild(
      svgEl("path", {
        class: "edge",
        d: edgePath(a, b),
        "marker-end": "url(#arrow)"
      })
    );
  }

  // Draw nodes.
  for (const id of visible) {
    const node = nodeMap.get(id);
    const pos = positions.get(id);

    const group = svgEl("g", {
      class: "node",
      transform: `translate(${pos.x},${pos.y})`,
      "data-id": id
    });

    group.appendChild(
      svgEl("circle", {
        r: NODE_RADIUS,
        fill: node.color || "#D9D9D9"
      })
    );

    const lines = wrapText(
      node.name,
      18
    );

    const text = svgEl("text", {
      fill:
        (
          node.type === "GCDecompBucket"
          || node.type === "GCDecompBucketL2"
        )
        ? "#FFFFFF"
        : "#111111"
    });

    const lineHeight = 15;

    const startY =
      -((lines.length - 1) * lineHeight) / 2
      - 3;

    lines.forEach((line, index) => {
      const span = svgEl("tspan", {
        x: 0,
        y: startY + index * lineHeight
      });

      span.textContent = line;
      text.appendChild(span);
    });

    group.appendChild(text);

    const typeText = svgEl("text", {
      class: "type-label",
      x: 0,
      y: NODE_RADIUS + 18
    });

    typeText.textContent = node.type;
    group.appendChild(typeText);

    // Double-click to expand/collapse the next layer.
    group.addEventListener(
      "dblclick",
      event => {
        event.stopPropagation();

        const hasKids =
          (children.get(id) || []).length > 0;

        if (!hasKids) {
          return;
        }

        if (expanded.has(id)) {
          expanded.delete(id);
        } else {
          expanded.add(id);
        }

        scaleMode = "actual";
        render();
        centerOnNode(id);
      }
    );

    // Hover tooltip.
    group.addEventListener(
      "mouseenter",
      () => {
        tooltip.style.display = "block";

        tooltip.innerHTML =
          `<strong>${escapeHtml(node.name)}</strong><br>`
          + `Type: ${escapeHtml(node.type)}`
          + (
            node.state
            ? `<br>State: ${escapeHtml(node.state)}`
            : ""
          );
      }
    );

    group.addEventListener(
      "mousemove",
      event => {
        tooltip.style.left =
          (event.clientX + 14) + "px";

        tooltip.style.top =
          (event.clientY + 14) + "px";
      }
    );

    group.addEventListener(
      "mouseleave",
      () => {
        tooltip.style.display = "none";
      }
    );

    svg.appendChild(group);
  }

  applyScaleMode();
}


document
  .getElementById("expandBtn")
  .addEventListener(
    "click",
    () => {
      for (const node of allNodes) {
        if (
          (children.get(node.id) || []).length > 0
        ) {
          expanded.add(node.id);
        }
      }

      scaleMode = "actual";
      render();

      viewport.scrollTo({
        left: 0,
        top: 0,
        behavior: "smooth"
      });
    }
  );


document
  .getElementById("collapseBtn")
  .addEventListener(
    "click",
    () => {
      expanded.clear();

      for (const node of allNodes) {
        if (node.level <= 1) {
          expanded.add(node.id);
        }
      }

      scaleMode = "actual";
      render();

      viewport.scrollTo({
        left: 0,
        top: 0,
        behavior: "smooth"
      });
    }
  );


document
  .getElementById("fitBtn")
  .addEventListener(
    "click",
    () => {
      scaleMode = "fit";
      applyScaleMode();

      viewport.scrollTo({
        left: 0,
        top: 0,
        behavior: "smooth"
      });
    }
  );


document
  .getElementById("actualBtn")
  .addEventListener(
    "click",
    () => {
      scaleMode = "actual";
      applyScaleMode();
    }
  );


window.addEventListener(
  "resize",
  () => {
    if (scaleMode === "fit") {
      applyScaleMode();
    }
  }
);


render();
</script>

</body>
</html>
"""


def render_interactive_html(
    nodes,
    relationships,
):
    browser_nodes = [
        {
            "id": node["node_id:ID"],
            "name": node["name"],
            "type": node["node_type:LABEL"],
            "state": node["state"],
            "color": node["color"],
            "level": int(node["level"]),
        }
        for node in nodes
    ]

    browser_edges = [
        {
            "from": rel[":START_ID"],
            "to": rel[":END_ID"],
            "type": rel[":TYPE"],
        }
        for rel in relationships
    ]

    html = HTML_TEMPLATE.replace(
        "__NODES_JSON__",
        json.dumps(
            browser_nodes,
            ensure_ascii=False,
        ),
    ).replace(
        "__EDGES_JSON__",
        json.dumps(
            browser_edges,
            ensure_ascii=False,
        ),
    )

    HTML_FILE.write_text(
        html,
        encoding="utf-8",
    )


def load_to_neo4j(
    uri,
    user,
    password,
    database="neo4j",
):
    """
    Optional Neo4j loader.

    Example:
        load_to_neo4j(
            uri="neo4j://localhost:7687",
            user="neo4j",
            password="YOUR_PASSWORD"
        )
    """

    from neo4j import GraphDatabase

    allowed_labels = {
        "GCDecompBucket",
        "GCDecompBucketL2",
        "DriverGroup",
        "ToplineKPI",
        "SubKPI",
        "DrillDownKPI",
    }

    allowed_rel_types = {
        "HAS_L2",
        "HAS_DRIVER_GROUP",
        "HAS_TOPLINE_KPI",
        "HAS_SUB_KPI",
        "HAS_DRILLDOWN_KPI",
    }

    db = GraphDatabase.driver(
        uri,
        auth=(user, password),
    )

    with db.session(
        database=database
    ) as session:

        session.run(
            """
            CREATE CONSTRAINT mvm_demo_node_id IF NOT EXISTS
            FOR (n:MVMNode)
            REQUIRE n.node_id IS UNIQUE
            """
        )

        with NODES_CSV.open(
            newline="",
            encoding="utf-8-sig",
        ) as f:

            for row in csv.DictReader(f):

                session.run(
                    """
                    MERGE (
                        n:MVMNode {
                            node_id: $node_id
                        }
                    )

                    SET
                        n.name = $name,
                        n.state = $state,
                        n.color = $color,
                        n.record_id = $record_id,
                        n.level = toInteger($level)
                    """,
                    node_id=row["node_id:ID"],
                    name=row["name"],
                    state=row["state"],
                    color=row["color"],
                    record_id=row["record_id"],
                    level=row["level"],
                )

                label = row["node_type:LABEL"]

                if label not in allowed_labels:
                    raise ValueError(
                        f"Unexpected node label: {label}"
                    )

                session.run(
                    f"""
                    MATCH (
                        n:MVMNode {{
                            node_id: $node_id
                        }}
                    )
                    SET n:{label}
                    """,
                    node_id=row["node_id:ID"],
                )

        with RELS_CSV.open(
            newline="",
            encoding="utf-8-sig",
        ) as f:

            for row in csv.DictReader(f):

                rel_type = row[":TYPE"]

                if rel_type not in allowed_rel_types:
                    raise ValueError(
                        f"Unexpected relationship type: {rel_type}"
                    )

                session.run(
                    f"""
                    MATCH (
                        a:MVMNode {{
                            node_id: $start_id
                        }}
                    )

                    MATCH (
                        b:MVMNode {{
                            node_id: $end_id
                        }}
                    )

                    MERGE (
                        a
                    )-[:{rel_type}]->(
                        b
                    )
                    """,
                    start_id=row[":START_ID"],
                    end_id=row[":END_ID"],
                )

    db.close()


if __name__ == "__main__":
    nodes, relationships = (
        build_neo4j_csvs()
    )

    render_png(
        nodes,
        relationships,
    )

    render_interactive_html(
        nodes,
        relationships,
    )

    print()
    print("Created:")
    print(f"  {NODES_CSV}")
    print(f"  {RELS_CSV}")
    print(f"  {PNG_FILE}")
    print(f"  {HTML_FILE}")
    print()

    # Optional Neo4j load:
    #
    # load_to_neo4j(
    #     uri="neo4j://localhost:7687",
    #     user="neo4j",
    #     password="YOUR_PASSWORD",
    # )
