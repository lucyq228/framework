
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from accounting_decomp import additive_hierarchy_decomp, ratio_hierarchy_decomp

# Clean tree: one consistent semantic palette (override via ``semantic_palette`` on plot).
# Negative → light red, positive → light green, near-zero / missing → neutral, root → blue-grey.
SEMANTIC_TREE_COLORS: Dict[str, str] = {
    "root": "#E8EDF4",
    "positive": "#C8E6C9",
    "negative": "#FFCDD2",
    "neutral": "#F0EDDE",
}
_SEMANTIC_KEYS = frozenset(SEMANTIC_TREE_COLORS.keys())

_CLR_EDGE = "#BBBBBB"
_CLR_AXIS = "#999999"


def _merge_semantic_palette(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a full palette dict; ``overrides`` may set any of root/positive/negative/neutral."""
    out = dict(SEMANTIC_TREE_COLORS)
    if not overrides:
        return out
    for k, v in overrides.items():
        if k in _SEMANTIC_KEYS and isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


@dataclass
class TreeNode:
    name: str
    value_text: str
    meta_lines: List[str] = field(default_factory=list)
    children: List["TreeNode"] = field(default_factory=list)
    color: Optional[str] = None
    #: Among **siblings**: ``"additive"`` = children partition / sum to the parent (solid box + edges);
    #: ``"parallel"`` = alternative views of the same contribution (dotted box + edges). Fills unchanged.
    sibling_relation: Optional[str] = None


def make_node_lines(node: TreeNode, *, include_meta_lines: bool = True, max_meta_lines: int = 2) -> List[str]:
    """Stacked label lines: name, value, then up to ``max_meta_lines`` meta lines (reference-style)."""
    lines = [node.name, node.value_text]
    if include_meta_lines and node.meta_lines:
        lines.extend(node.meta_lines[:max_meta_lines])
    return lines


def _parse_first_numeric(value_text: str) -> Optional[float]:
    """Parse the first signed number from a display string (e.g. ``+1.12%``, ``-2.10 pp``)."""
    if not value_text or not str(value_text).strip():
        return None
    s = str(value_text).strip()
    if s in ("—", "-"):
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return None
    return float(m.group(0))


def _node_fill_color(
    value_text: str,
    *,
    depth: int,
    explicit: Optional[str],
    sentiment_threshold: float = 0.5,
    palette: Dict[str, str],
) -> str:
    if explicit:
        return explicit
    if depth == 0:
        return palette["root"]
    v = _parse_first_numeric(value_text)
    if v is None:
        return palette["neutral"]
    if v > sentiment_threshold:
        return palette["positive"]
    if v < -sentiment_threshold:
        return palette["negative"]
    return palette["neutral"]


def _linestyle_for_sibling_relation(relation: Optional[str]) -> str:
    if relation == "parallel":
        return "dotted"
    return "solid"


def _max_tree_depth(node: TreeNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_max_tree_depth(c) for c in node.children)


def _set_node_depth(node: TreeNode, depth: int, out: Dict[int, int]) -> None:
    out[id(node)] = depth
    for c in node.children:
        _set_node_depth(c, depth + 1, out)


def _collect_leaves_dfs(node: TreeNode) -> List[TreeNode]:
    """Leaves in depth-first (preorder) order (same order as old uniform-spacing layout)."""
    if not node.children:
        return [node]
    out: List[TreeNode] = []
    for c in node.children:
        out.extend(_collect_leaves_dfs(c))
    return out


def _leaf_gap_pad(*, n_leaves: int, layout_scale: float, presentation: bool) -> float:
    """Base vertical gap between leaf box centers (before ``leaf_spacing`` add-on)."""
    base = 0.42 * layout_scale * (1.35 if presentation else 1.1)
    return base + 0.05 * layout_scale * max(0, n_leaves - 6)


def _build_y_map_height_aware(
    root: TreeNode,
    metrics_by_id: Dict[int, dict],
    *,
    y_margin: float,
    gap_pad: float,
) -> Dict[int, float]:
    """Place leaf centers so adjacent leaves never overlap; internal y = mean(children)."""
    leaves = _collect_leaves_dfs(root)
    y_map: Dict[int, float] = {}
    cy = y_margin + metrics_by_id[id(leaves[0])]["box_h"] / 2.0
    y_map[id(leaves[0])] = cy
    for i in range(1, len(leaves)):
        h_prev = metrics_by_id[id(leaves[i - 1])]["box_h"]
        h_curr = metrics_by_id[id(leaves[i])]["box_h"]
        cy = cy + (h_prev + h_curr) / 2.0 + gap_pad
        y_map[id(leaves[i])] = cy

    def _fill_internal_y(n: TreeNode) -> float:
        if not n.children:
            return y_map[id(n)]
        child_y = [_fill_internal_y(c) for c in n.children]
        y = float(np.mean(child_y))
        y_map[id(n)] = y
        return y

    _fill_internal_y(root)
    return y_map


def _iter_nodes_preorder(node: TreeNode) -> List[TreeNode]:
    out: List[TreeNode] = [node]
    for c in node.children:
        out.extend(_iter_nodes_preorder(c))
    return out


def _measure_text_bbox_data(
    ax,
    fig,
    s: str,
    fontsize: float,
    fontweight: str,
) -> Tuple[float, float]:
    """Return text width and height in **data** coordinates (requires ``fig.canvas.draw()``)."""
    if not s:
        return (1e-3, 1e-3)
    t = ax.text(
        0.0,
        0.0,
        s,
        fontsize=fontsize,
        fontweight=fontweight,
        ha="center",
        va="center",
        alpha=0.0,
        family="sans-serif",
    )
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    t.remove()
    inv = ax.transData.inverted()
    p0 = inv.transform((bb.x0, bb.y0))
    p1 = inv.transform((bb.x1, bb.y1))
    w = abs(p1[0] - p0[0])
    h = abs(p1[1] - p0[1])
    return (max(w, 1e-6), max(h, 1e-6))


def _draw_node_measured(
    ax,
    cx: float,
    cy: float,
    lines: Sequence[str],
    line_dims: Sequence[Tuple[float, float]],
    *,
    box_w: float,
    box_h: float,
    gap: float,
    facecolor: str,
    fontsize: float,
    meta_fontsize: float,
    layout_scale: float,
    edgecolor: str = _CLR_AXIS,
    edgewidth: float = 0.6,
    linestyle: str = "solid",
) -> None:
    """Draw a rounded box sized to measured text; lines are vertically centered without overlap."""
    pad = 0.08 * layout_scale
    patch = FancyBboxPatch(
        (cx - box_w / 2, cy - box_h / 2),
        box_w,
        box_h,
        boxstyle=f"round,pad={pad}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=edgewidth * max(1.0, layout_scale * 0.85),
        linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(patch)
    total_txt_h = sum(h for _, h in line_dims) + gap * max(0, len(lines) - 1)
    y = cy + total_txt_h / 2 - line_dims[0][1] / 2
    for i, line in enumerate(lines):
        fs = fontsize if i <= 1 else meta_fontsize
        ax.text(
            cx,
            y,
            line,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if i == 0 else "normal",
            family="sans-serif",
            color="#222222",
            zorder=4,
        )
        if i + 1 < len(lines):
            y -= line_dims[i][1] / 2 + gap + line_dims[i + 1][1] / 2


def _suggest_figsize_clean(
    x_span: float,
    y_span: float,
    *,
    n_levels: int,
    presentation: bool,
    figsize: Optional[Tuple[float, float]],
    n_leaves: Optional[int] = None,
    max_fig_height_inches: float = 40.0,
) -> Tuple[float, float]:
    """Pick figure size so the axes aspect matches the data (avoids a wide, flat 'strip' in PPT)."""
    if figsize is not None:
        return float(figsize[0]), float(figsize[1])
    y_span = max(float(y_span), 1e-6)
    x_span = max(float(x_span), 1e-6)
    ar = x_span / y_span
    h_min = 9.5 if presentation else 7.0
    w_cap = 34.0 if presentation else 24.0
    w_seed = min(w_cap, max(12.0 if presentation else 9.0, 5.0 + n_levels * (3.0 if presentation else 2.6)))
    h = max(w_seed / ar, h_min)
    w = h * ar
    if w > w_cap:
        w = w_cap
        h = max(w / ar, h_min)
    if w < 10.0:
        w = 10.0
        h = max(w / ar, h_min)
    h_cap = 22.0
    if n_leaves is not None and n_leaves > 12:
        h_cap = float(min(max_fig_height_inches, max(22.0, 12.0 + 0.32 * n_leaves)))
    h = float(min(max(h, h_min), h_cap))
    w = float(min(max(w, 10.0), w_cap))
    return (w, h)


def _draw_edge_orthogonal(
    ax,
    x0_right: float,
    y0: float,
    x1_left: float,
    y1: float,
    *,
    zorder: float = 1,
    linewidth: float = 1.1,
    linestyle: str = "solid",
) -> None:
    xm = (x0_right + x1_left) / 2
    ax.plot(
        [x0_right, xm, xm, x1_left],
        [y0, y0, y1, y1],
        color=_CLR_EDGE,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=zorder,
    )


def plot_decomposition_tree(
    root: TreeNode,
    save_path: Optional[str | Path] = None,
    *,
    title: str = "Identity-Based Decomposition Tree",
    subtitle: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    font_size: int = 18,
    show: bool = True,
    show_meta_lines: bool = True,
    respect_node_color: bool = False,
    semantic_palette: Optional[Dict[str, str]] = None,
    sentiment_threshold: float = 0.5,
    encode_structural_lines: bool = True,
    leaf_spacing: Optional[float] = None,
    min_leaf_edge_gap: float = 0.18,
    level_dx: float = 5.0,
    x_margin: float = 1.35,
    root_width: float = 3.4,
    child_width: float = 3.15,
    depth_width_factor: float = 0.07,
    max_fig_height_inches: float = 40.0,
    presentation: bool = True,
    layout_scale: Optional[float] = None,
    ppt_scale: float = 2.05,
    title_fontsize: Optional[float] = None,
    subtitle_fontsize: Optional[float] = None,
    dpi: int = 240,
) -> Optional[Path]:
    """Render the decomposition tree (horizontal levels, measured boxes, orthogonal edges).

    **Semantic** fills by default: negative **value_text** → light red, positive → light green,
    near-zero / missing → neutral warm grey, root (depth 0) → light blue-grey. See
    ``SEMANTIC_TREE_COLORS`` and ``semantic_palette``.

    ``presentation=True`` (default): scales boxes, spacing, and type for **slide / PPT** use
    (larger fonts, wider nodes, higher export ``dpi``). Figure **width/height** follow the
    data extent so wide trees get enough **height** (avoids a tiny horizontal strip with empty
    space). Set ``presentation=False`` for a denser notebook-sized figure. Override sizing with
    ``layout_scale`` / ``ppt_scale`` (defaults are tuned for full-slide PNGs), or pass ``figsize``
    explicitly.

    ``respect_node_color`` (default ``False``): when ``True``, ``TreeNode.color`` overrides the
    semantic palette for that node. When ``False``, fills follow only ``value_text`` + depth.

    ``semantic_palette``: optional partial override of ``SEMANTIC_TREE_COLORS`` keys
    (``root``, ``positive``, ``negative``, ``neutral``).

    ``encode_structural_lines``: when ``True`` (default), ``TreeNode.sibling_relation`` controls only
    **outline and connector** style: ``"additive"`` → solid box border and solid edges to parent;
    ``"parallel"`` → dotted border and dotted edges. Node **fill colors** stay sentiment-based.

    When ``show_meta_lines`` is False, each node shows only name + value (no meta lines).

    Box **width** and **height** follow **measured** text extents; line spacing uses measured
    heights plus a gap (avoids squeezed or overlapping text).

    **Layout:** Vertical positions use **height-aware** spacing between leaf box centers (each leaf’s
    measured ``box_h``), so deep / wide trees do not stack overlapping boxes. Deeper columns use
    slightly wider boxes (``depth_width_factor``). ``max_fig_height_inches`` caps figure height when
    there are many leaves; increase it for very tall trees.

    **``leaf_spacing``** is **added on top of** the automatic gap (axis data units). Try ``1.0``–``2.5``
    for visibly looser vertical stacks.

    **``min_leaf_edge_gap``** is the minimum **clear space between adjacent leaf box edges**; the
    layout retries with a larger gap multiplier if edges would touch.
    """
    ls = layout_scale if layout_scale is not None else (ppt_scale if presentation else 1.0)
    edge_lw = 1.15 + 0.45 * (ls - 1.0)
    ts = title_fontsize if title_fontsize is not None else (22 if presentation else 14)
    ss = subtitle_fontsize if subtitle_fontsize is not None else (14.0 if presentation else 10.5)
    palette = _merge_semantic_palette(semantic_palette)

    depth_by_id: Dict[int, int] = {}
    _set_node_depth(root, 0, depth_by_id)
    max_depth = _max_tree_depth(root)
    n_levels = max_depth + 1

    level_dx_s = level_dx * ls
    x_margin_s = x_margin * ls
    root_width_s = root_width * ls
    child_width_s = child_width * ls

    def _max_lines_subtree(node: TreeNode) -> int:
        ml = len(make_node_lines(node, include_meta_lines=show_meta_lines))
        if not node.children:
            return ml
        return max(ml, max(_max_lines_subtree(c) for c in node.children))

    n_leaves = len(_collect_leaves_dfs(root))
    y_margin = 1.0 * ls
    y_span_est = max(16.0, n_leaves * 2.2 * ls + 8.0)

    x_centers = [x_margin_s + i * level_dx_s for i in range(n_levels)]

    def width_for_depth(d: int) -> float:
        base = root_width_s if d == 0 else child_width_s
        if d <= 0:
            return base
        return base * (1.0 + depth_width_factor * max(0, d - 1))

    x_right_est = x_centers[-1] + width_for_depth(max_depth) / 2 + 1.2 * ls
    use_figsize = _suggest_figsize_clean(
        x_right_est,
        y_span_est,
        n_levels=n_levels,
        presentation=presentation,
        figsize=figsize,
        n_leaves=n_leaves,
        max_fig_height_inches=max_fig_height_inches,
    )

    fig, ax = plt.subplots(figsize=use_figsize)
    ax.set_facecolor("#F3F3F5")
    fig.patch.set_facecolor("#EDEDF0")
    ax.axis("off")

    ax.set_xlim(0.0, x_right_est)
    ax.set_ylim(0.0, y_span_est)
    ax.invert_yaxis()
    fig.canvas.draw()

    node_title_fs = float(font_size) * (1.06 if presentation else 1.0)
    node_meta_fs = max(node_title_fs * 0.9, node_title_fs - 1.5)

    gap_ln = 0.28 * ls
    pad_x = 0.42 * ls
    pad_y = 0.32 * ls

    def _node_metrics(n: TreeNode) -> dict:
        lines = make_node_lines(n, include_meta_lines=show_meta_lines)
        d = depth_by_id[id(n)]
        line_dims: List[Tuple[float, float]] = []
        for i, line in enumerate(lines):
            fw = "bold" if i == 0 else "normal"
            fs = node_title_fs if i <= 1 else node_meta_fs
            line_dims.append(_measure_text_bbox_data(ax, fig, line, fs, fw))
        max_w = max((w for w, _ in line_dims), default=0.0)
        total_txt_h = sum(h for _, h in line_dims) + gap_ln * max(0, len(lines) - 1)
        floor_w = width_for_depth(d)
        box_w = max(floor_w, max_w + 2.0 * pad_x)
        box_h = max(ls * (0.46 + 0.36 * len(lines)), total_txt_h + 2.0 * pad_y)
        return {
            "lines": lines,
            "line_dims": line_dims,
            "box_w": float(box_w),
            "box_h": float(box_h),
        }

    def _rebuild_metrics() -> Dict[int, dict]:
        return {id(n): _node_metrics(n) for n in _iter_nodes_preorder(root)}

    metrics = _rebuild_metrics()
    gap_pad_base = _leaf_gap_pad(n_leaves=n_leaves, layout_scale=ls, presentation=presentation)
    gap_extra = float(leaf_spacing) if leaf_spacing is not None else 0.0
    gap_pad = gap_pad_base + gap_extra

    y_map: Dict[int, float] = {}
    for _attempt in range(5):
        y_map = _build_y_map_height_aware(
            root,
            metrics,
            y_margin=y_margin,
            gap_pad=gap_pad * (1.12**_attempt),
        )
        nodes_all = _iter_nodes_preorder(root)
        min_edge = min(y_map[id(n)] - metrics[id(n)]["box_h"] / 2 for n in nodes_all)
        if min_edge < y_margin:
            shift = y_margin - min_edge
            for k in y_map:
                y_map[k] += shift
        max_edge = max(y_map[id(n)] + metrics[id(n)]["box_h"] / 2 for n in nodes_all)
        y_span_ax = max_edge + y_margin * 1.35
        ax.set_ylim(0.0, y_span_ax)
        ax.invert_yaxis()
        fig.canvas.draw()
        metrics = _rebuild_metrics()
        leaves_ord = _collect_leaves_dfs(root)
        ok = True
        for i in range(len(leaves_ord) - 1):
            a, b = leaves_ord[i], leaves_ord[i + 1]
            dy = abs(y_map[id(b)] - y_map[id(a)])
            ha, hb = metrics[id(a)]["box_h"], metrics[id(b)]["box_h"]
            # Require a visible gap between box edges, not merely non-overlap (float / font tweaks).
            need = (ha + hb) / 2 + max(0.12 * ls, float(min_leaf_edge_gap))
            if dy < need - 1e-6:
                ok = False
                break
        if ok:
            break

    x_right = max(
        x_centers[depth_by_id[id(n)]] + metrics[id(n)]["box_w"] / 2.0 for n in _iter_nodes_preorder(root)
    ) + 1.0 * ls
    ax.set_xlim(0.0, x_right)
    fig.canvas.draw()
    metrics = _rebuild_metrics()
    x_right = max(
        x_centers[depth_by_id[id(n)]] + metrics[id(n)]["box_w"] / 2.0 for n in _iter_nodes_preorder(root)
    ) + 1.0 * ls
    ax.set_xlim(0.0, x_right)

    use_figsize2 = _suggest_figsize_clean(
        x_right,
        y_span_ax,
        n_levels=n_levels,
        presentation=presentation,
        figsize=figsize,
        n_leaves=n_leaves,
        max_fig_height_inches=max_fig_height_inches,
    )
    if figsize is None and (use_figsize2[0] != use_figsize[0] or use_figsize2[1] != use_figsize[1]):
        fig.set_size_inches(use_figsize2[0], use_figsize2[1], forward=True)
        fig.canvas.draw()
        metrics = _rebuild_metrics()
        x_right = max(
            x_centers[depth_by_id[id(n)]] + metrics[id(n)]["box_w"] / 2.0 for n in _iter_nodes_preorder(root)
        ) + 1.0 * ls
        ax.set_xlim(0.0, x_right)
        fig.canvas.draw()

    title_y = 0.98 if not subtitle else 0.96
    fig.suptitle(
        title,
        fontsize=ts,
        fontweight="semibold",
        color="#1a1a1a",
        y=title_y,
        family="sans-serif",
    )
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=ss, color="#444444", family="sans-serif")

    def draw_edges_clean(node: TreeNode) -> None:
        d0 = depth_by_id[id(node)]
        x0 = x_centers[d0]
        y0 = y_map[id(node)]
        w0 = metrics[id(node)]["box_w"]
        x0_right = x0 + w0 / 2
        for child in node.children:
            d1 = depth_by_id[id(child)]
            x1 = x_centers[d1]
            y1 = y_map[id(child)]
            w1 = metrics[id(child)]["box_w"]
            x1_left = x1 - w1 / 2
            rel = child.sibling_relation if encode_structural_lines else None
            els = _linestyle_for_sibling_relation(rel)
            _draw_edge_orthogonal(ax, x0_right, y0, x1_left, y1, zorder=1, linewidth=edge_lw, linestyle=els)
            draw_edges_clean(child)

    draw_edges_clean(root)

    def draw_nodes_clean(node: TreeNode) -> None:
        d = depth_by_id[id(node)]
        cx = x_centers[d]
        cy = y_map[id(node)]
        m = metrics[id(node)]
        explicit = node.color if respect_node_color else None
        fc = _node_fill_color(
            node.value_text,
            depth=d,
            explicit=explicit,
            sentiment_threshold=sentiment_threshold,
            palette=palette,
        )
        rel = node.sibling_relation if encode_structural_lines else None
        bls = _linestyle_for_sibling_relation(rel)
        _draw_node_measured(
            ax,
            cx,
            cy,
            m["lines"],
            m["line_dims"],
            box_w=m["box_w"],
            box_h=m["box_h"],
            gap=gap_ln,
            facecolor=fc,
            fontsize=node_title_fs,
            meta_fontsize=node_meta_fs,
            layout_scale=ls,
            linestyle=bls,
        )
        for child in node.children:
            draw_nodes_clean(child)

    draw_nodes_clean(root)

    ax.set_xlim(0.0, x_right)
    ax.set_ylim(0.0, y_span_ax)
    ax.invert_yaxis()

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.88 if subtitle else 0.92])
    out: Optional[Path] = None
    if save_path is not None:
        out = Path(save_path)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.2)
    if show:
        plt.show()
    plt.close(fig)
    return out


def _fmt_sales_pct(x: float, *, decimals: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{100.0 * x:.{decimals}f}%"


def _fmt_ratio_row(row: pd.Series, ratio_display: str) -> str:
    """Match ratio decomposition tables: ``total_pct_of_root_metric`` or ``total_abs_root``."""
    if ratio_display == "abs_root":
        v = float(row["total_abs_root"])
        if not np.isfinite(v):
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.6f}"
    v = float(row["total_pct_of_root_metric"])
    if not np.isfinite(v):
        return "—"
    return _fmt_sales_pct(v)


def _fmt_ac_mix_within_part(row: pd.Series, part: str, ratio_display: str) -> str:
    """Format **mix** or **within** slice of AC ratio decomposition (same scaling as ``_fmt_ratio_row`` total)."""
    part = part.lower()
    if part not in ("mix", "within"):
        raise ValueError('part must be "mix" or "within"')
    if ratio_display == "abs_root":
        k = "mix_abs_root" if part == "mix" else "within_abs_root"
        v = float(row[k])
        if not np.isfinite(v):
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.6f}"
    k = "mix_pct_of_root_metric" if part == "mix" else "within_pct_of_root_metric"
    v = float(row[k])
    if not np.isfinite(v):
        return "—"
    return _fmt_sales_pct(v)


def _dim_display_name(dim: str) -> str:
    return str(dim).replace("_", " ").strip().title()


def _build_ac_mix_within_panel(
    df: pd.DataFrame,
    outputs: Dict[str, pd.DataFrame],
    *,
    period_col: str,
    base_period: int,
    current_period: int,
    sales_metric: str,
    hierarchy_dims: Sequence[str],
    ratio_display: str,
    show_mix_within_leaves: bool = True,
) -> TreeNode:
    """AC → **dimension** (label) → segment values; optionally → Mix / Within under each segment."""
    if not hierarchy_dims:
        raise ValueError("hierarchy_dims must name at least one column")
    dim = str(hierarchy_dims[0])
    ac_hw = (dim,)
    display_name = _dim_display_name(dim)
    ac_f = _get_ratio_frame_on_demand(
        df,
        outputs,
        "ratio__ac",
        "ac",
        sales_metric,
        "gc",
        ac_hw,
        period_col,
        base_period,
        current_period,
    )
    l1 = ac_f[ac_f["level"] == 1]
    if not l1.empty:
        l1 = l1.assign(_abs_t=l1["total_abs_root"].abs()).sort_values("_abs_t", ascending=False)
    segments: List[TreeNode] = []
    for _, row in l1.iterrows():
        nv = str(row["node_value"])
        seg_meta = [f"Segment total AC (ratio__ac); {dim}={nv}"]
        if show_mix_within_leaves:
            seg_children: List[TreeNode] = [
                TreeNode(
                    name="Mix",
                    value_text=_fmt_ac_mix_within_part(row, "mix", ratio_display),
                    meta_lines=["(w1−w0)·AC_base; GC share shift × base AC"],
                    sibling_relation="additive",
                ),
                TreeNode(
                    name="Within",
                    value_text=_fmt_ac_mix_within_part(row, "within", ratio_display),
                    meta_lines=["w1·(AC_current−AC_base); weight × segment AC change"],
                    sibling_relation="additive",
                ),
            ]
        else:
            seg_children = []
            seg_meta.append("Mix/within breakdown hidden; segment total only")
        segments.append(
            TreeNode(
                name=nv,
                value_text=_fmt_ratio_row(row, ratio_display),
                meta_lines=seg_meta,
                children=seg_children,
                sibling_relation="additive",
            )
        )
    parent_val = "—"
    if not l1.empty:
        if ratio_display == "pct_of_base_ratio":
            total_pct = float(l1["total_pct_of_root_metric"].sum())
            if np.isfinite(total_pct):
                parent_val = _fmt_sales_pct(total_pct)
        else:
            s = float(l1["total_abs_root"].sum())
            if np.isfinite(s):
                parent_val = ("+" if s >= 0 else "") + f"{s:.6f}"
    return TreeNode(
        name=display_name,
        value_text=parent_val,
        meta_lines=[f"AC mix/within by {dim} (ratio__ac); segments sum to aggregate AC Δ"],
        children=segments,
        sibling_relation="parallel",
    )


def _aur_root_display_scalar(aur_row: pd.Series, ratio_display: str) -> float:
    k = "total_abs_root" if ratio_display == "abs_root" else "total_pct_of_root_metric"
    return float(aur_row[k])


def _fmt_aur_allocated_contribution(
    aur_root_scalar: float,
    ta: float,
    t_branch_abs: float,
    *,
    branch: str,
    ratio_display: str,
) -> str:
    """Allocate AUR root to list price vs discount so parts sum to AUR: ``A * (tg/ta)`` and ``A * (-td/ta)``.

    Level-1 segments use ``A * (tg_i/ta)`` and ``A * (-td_j/ta)``.
    """
    if branch not in ("gpu", "dpu"):
        raise ValueError('branch must be "gpu" or "dpu"')
    if not np.isfinite(ta) or ta == 0 or not np.isfinite(aur_root_scalar):
        return "—"
    if not np.isfinite(t_branch_abs):
        return "—"
    if branch == "dpu":
        raw = aur_root_scalar * (-t_branch_abs / ta)
    else:
        raw = aur_root_scalar * (t_branch_abs / ta)
    if ratio_display == "abs_root":
        sign = "+" if raw >= 0 else ""
        return f"{sign}{raw:.6f}"
    if not np.isfinite(raw):
        return "—"
    sign = "+" if raw >= 0 else ""
    return f"{sign}{100.0 * raw:.2f}%"


def _hierarchy_name(dims: Sequence[str]) -> str:
    return " > ".join(dims)


def _get_additive_gc_frame(
    df: pd.DataFrame,
    outputs: Dict[str, pd.DataFrame],
    hierarchy_dims: Sequence[str],
    period_col: str,
    base_period: int,
    current_period: int,
) -> pd.DataFrame:
    h = _hierarchy_name(hierarchy_dims)
    if "additive__gc" in outputs:
        sub = outputs["additive__gc"]
        m = sub[sub["hierarchy_name"] == h]
        if not m.empty:
            return m
    frame = additive_hierarchy_decomp(df, "gc", list(hierarchy_dims), period_col, base_period, current_period)
    frame.insert(0, "metric_name", "gc")
    return frame


def _pick_ratio_hierarchy_name_from_outputs(outputs: Dict[str, pd.DataFrame], out_key: str) -> Optional[str]:
    """First available ``hierarchy_name`` in precomputed outputs (shallowest, then alphabetical)."""
    if out_key not in outputs or outputs[out_key].empty:
        return None
    names = outputs[out_key]["hierarchy_name"].dropna().unique().tolist()
    if not names:
        return None
    return sorted(names, key=lambda s: (len(s.split(" > ")), s))[0]


def _get_ratio_frame_on_demand(
    df: pd.DataFrame,
    outputs: Dict[str, pd.DataFrame],
    out_key: str,
    metric_name: str,
    num_col: str,
    den_col: str,
    hierarchy_dims: Sequence[str],
    period_col: str,
    base_period: int,
    current_period: int,
) -> pd.DataFrame:
    """Prefer exact ``hierarchy_dims``; if missing from ``outputs``, use any hierarchy already in ``outputs`` (do not recompute a different split)."""
    h = _hierarchy_name(hierarchy_dims)
    if out_key in outputs:
        sub = outputs[out_key]
        if not sub.empty:
            m = sub[sub["hierarchy_name"] == h]
            if not m.empty:
                return m
            picked = _pick_ratio_hierarchy_name_from_outputs(outputs, out_key)
            if picked is not None:
                return sub[sub["hierarchy_name"] == picked]
    dims = list(hierarchy_dims)
    if out_key in outputs and not outputs[out_key].empty:
        picked = _pick_ratio_hierarchy_name_from_outputs(outputs, out_key)
        if picked:
            dims = picked.split(" > ")
    frame = ratio_hierarchy_decomp(
        df, num_col, den_col, dims, metric_name, period_col, base_period, current_period
    )
    frame.insert(0, "metric_name", metric_name)
    return frame


def build_sales_decomposition_tree(
    df: pd.DataFrame,
    outputs: Dict[str, pd.DataFrame],
    *,
    period_col: str = "year",
    base_period: int = 2024,
    current_period: int = 2025,
    sales_metric: str = "net_sales",
    gc_domains: Sequence[str] = ("daypart", "channel"),
    gc_group_labels: bool = True,
    upt_hierarchy: Sequence[str] = ("channel", "daypart"),
    aur_price_hierarchy: Sequence[str] = ("channel", "daypart"),
    aur_discount_hierarchy: Sequence[str] = ("channel", "daypart"),
    include_interaction: bool = True,
    ratio_display: str = "pct_of_base_ratio",
    ac_show_upt_aur: bool = True,
    ac_show_mix_within: bool = True,
    ac_mix_within_branches: Sequence[Sequence[str]] = (("channel",), ("daypart",)),
    ac_show_mix_within_leaves: bool = True,
) -> TreeNode:
    """Build a Sales → GC / AC / interaction tree with optional GC domains and AC → UPT / AUR → price & discount.

    ``gc_domains`` selects which **separate** GC splits to show, e.g. ``("daypart",)``, ``("channel",)``,
    or ``("daypart", "channel")`` for two grouped branches. If more than one domain is listed,
    use ``gc_group_labels=True`` (default) so each domain is its own branch.

    **Sales bridge** (Sales, GC, AC, Interaction): ``pct_of_base_sales`` from ``sales_gc_ac_bridge``.

    **Line semantics** (for :func:`plot_decomposition_tree` with ``encode_structural_lines=True``): each node
    sets ``TreeNode.sibling_relation`` to ``"additive"`` (children partition / sum to the parent → **solid**
    box and edges) or ``"parallel"`` (alternative views of the same slice, e.g. GC by ``is_offer`` vs by
    ``channel``, or Channel vs Daypart AC panels → **dotted**). Fills stay sentiment-only.

    **AC subtree** — combine as needed:

    - ``ac_show_upt_aur``: **Units per transaction** × **Net sales per unit**, with AUR split into
      original price vs discount.
    - ``ac_show_mix_within`` + ``ac_mix_within_branches``: for each inner tuple (e.g. ``("channel",)``,
      ``("daypart",)``), one **parallel** panel **Dimension → segments** using ``ratio__ac`` (only the
      **first** column in each tuple is used). Example: Channel and Daypart appear as sibling branches
      under AC alongside UPT/AUR when both toggles are on.

    - ``ac_show_mix_within_leaves``: if ``True`` (default), each segment (e.g. lunch) expands to
      **Mix** and **Within** child nodes. If ``False``, the tree stops at segment level (lunch, dinner,
      …) with segment totals only.

    Set ``ac_show_upt_aur=False`` to hide UPT/AUR; set ``ac_show_mix_within=False`` to hide all mix/within
    panels; set ``ac_mix_within_branches=()`` when mix is off or you want no branch panels.

    **UPT** (and level-1 children): raw ratio table values — ``total_pct_of_root_metric`` or ``total_abs_root``.

    **AUR** root: same as ``ratio__aur`` (total change vs base).

    **Original price** and **Discount** (and their level-1 children): **contributions to AUR** that sum to
    the AUR root: ``contrib = A * (t / ta)`` for GPU branches and ``A * (-t / ta)`` for discount branches,
    where ``t`` is ``total_abs_root`` for that row (or segment), ``ta`` is AUR root ``total_abs_root``, and
    ``A`` is the AUR root in the chosen display (pct or abs). So ``A * (tg/ta) + A * (-td/ta) = A``.
    """
    if ratio_display not in ("pct_of_base_ratio", "abs_root"):
        raise ValueError('ratio_display must be "pct_of_base_ratio" or "abs_root"')
    if not ac_show_upt_aur and not ac_show_mix_within:
        raise ValueError("At least one of ac_show_upt_aur or ac_show_mix_within must be True")
    if ac_show_mix_within and not tuple(ac_mix_within_branches):
        raise ValueError("ac_mix_within_branches must list at least one dimension, e.g. (('channel',), ('daypart',)), when ac_show_mix_within is True")
    if "sales_gc_ac_bridge" not in outputs:
        raise KeyError("outputs must include sales_gc_ac_bridge")

    if len(gc_domains) > 1 and not gc_group_labels:
        raise ValueError("gc_group_labels must be True when multiple gc_domains are requested")

    bridge = outputs["sales_gc_ac_bridge"].set_index("component")
    sales_pct = float(bridge.loc["sales_growth_total", "pct_of_base_sales"])
    gc_pct = float(bridge.loc["gc_contribution_to_sales", "pct_of_base_sales"])
    ac_pct = float(bridge.loc["ac_contribution_to_sales", "pct_of_base_sales"])
    int_pct = float(bridge.loc["interaction", "pct_of_base_sales"])

    base_gc = df.loc[df[period_col] == base_period, "gc"].sum()
    cur_gc = df.loc[df[period_col] == current_period, "gc"].sum()
    delta_gc = float(cur_gc - base_gc)

    gc_children: List[TreeNode] = []
    valid_gc = {"is_offer", "daypart", "channel"}
    for dom in gc_domains:
        if dom not in valid_gc:
            raise ValueError(f"gc_domains must be a subset of {valid_gc}, got {dom!r}")
        addf = _get_additive_gc_frame(df, outputs, [dom], period_col, base_period, current_period)
        lvl1 = addf[addf["level"] == 1]
        leaves: List[TreeNode] = []
        dom_sum = 0.0
        for _, row in lvl1.iterrows():
            dv = float(row["delta_value"])
            seg_pct = gc_pct * (dv / delta_gc) if delta_gc else 0.0
            dom_sum += seg_pct
            leaves.append(
                TreeNode(
                    name=str(row["node_value"]),
                    value_text=_fmt_sales_pct(seg_pct),
                    meta_lines=["Share of GC-to-sales via segment dGC"],
                    sibling_relation="additive",
                )
            )
        if not leaves:
            continue
        if gc_group_labels:
            gc_children.append(
                TreeNode(
                    name=dom,
                    value_text=_fmt_sales_pct(dom_sum),
                    meta_lines=[f"GC by {dom}"],
                    children=leaves,
                    sibling_relation="parallel",
                )
            )
        else:
            for lf in leaves:
                lf.sibling_relation = "parallel"
            gc_children.extend(leaves)

    gc_node = TreeNode(
        name="GC",
        value_text=_fmt_sales_pct(gc_pct),
        meta_lines=["GC to sales (bridge)"],
        children=gc_children,
        sibling_relation="additive",
    )

    ac_children: List[TreeNode] = []

    if ac_show_upt_aur:
        upt_f = _get_ratio_frame_on_demand(
            df, outputs, "ratio__upt", "upt", "units", "gc", upt_hierarchy, period_col, base_period, current_period
        )
        aur_f = _get_ratio_frame_on_demand(
            df,
            outputs,
            "ratio__aur",
            "aur",
            sales_metric,
            "units",
            aur_price_hierarchy,
            period_col,
            base_period,
            current_period,
        )
        upt0 = upt_f[upt_f["level"] == 0].iloc[0]

        upt_children: List[TreeNode] = []
        for _, row in upt_f[upt_f["level"] == 1].iterrows():
            upt_children.append(
                TreeNode(
                    name=str(row["node_value"]),
                    value_text=_fmt_ratio_row(row, ratio_display),
                    meta_lines=["UPT: same column as ratio__upt table"],
                    sibling_relation="additive",
                )
            )

        upt_meta = (
            ["units / guest; total_pct_of_root_metric vs base UPT"]
            if ratio_display == "pct_of_base_ratio"
            else ["units / guest; total_abs_root (ratio units)"]
        )
        upt_node = TreeNode(
            name="Units per Transaction",  # UPT
            value_text=_fmt_ratio_row(upt0, ratio_display),
            meta_lines=upt_meta,
            children=upt_children,
            sibling_relation="additive",
        )

        gpu_f = _get_ratio_frame_on_demand(
            df,
            outputs,
            "ratio__original_price_per_unit",
            "original_price_per_unit",
            "original_price_sales",
            "units",
            aur_price_hierarchy,
            period_col,
            base_period,
            current_period,
        )
        dpu_f = _get_ratio_frame_on_demand(
            df,
            outputs,
            "ratio__discount_per_unit",
            "discount_per_unit",
            "discount_amount",
            "units",
            aur_discount_hierarchy,
            period_col,
            base_period,
            current_period,
        )
        g0 = gpu_f[gpu_f["level"] == 0].iloc[0]
        d0 = dpu_f[dpu_f["level"] == 0].iloc[0]
        a0 = aur_f[aur_f["level"] == 0].iloc[0]

        ta = float(a0["total_abs_root"])
        tg = float(g0["total_abs_root"])
        td = float(d0["total_abs_root"])
        a_sc = _aur_root_display_scalar(a0, ratio_display)

        gpu_l1: List[TreeNode] = []
        for _, row in gpu_f[gpu_f["level"] == 1].iterrows():
            tr = float(row["total_abs_root"])
            gpu_l1.append(
                TreeNode(
                    name=str(row["node_value"]),
                    value_text=_fmt_aur_allocated_contribution(a_sc, ta, tr, branch="gpu", ratio_display=ratio_display),
                    meta_lines=["Share of AUR change attributed to list $/u (sums to AUR parent)"],
                    sibling_relation="additive",
                )
            )
        dpu_l1: List[TreeNode] = []
        d1 = dpu_f[dpu_f["level"] == 1]
        for _, row in d1.iterrows():
            tr = float(row["total_abs_root"])
            dpu_l1.append(
                TreeNode(
                    name=str(row["node_value"]),
                    value_text=_fmt_aur_allocated_contribution(a_sc, ta, tr, branch="dpu", ratio_display=ratio_display),
                    meta_lines=["Share of AUR change attributed to discount $/u (sums to AUR parent)"],
                    sibling_relation="additive",
                )
            )

        aur_meta = (
            ["net_sales / unit; total_pct_of_root_metric vs base AUR"]
            if ratio_display == "pct_of_base_ratio"
            else ["net_sales / unit; total_abs_root (ratio units)"]
        )
        aur_node = TreeNode(
            name="Net Sales per Unit",  # AUR
            value_text=_fmt_ratio_row(a0, ratio_display),
            meta_lines=aur_meta,
            children=[
                TreeNode(
                    name="Original price",
                    value_text=_fmt_aur_allocated_contribution(a_sc, ta, tg, branch="gpu", ratio_display=ratio_display),
                    meta_lines=["Contribution to AUR = A * (tg/ta); tg,ta from total_abs_root"],
                    children=gpu_l1,
                    sibling_relation="additive",
                ),
                TreeNode(
                    name="Discount",
                    value_text=_fmt_aur_allocated_contribution(a_sc, ta, td, branch="dpu", ratio_display=ratio_display),
                    meta_lines=["Contribution to AUR = A * (-td/ta)"],
                    children=dpu_l1,
                    sibling_relation="additive",
                ),
            ],
            sibling_relation="additive",
        )

        ac_children.extend([upt_node, aur_node])

    if ac_show_mix_within:
        for branch in ac_mix_within_branches:
            if not branch:
                continue
            ac_children.append(
                _build_ac_mix_within_panel(
                    df,
                    outputs,
                    period_col=period_col,
                    base_period=base_period,
                    current_period=current_period,
                    sales_metric=sales_metric,
                    hierarchy_dims=list(branch[:1]),
                    ratio_display=ratio_display,
                    show_mix_within_leaves=ac_show_mix_within_leaves,
                )
            )

    panel_parts: List[str] = []
    if ac_show_upt_aur:
        panel_parts.append("UPT × AUR")
    if ac_show_mix_within and ac_mix_within_branches:
        panel_parts.append(
            "mix/within: "
            + ", ".join(_dim_display_name(b[0]) for b in ac_mix_within_branches if b)
        )
    elif ac_show_mix_within:
        panel_parts.append("mix/within")
    ac_meta_lines = [
        "AC to sales (pct of base sales, bridge)",
        "Panels: " + "; ".join(panel_parts) + " (ratio metrics not rescaled to bridge %)",
    ]

    ac_node = TreeNode(
        name="AC",
        value_text=_fmt_sales_pct(ac_pct),
        meta_lines=ac_meta_lines,
        children=ac_children,
        sibling_relation="additive",
    )

    sales_children: List[TreeNode] = [ac_node, gc_node]
    if include_interaction:
        sales_children.append(
            TreeNode(
                name="Interaction",
                value_text=_fmt_sales_pct(int_pct),
                meta_lines=["dGC x dAC"],
                sibling_relation="additive",
            )
        )

    return TreeNode(
        name="Sales",
        value_text=_fmt_sales_pct(sales_pct),
        meta_lines=[f"d{sales_metric} / base {sales_metric}"],
        children=sales_children,
    )


def build_example_tree() -> TreeNode:
    # This mirrors your requested visual exactly, but in data-driven node form.
    return TreeNode(
        name="Sales",
        value_text="+15.4%",
        meta_lines=[
            "Formula: Sales growth = GC pp + AC pp (+ interaction)",
        ],
        children=[
            TreeNode(
                name="GC",
                value_text="+8.0 pp",
                meta_lines=[
                    "Contribution = base share × segment growth",
                ],
                children=[
                    TreeNode(
                        name="LTO",
                        value_text="+5.0 pp",
                        meta_lines=[
                            "share = 8.0%",
                            "growth = 62.0%",
                            "calc = 8.0% × 62.0% = 5.0 pp",
                        ],
                        children=[
                            TreeNode(
                                name="Mobile",
                                value_text="+2.6 pp",
                                meta_lines=[
                                    "within LTO branch",
                                    "further split by daypart",
                                ],
                                children=[
                                    TreeNode(
                                        name="Lunch",
                                        value_text="+1.4 pp",
                                        meta_lines=["within Mobile LTO"],
                                    ),
                                    TreeNode(
                                        name="Dinner",
                                        value_text="+1.2 pp",
                                        meta_lines=["within Mobile LTO"],
                                    ),
                                ],
                            ),
                            TreeNode(
                                name="Delivery",
                                value_text="+1.2 pp",
                                meta_lines=["within LTO branch"],
                            ),
                        ],
                    ),
                    TreeNode(
                        name="Value",
                        value_text="+3.0 pp",
                        meta_lines=[
                            "share × growth view",
                        ],
                    ),
                    TreeNode(
                        name="Core",
                        value_text="-4.0 pp",
                        meta_lines=[
                            "negative contribution",
                        ],
                    ),
                ],
            ),
            TreeNode(
                name="AC",
                value_text="+6.8 pp",
                meta_lines=[
                    "Approx bridge: AC growth ≈ UPT growth + AUR growth",
                ],
                children=[
                    TreeNode(
                        name="Units per Transaction", # UPT
                        value_text="+4.0%",
                        meta_lines=[
                            "Units per transaction",
                            "Ratio decomposition",
                        ],
                        children=[
                            TreeNode(
                                name="Bundle mix",
                                value_text="+2.5",
                                meta_lines=[
                                    "mix / penetration effect",
                                ],
                                children=[
                                    TreeNode(
                                        name="Meal deal",
                                        value_text="+1.5",
                                        meta_lines=["bundle subtype"],
                                    ),
                                    TreeNode(
                                        name="Family bundle",
                                        value_text="+1.0",
                                        meta_lines=["bundle subtype"],
                                    ),
                                ],
                            ),
                            TreeNode(
                                name="Promo bundle attach",
                                value_text="+1.5",
                                meta_lines=[
                                    "within / attachment effect",
                                ],
                            ),
                        ],
                    ),
                    TreeNode(
                        name="AUR",
                        value_text="+2.7%",
                        meta_lines=[
                            "AUR = Gross price / unit - Discount / unit",
                        ],
                        children=[
                            TreeNode(
                                name="Gross",
                                value_text="+4.2%",
                                meta_lines=[
                                    "ratio branch",
                                    "can show mix + within",
                                ],
                                children=[
                                    TreeNode(
                                        name="LTO",
                                        value_text="+3.2",
                                        meta_lines=[
                                            "mix + within",
                                        ],
                                    ),
                                    TreeNode(
                                        name="Premium",
                                        value_text="+1.3",
                                        meta_lines=[
                                            "mix + within",
                                        ],
                                    ),
                                    TreeNode(
                                        name="Core",
                                        value_text="-2.0",
                                        meta_lines=[
                                            "mix + within",
                                        ],
                                    ),
                                ],
                            ),
                            TreeNode(
                                name="Discount",
                                value_text="-1.7%",
                                meta_lines=[
                                    "negative drag on AUR",
                                ],
                                children=[
                                    TreeNode(
                                        name="App offers",
                                        value_text="-0.8",
                                        meta_lines=["promo type"],
                                    ),
                                    TreeNode(
                                        name="BOGO",
                                        value_text="-0.4",
                                        meta_lines=["promo type"],
                                    ),
                                    TreeNode(
                                        name="Meal deal",
                                        value_text="-0.3",
                                        meta_lines=["promo type"],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "accounting_decomp_tree_example.png"
    tree = build_example_tree()
    plot_decomposition_tree(
        tree,
        out,
        title="Sales / GC / AC Accounting Decomposition Tree",
        subtitle="Example hierarchy (clean style, PPT-friendly)",
        font_size=18,
        show_meta_lines=True,
        presentation=True,
        show=False,
    )
    print(out)
