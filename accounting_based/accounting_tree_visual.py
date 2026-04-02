
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accounting_decomp import additive_hierarchy_decomp, ratio_hierarchy_decomp


@dataclass
class TreeNode:
    name: str
    value_text: str
    meta_lines: List[str] = field(default_factory=list)
    children: List["TreeNode"] = field(default_factory=list)
    color: Optional[str] = None


def make_node_text(node: TreeNode, *, include_meta_lines: bool = True) -> str:
    lines = [f"{node.name} {node.value_text}"]
    if include_meta_lines:
        lines.extend(node.meta_lines)
    return "\n".join(lines)


def _count_leaves(node: TreeNode) -> int:
    if not node.children:
        return 1
    return sum(_count_leaves(child) for child in node.children)


def _assign_positions(
    node: TreeNode,
    depth: int,
    y_start: float,
    y_end: float,
    positions: dict,
) -> None:
    x = depth
    y = (y_start + y_end) / 2
    positions[id(node)] = (x, y)

    if not node.children:
        return

    total_leaves = sum(_count_leaves(child) for child in node.children)
    cursor = y_start
    for child in node.children:
        leaves = _count_leaves(child)
        child_span = (y_end - y_start) * leaves / total_leaves
        _assign_positions(child, depth + 1, cursor, cursor + child_span, positions)
        cursor += child_span


def plot_decomposition_tree(
    root: TreeNode,
    save_path: Optional[str | Path] = None,
    *,
    title: str = "Identity-Based Decomposition Tree",
    figsize: Tuple[int, int] = (18, 10),
    box_width: float = 0.82,
    font_size: int = 10,
    show: bool = True,
    show_meta_lines: bool = True,
) -> Optional[Path]:
    """Render the tree. When show_meta_lines is False, each box shows only \"Name +value\" (no meta_lines)."""
    positions = {}
    _assign_positions(root, depth=0, y_start=0, y_end=1, positions=positions)

    # normalize x coordinates to [0, 1]
    max_depth = max(x for x, _ in positions.values())
    positions = {
        k: ((x / max_depth) if max_depth > 0 else 0.0, y)
        for k, (x, y) in positions.items()
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=18)

    def draw_edges(node: TreeNode):
        x0, y0 = positions[id(node)]
        for child in node.children:
            x1, y1 = positions[id(child)]
            ax.plot(
                [x0 + 0.03, x1 - 0.03],
                [y0, y1],
                linewidth=1.4,
                color="gray",
            )
            draw_edges(child)

    def draw_nodes(node: TreeNode):
        x, y = positions[id(node)]
        txt = make_node_text(node, include_meta_lines=show_meta_lines)
        bbox_props = dict(
            boxstyle="round,pad=0.4",
            facecolor=node.color if node.color else "white",
            edgecolor="gray",
            linewidth=1.0,
        )
        ax.text(
            x,
            y,
            txt,
            ha="center",
            va="center",
            fontsize=font_size,
            family="DejaVu Sans Mono",
            bbox=bbox_props,
        )
        for child in node.children:
            draw_nodes(child)

    draw_edges(root)
    draw_nodes(root)

    fig.tight_layout()
    out: Optional[Path] = None
    if save_path is not None:
        out = Path(save_path)
        fig.savefig(out, dpi=160, bbox_inches="tight")
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
) -> TreeNode:
    """Build a Sales → GC / AC / interaction tree with optional GC domains and AC → UPT / AUR → price & discount.

    ``gc_domains`` selects which **separate** GC splits to show, e.g. ``("daypart",)``, ``("channel",)``,
    or ``("daypart", "channel")`` for two grouped branches. If more than one domain is listed,
    use ``gc_group_labels=True`` (default) so each domain is its own branch.

    **Sales bridge** (Sales, GC, AC, Interaction): ``pct_of_base_sales`` from ``sales_gc_ac_bridge``.

    **UPT** (and level-1 children): raw ratio table values — ``total_pct_of_root_metric`` or ``total_abs_root``.

    **AUR** root: same as ``ratio__aur`` (total change vs base).

    **Original price** and **Discount** (and their level-1 children): **contributions to AUR** that sum to
    the AUR root: ``contrib = A * (t / ta)`` for GPU branches and ``A * (-t / ta)`` for discount branches,
    where ``t`` is ``total_abs_root`` for that row (or segment), ``ta`` is AUR root ``total_abs_root``, and
    ``A`` is the AUR root in the chosen display (pct or abs). So ``A * (tg/ta) + A * (-td/ta) = A``.
    """
    if ratio_display not in ("pct_of_base_ratio", "abs_root"):
        raise ValueError('ratio_display must be "pct_of_base_ratio" or "abs_root"')
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
                    color="#D8ECFF",
                )
            )
        else:
            gc_children.extend(leaves)

    gc_node = TreeNode(
        name="GC",
        value_text=_fmt_sales_pct(gc_pct),
        meta_lines=["GC to sales (bridge)"],
        color="#E8F1FB",
        children=gc_children,
    )

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
                color="#FFF8D9",
            )
        )

    upt_meta = (
        ["units / guest; total_pct_of_root_metric vs base UPT"]
        if ratio_display == "pct_of_base_ratio"
        else ["units / guest; total_abs_root (ratio units)"]
    )
    upt_node = TreeNode(
        name="UPT",
        value_text=_fmt_ratio_row(upt0, ratio_display),
        meta_lines=upt_meta,
        color="#FFF0B8",
        children=upt_children,
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
                color="#FFF8D9",
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
                color="#FFE0E0",
            )
        )

    aur_meta = (
        ["net_sales / unit; total_pct_of_root_metric vs base AUR"]
        if ratio_display == "pct_of_base_ratio"
        else ["net_sales / unit; total_abs_root (ratio units)"]
    )
    aur_node = TreeNode(
        name="Net Sales per Unit", # AUR
        value_text=_fmt_ratio_row(a0, ratio_display),
        meta_lines=aur_meta,
        color="#FFE7A0",
        children=[
            TreeNode(
                name="Original price",
                value_text=_fmt_aur_allocated_contribution(a_sc, ta, tg, branch="gpu", ratio_display=ratio_display),
                meta_lines=["Contribution to AUR = A * (tg/ta); tg,ta from total_abs_root"],
                color="#FFF0B8",
                children=gpu_l1,
            ),
            TreeNode(
                name="Discount",
                value_text=_fmt_aur_allocated_contribution(a_sc, ta, td, branch="dpu", ratio_display=ratio_display),
                meta_lines=["Contribution to AUR = A * (-td/ta)"],
                color="#FFD6D6",
                children=dpu_l1,
            ),
        ],
    )

    ac_node = TreeNode(
        name="AC",
        value_text=_fmt_sales_pct(ac_pct),
        meta_lines=[
            "AC to sales (pct of base sales, bridge)",
            "UPT/AUR below use ratio table metrics (not rescaled to this %)",
        ],
        color="#FFF4D6",
        children=[upt_node, aur_node],
    )

    sales_children: List[TreeNode] = [gc_node, ac_node]
    if include_interaction:
        sales_children.append(
            TreeNode(
                name="Interaction",
                value_text=_fmt_sales_pct(int_pct),
                meta_lines=["dGC x dAC"],
                color="#EEEEEE",
            )
        )

    return TreeNode(
        name="Sales",
        value_text=_fmt_sales_pct(sales_pct),
        meta_lines=[f"d{sales_metric} / base {sales_metric}"],
        color="#EFEFEF",
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
        color="#EFEFEF",
        children=[
            TreeNode(
                name="GC",
                value_text="+8.0 pp",
                meta_lines=[
                    "Contribution = base share × segment growth",
                ],
                color="#E8F1FB",
                children=[
                    TreeNode(
                        name="LTO",
                        value_text="+5.0 pp",
                        meta_lines=[
                            "share = 8.0%",
                            "growth = 62.0%",
                            "calc = 8.0% × 62.0% = 5.0 pp",
                        ],
                        color="#D8ECFF",
                        children=[
                            TreeNode(
                                name="Mobile",
                                value_text="+2.6 pp",
                                meta_lines=[
                                    "within LTO branch",
                                    "further split by daypart",
                                ],
                                color="#DFF7E5",
                                children=[
                                    TreeNode(
                                        name="Lunch",
                                        value_text="+1.4 pp",
                                        meta_lines=["within Mobile LTO"],
                                        color="#F9F9F9",
                                    ),
                                    TreeNode(
                                        name="Dinner",
                                        value_text="+1.2 pp",
                                        meta_lines=["within Mobile LTO"],
                                        color="#F9F9F9",
                                    ),
                                ],
                            ),
                            TreeNode(
                                name="Delivery",
                                value_text="+1.2 pp",
                                meta_lines=["within LTO branch"],
                                color="#DFF7E5",
                            ),
                        ],
                    ),
                    TreeNode(
                        name="Value",
                        value_text="+3.0 pp",
                        meta_lines=[
                            "share × growth view",
                        ],
                        color="#D8ECFF",
                    ),
                    TreeNode(
                        name="Core",
                        value_text="-4.0 pp",
                        meta_lines=[
                            "negative contribution",
                        ],
                        color="#FBE4E6",
                    ),
                ],
            ),
            TreeNode(
                name="AC",
                value_text="+6.8 pp",
                meta_lines=[
                    "Approx bridge: AC growth ≈ UPT growth + AUR growth",
                ],
                color="#FFF4D6",
                children=[
                    TreeNode(
                        name="UPT",
                        value_text="+4.0%",
                        meta_lines=[
                            "Units per transaction",
                            "Ratio decomposition",
                        ],
                        color="#FFF0B8",
                        children=[
                            TreeNode(
                                name="Bundle mix",
                                value_text="+2.5",
                                meta_lines=[
                                    "mix / penetration effect",
                                ],
                                color="#FFF8D9",
                                children=[
                                    TreeNode(
                                        name="Meal deal",
                                        value_text="+1.5",
                                        meta_lines=["bundle subtype"],
                                        color="#F9F9F9",
                                    ),
                                    TreeNode(
                                        name="Family bundle",
                                        value_text="+1.0",
                                        meta_lines=["bundle subtype"],
                                        color="#F9F9F9",
                                    ),
                                ],
                            ),
                            TreeNode(
                                name="Promo bundle attach",
                                value_text="+1.5",
                                meta_lines=[
                                    "within / attachment effect",
                                ],
                                color="#FFF8D9",
                            ),
                        ],
                    ),
                    TreeNode(
                        name="AUR",
                        value_text="+2.7%",
                        meta_lines=[
                            "AUR = Gross price / unit - Discount / unit",
                        ],
                        color="#FFE7A0",
                        children=[
                            TreeNode(
                                name="Gross",
                                value_text="+4.2%",
                                meta_lines=[
                                    "ratio branch",
                                    "can show mix + within",
                                ],
                                color="#FFF0B8",
                                children=[
                                    TreeNode(
                                        name="LTO",
                                        value_text="+3.2",
                                        meta_lines=[
                                            "mix + within",
                                        ],
                                        color="#F9F9F9",
                                    ),
                                    TreeNode(
                                        name="Premium",
                                        value_text="+1.3",
                                        meta_lines=[
                                            "mix + within",
                                        ],
                                        color="#F9F9F9",
                                    ),
                                    TreeNode(
                                        name="Core",
                                        value_text="-2.0",
                                        meta_lines=[
                                            "mix + within",
                                        ],
                                        color="#F9F9F9",
                                    ),
                                ],
                            ),
                            TreeNode(
                                name="Discount",
                                value_text="-1.7%",
                                meta_lines=[
                                    "negative drag on AUR",
                                ],
                                color="#FFD6D6",
                                children=[
                                    TreeNode(
                                        name="App offers",
                                        value_text="-0.8",
                                        meta_lines=["promo type"],
                                        color="#F9F9F9",
                                    ),
                                    TreeNode(
                                        name="BOGO",
                                        value_text="-0.4",
                                        meta_lines=["promo type"],
                                        color="#F9F9F9",
                                    ),
                                    TreeNode(
                                        name="Meal deal",
                                        value_text="-0.3",
                                        meta_lines=["promo type"],
                                        color="#F9F9F9",
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
    out = Path("/mnt/data/accounting_decomp_tree_example.png")
    tree = build_example_tree()
    plot_decomposition_tree(
        tree,
        out,
        title="Sales / GC / AC Accounting Decomposition Tree",
        figsize=(20, 11),
        font_size=9,
    )
    print(out)
