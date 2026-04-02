
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def generate_synthetic_mcd_data(start: str = "2024-01-01", end: str = "2025-12-31", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start=start, end=end, freq="W-MON")
    channels = ["drive_thru", "digital", "delivery", "instore"]
    dayparts = ["breakfast", "lunch", "dinner", "late_night"]
    menu_types = ["core", "value", "premium"]
    promo_types = ["none", "offer", "reward", "bogo"]

    rows = []
    for week in weeks:
        year = week.year
        week_num = int(week.isocalendar().week)
        month = week.month
        quarter = (month - 1) // 3 + 1
        yoy_factor = 1.06 if year == 2025 else 1.00
        seasonality = 1 + 0.06 * np.sin(2 * np.pi * week_num / 52.0) + 0.03 * np.cos(2 * np.pi * week_num / 26.0)
        summer_delivery_lift = 1.05 if month in [6, 7, 8] else 1.0
        breakfast_softness_2025 = 0.98 if year == 2025 else 1.0
        digital_trend = 1.18 if year == 2025 else 1.0
        lto_push = 1.20 if (year == 2025 and quarter in [2, 3]) else 1.0

        channel_mult = {
            "drive_thru": 1.60,
            "digital": 0.95 * digital_trend,
            "delivery": 0.70 * summer_delivery_lift,
            "instore": 0.90 if year == 2024 else 0.82,
        }
        daypart_mult = {"breakfast": 1.00 * breakfast_softness_2025, "lunch": 1.18, "dinner": 1.28, "late_night": 0.42}
        menu_mult = {"core": 1.00, "value": 0.75, "premium": 0.42}
        base_gpu = {"core": 6.20, "value": 4.90, "premium": 8.90}
        price_increase_2025 = {"core": 0.18, "value": 0.10, "premium": 0.30}

        for channel in channels:
            for daypart in dayparts:
                for menu_type in menu_types:
                    for promo_type in promo_types:
                        promo_mix = {"none": 0.62, "offer": 0.18, "reward": 0.12, "bogo": 0.08}[promo_type]
                        promo_struct_mult = 1.0
                        if promo_type == "offer" and channel == "digital":
                            promo_struct_mult *= 1.35
                        if promo_type == "reward" and channel == "digital":
                            promo_struct_mult *= 1.30
                        if promo_type == "bogo" and daypart == "lunch":
                            promo_struct_mult *= 1.20
                        if promo_type == "none" and channel == "delivery":
                            promo_struct_mult *= 0.92

                        menu_struct_mult = 1.0
                        if menu_type == "premium" and daypart == "dinner":
                            menu_struct_mult *= 1.18
                        if menu_type == "value" and daypart == "breakfast":
                            menu_struct_mult *= 0.90
                        if menu_type == "premium" and channel == "delivery":
                            menu_struct_mult *= 1.15
                        if menu_type == "value" and channel == "drive_thru":
                            menu_struct_mult *= 1.08
                        if menu_type == "premium":
                            menu_struct_mult *= lto_push

                        base_gc = (
                            42
                            * yoy_factor
                            * seasonality
                            * channel_mult[channel]
                            * daypart_mult[daypart]
                            * menu_mult[menu_type]
                            * promo_mix
                            * promo_struct_mult
                            * menu_struct_mult
                        )
                        gc = max(0, int(np.round(base_gc + rng.normal(0, 4))))

                        offer_rate = {"none": 0.00, "offer": 0.95, "reward": 0.82, "bogo": 0.98}[promo_type]
                        reward_rate = {"none": 0.00, "offer": 0.00, "reward": 0.82, "bogo": 0.00}[promo_type]
                        offer_gc = int(np.round(gc * offer_rate))
                        reward_gc = int(np.round(gc * reward_rate))

                        upt_base = {"core": 1.30, "value": 1.22, "premium": 1.46}[menu_type]
                        daypart_upt = {"breakfast": 0.96, "lunch": 1.00, "dinner": 1.08, "late_night": 1.03}[daypart]
                        promo_upt = {"none": 1.00, "offer": 1.06, "reward": 1.03, "bogo": 1.14}[promo_type]
                        channel_upt = {"drive_thru": 1.01, "digital": 1.03, "delivery": 1.10, "instore": 0.98}[channel]
                        upt_growth_2025 = 1.02 if year == 2025 else 1.00
                        upt = upt_base * daypart_upt * promo_upt * channel_upt * upt_growth_2025
                        units = max(gc, int(np.round(gc * upt + rng.normal(0, max(1, gc * 0.03)))))

                        gpu = base_gpu[menu_type]
                        if year == 2025:
                            gpu += price_increase_2025[menu_type]
                        gpu *= {"breakfast": 0.92, "lunch": 1.00, "dinner": 1.08, "late_night": 1.04}[daypart]
                        gpu *= {"drive_thru": 1.00, "digital": 1.01, "delivery": 1.10, "instore": 0.99}[channel]
                        gpu *= {"none": 1.00, "offer": 1.00, "reward": 1.00, "bogo": 0.98}[promo_type]

                        original_price_sales = units * gpu
                        discount_per_offer_txn = {"none": 0.00, "offer": 1.15, "reward": 0.72, "bogo": 1.55}[promo_type]
                        discount_scale = {"drive_thru": 1.00, "digital": 1.10, "delivery": 1.05, "instore": 0.96}[channel]
                        discount_amount = offer_gc * discount_per_offer_txn * discount_scale

                        original_price_sales = max(0.0, original_price_sales + rng.normal(0, original_price_sales * 0.015))
                        discount_amount = max(0.0, discount_amount + rng.normal(0, max(0.2, discount_amount * 0.04)))
                        original_price_sales = round(original_price_sales, 2)
                        discount_amount = round(discount_amount, 2)
                        if discount_amount > original_price_sales:
                            discount_amount = original_price_sales
                        # Enforce net = original - discount so aggregate AUR = GPU - DPU in ratio decomps
                        net_sales = round(original_price_sales - discount_amount, 2)

                        rows.append(
                            {
                                "week_start": week,
                                "year": year,
                                "week_num": week_num,
                                "quarter": quarter,
                                "month": month,
                                "channel": channel,
                                "daypart": daypart,
                                "menu_type": menu_type,
                                "promo_type": promo_type,
                                "gc": gc,
                                "offer_gc": offer_gc,
                                "reward_gc": reward_gc,
                                "units": units,
                                "original_price_sales": original_price_sales,
                                "discount_amount": discount_amount,
                                "net_sales": net_sales,
                            }
                        )

    df = pd.DataFrame(rows)
    df["is_offer"] = (df["promo_type"] != "none").astype(int)
    df["is_reward"] = (df["promo_type"] == "reward").astype(int)
    df["is_bogo"] = (df["promo_type"] == "bogo").astype(int)
    df["ac"] = np.where(df["gc"] > 0, df["net_sales"] / df["gc"], np.nan)
    df["upt"] = np.where(df["gc"] > 0, df["units"] / df["gc"], np.nan)
    df["aur"] = np.where(df["units"] > 0, df["net_sales"] / df["units"], np.nan)
    df["original_price_per_unit"] = np.where(df["units"] > 0, df["original_price_sales"] / df["units"], np.nan)
    df["discount_per_unit"] = np.where(df["units"] > 0, df["discount_amount"] / df["units"], np.nan)
    return df


def build_hierarchy_orders(dimensions: Sequence[str], max_depth: Optional[int] = None) -> List[List[str]]:
    dims = list(dict.fromkeys(dimensions))
    max_depth = max_depth or len(dims)
    all_orders = []
    for depth in range(1, max_depth + 1):
        for perm in itertools.permutations(dims, depth):
            all_orders.append(list(perm))
    return all_orders


def _safe_growth(base_value: float, current_value: float) -> float:
    if pd.isna(base_value) or base_value == 0:
        return np.nan
    return (current_value - base_value) / base_value


def _path_to_key(path_parts: Sequence[str]) -> str:
    return "ROOT" if not path_parts else " > ".join(path_parts)


def _aggregate_period(df: pd.DataFrame, period_col: str, metric_cols: Sequence[str], dims: Sequence[str], base_period, current_period) -> pd.DataFrame:
    g = df.groupby([period_col] + list(dims), dropna=False)[list(metric_cols)].sum().reset_index()
    out = g.pivot_table(index=list(dims), columns=period_col, values=list(metric_cols), aggfunc="sum", fill_value=0.0)
    out.columns = [f"{m}_{p}" for m, p in out.columns]
    out = out.reset_index()
    for m in metric_cols:
        for p in [base_period, current_period]:
            col = f"{m}_{p}"
            if col not in out.columns:
                out[col] = 0.0
    return out


def additive_hierarchy_decomp(df: pd.DataFrame, metric_col: str, hierarchy: Sequence[str], period_col: str = "year", base_period=2024, current_period=2025) -> pd.DataFrame:
    hierarchy = list(hierarchy)
    metric_period = _aggregate_period(df, period_col, [metric_col], hierarchy, base_period, current_period)
    base_total = df.loc[df[period_col] == base_period, metric_col].sum()
    current_total = df.loc[df[period_col] == current_period, metric_col].sum()

    rows = [{
        "analysis_type": "additive",
        "metric": metric_col,
        "hierarchy_name": " > ".join(hierarchy),
        "level": 0,
        "node_dimension": "ROOT",
        "node_value": "ALL",
        "node_key": "ROOT",
        "parent_key": None,
        "path": "ROOT",
        "base_value": base_total,
        "current_value": current_total,
        "delta_value": current_total - base_total,
        "base_share_of_total": 1.0,
        "current_share_of_total": 1.0,
        "base_share_within_parent": 1.0,
        "current_share_within_parent": 1.0,
        "growth_pct": _safe_growth(base_total, current_total),
        "contribution_pct_of_total": _safe_growth(base_total, current_total),
        "calc_formula": "contribution_pct_of_total = (current_value - base_value) / base_total = base_share_of_total * growth_pct",
    }]

    for level in range(1, len(hierarchy) + 1):
        dims = hierarchy[:level]
        grouped = metric_period.groupby(dims, dropna=False)[[f"{metric_col}_{base_period}", f"{metric_col}_{current_period}"]].sum().reset_index()
        parent_dims = dims[:-1]

        if parent_dims:
            parent_totals = (
                grouped.groupby(parent_dims, dropna=False)[[f"{metric_col}_{base_period}", f"{metric_col}_{current_period}"]]
                .sum()
                .rename(columns={f"{metric_col}_{base_period}": "parent_base_value", f"{metric_col}_{current_period}": "parent_current_value"})
                .reset_index()
            )
            grouped = grouped.merge(parent_totals, on=parent_dims, how="left")
        else:
            grouped["parent_base_value"] = base_total
            grouped["parent_current_value"] = current_total

        for _, row in grouped.iterrows():
            path_parts = [f"{d}={row[d]}" for d in dims]
            parent_parts = [f"{d}={row[d]}" for d in parent_dims]
            node_base = row[f"{metric_col}_{base_period}"]
            node_current = row[f"{metric_col}_{current_period}"]
            parent_base = row["parent_base_value"]
            parent_current = row["parent_current_value"]

            rows.append({
                "analysis_type": "additive",
                "metric": metric_col,
                "hierarchy_name": " > ".join(hierarchy),
                "level": level,
                "node_dimension": dims[-1],
                "node_value": row[dims[-1]],
                "node_key": _path_to_key(path_parts),
                "parent_key": _path_to_key(parent_parts),
                "path": " | ".join(path_parts),
                "base_value": node_base,
                "current_value": node_current,
                "delta_value": node_current - node_base,
                "base_share_of_total": node_base / base_total if base_total else np.nan,
                "current_share_of_total": node_current / current_total if current_total else np.nan,
                "base_share_within_parent": node_base / parent_base if parent_base else np.nan,
                "current_share_within_parent": node_current / parent_current if parent_current else np.nan,
                "growth_pct": _safe_growth(node_base, node_current),
                "contribution_pct_of_total": (node_current - node_base) / base_total if base_total else np.nan,
                "calc_formula": "contribution_pct_of_total = (current_value - base_value) / root_base_total",
            })

    return pd.DataFrame(rows)


def ratio_hierarchy_decomp(df: pd.DataFrame, numerator_col: str, denominator_col: str, hierarchy: Sequence[str], metric_name: Optional[str] = None, period_col: str = "year", base_period=2024, current_period=2025) -> pd.DataFrame:
    hierarchy = list(hierarchy)
    metric_name = metric_name or f"{numerator_col}_per_{denominator_col}"
    agg = _aggregate_period(df, period_col, [numerator_col, denominator_col], hierarchy, base_period, current_period)

    root_num_base = df.loc[df[period_col] == base_period, numerator_col].sum()
    root_num_current = df.loc[df[period_col] == current_period, numerator_col].sum()
    root_den_base = df.loc[df[period_col] == base_period, denominator_col].sum()
    root_den_current = df.loc[df[period_col] == current_period, denominator_col].sum()
    root_ratio_base = root_num_base / root_den_base if root_den_base else np.nan
    root_ratio_current = root_num_current / root_den_current if root_den_current else np.nan

    rows = [{
        "analysis_type": "ratio",
        "metric": metric_name,
        "numerator_col": numerator_col,
        "denominator_col": denominator_col,
        "hierarchy_name": " > ".join(hierarchy),
        "level": 0,
        "node_dimension": "ROOT",
        "node_value": "ALL",
        "node_key": "ROOT",
        "parent_key": None,
        "path": "ROOT",
        "base_numerator": root_num_base,
        "current_numerator": root_num_current,
        "base_denominator": root_den_base,
        "current_denominator": root_den_current,
        "ratio_base": root_ratio_base,
        "ratio_current": root_ratio_current,
        "ratio_delta": root_ratio_current - root_ratio_base,
        "base_weight_of_root": 1.0,
        "current_weight_of_root": 1.0,
        "base_share_within_parent_den": 1.0,
        "current_share_within_parent_den": 1.0,
        "mix_abs_root": 0.0,
        "within_abs_root": root_ratio_current - root_ratio_base,
        "total_abs_root": root_ratio_current - root_ratio_base,
        "mix_pct_of_root_metric": 0.0,
        "within_pct_of_root_metric": (root_ratio_current - root_ratio_base) / root_ratio_base if root_ratio_base else np.nan,
        "total_pct_of_root_metric": (root_ratio_current - root_ratio_base) / root_ratio_base if root_ratio_base else np.nan,
        "calc_formula": "total_abs_root = (w1-w0)*ratio_base + w1*(ratio_current-ratio_base)",
    }]

    for level in range(1, len(hierarchy) + 1):
        dims = hierarchy[:level]
        grouped = agg.groupby(dims, dropna=False)[[f"{numerator_col}_{base_period}", f"{numerator_col}_{current_period}", f"{denominator_col}_{base_period}", f"{denominator_col}_{current_period}"]].sum().reset_index()
        parent_dims = dims[:-1]

        if parent_dims:
            parent_totals = (
                grouped.groupby(parent_dims, dropna=False)[[f"{denominator_col}_{base_period}", f"{denominator_col}_{current_period}"]]
                .sum()
                .rename(columns={f"{denominator_col}_{base_period}": "parent_base_denominator", f"{denominator_col}_{current_period}": "parent_current_denominator"})
                .reset_index()
            )
            grouped = grouped.merge(parent_totals, on=parent_dims, how="left")
        else:
            grouped["parent_base_denominator"] = root_den_base
            grouped["parent_current_denominator"] = root_den_current

        for _, row in grouped.iterrows():
            num_base = row[f"{numerator_col}_{base_period}"]
            num_current = row[f"{numerator_col}_{current_period}"]
            den_base = row[f"{denominator_col}_{base_period}"]
            den_current = row[f"{denominator_col}_{current_period}"]
            ratio_base = num_base / den_base if den_base else np.nan
            ratio_current = num_current / den_current if den_current else np.nan
            w0_root = den_base / root_den_base if root_den_base else np.nan
            w1_root = den_current / root_den_current if root_den_current else np.nan
            mix_abs_root = (w1_root - w0_root) * ratio_base if pd.notna(ratio_base) else np.nan
            within_abs_root = w1_root * (ratio_current - ratio_base) if pd.notna(ratio_base) and pd.notna(ratio_current) else np.nan
            total_abs_root = mix_abs_root + within_abs_root if pd.notna(mix_abs_root) and pd.notna(within_abs_root) else np.nan

            path_parts = [f"{d}={row[d]}" for d in dims]
            parent_parts = [f"{d}={row[d]}" for d in parent_dims]
            parent_base_den = row["parent_base_denominator"]
            parent_current_den = row["parent_current_denominator"]

            rows.append({
                "analysis_type": "ratio",
                "metric": metric_name,
                "numerator_col": numerator_col,
                "denominator_col": denominator_col,
                "hierarchy_name": " > ".join(hierarchy),
                "level": level,
                "node_dimension": dims[-1],
                "node_value": row[dims[-1]],
                "node_key": _path_to_key(path_parts),
                "parent_key": _path_to_key(parent_parts),
                "path": " | ".join(path_parts),
                "base_numerator": num_base,
                "current_numerator": num_current,
                "base_denominator": den_base,
                "current_denominator": den_current,
                "ratio_base": ratio_base,
                "ratio_current": ratio_current,
                "ratio_delta": ratio_current - ratio_base if pd.notna(ratio_base) and pd.notna(ratio_current) else np.nan,
                "base_weight_of_root": w0_root,
                "current_weight_of_root": w1_root,
                "base_share_within_parent_den": den_base / parent_base_den if parent_base_den else np.nan,
                "current_share_within_parent_den": den_current / parent_current_den if parent_current_den else np.nan,
                "mix_abs_root": mix_abs_root,
                "within_abs_root": within_abs_root,
                "total_abs_root": total_abs_root,
                "mix_pct_of_root_metric": mix_abs_root / root_ratio_base if root_ratio_base else np.nan,
                "within_pct_of_root_metric": within_abs_root / root_ratio_base if root_ratio_base else np.nan,
                "total_pct_of_root_metric": total_abs_root / root_ratio_base if root_ratio_base else np.nan,
                "calc_formula": "total_abs_root = (w1-w0)*ratio_base + w1*(ratio_current-ratio_base)",
            })

    return pd.DataFrame(rows)


def merge_aur_gpu_discount_ratio_decomps(
    aur_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    dpu_df: pd.DataFrame,
    *,
    keys: Sequence[str] = ("hierarchy_name", "level", "node_key", "path"),
) -> pd.DataFrame:
    """Join AUR, original price / unit, and discount / unit ratio decompositions on ``keys``.

    When data satisfy ``net_sales = original_price_sales - discount_amount`` (row-level),
    ``total_abs_root`` for AUR equals ``gpu.total_abs_root - dpu.total_abs_root`` per row.

    If tiny residuals remain (rounding / external feeds), ``total_abs_root_*_adj`` split the residual
    so ``gpu_adj - dpu_adj == aur.total_abs_root`` exactly.
    """
    for name, df in [("aur", aur_df), ("gpu", gpu_df), ("dpu", dpu_df)]:
        miss = [k for k in keys if k not in df.columns]
        if miss:
            raise ValueError(f"{name} df missing columns: {miss}")

    a_cols = {
        "total_abs_root": "total_abs_root_aur",
        "mix_abs_root": "mix_abs_root_aur",
        "within_abs_root": "within_abs_root_aur",
        "ratio_base": "ratio_base_aur",
        "ratio_current": "ratio_current_aur",
    }
    g_cols = {
        "total_abs_root": "total_abs_root_gpu",
        "mix_abs_root": "mix_abs_root_gpu",
        "within_abs_root": "within_abs_root_gpu",
        "ratio_base": "ratio_base_gpu",
        "ratio_current": "ratio_current_gpu",
    }
    d_cols = {
        "total_abs_root": "total_abs_root_dpu",
        "mix_abs_root": "mix_abs_root_dpu",
        "within_abs_root": "within_abs_root_dpu",
        "ratio_base": "ratio_base_dpu",
        "ratio_current": "ratio_current_dpu",
    }
    keep_metric = list(a_cols.keys())
    a = aur_df[list(keys) + [c for c in keep_metric if c in aur_df.columns]].rename(columns=a_cols)
    g = gpu_df[list(keys) + [c for c in keep_metric if c in gpu_df.columns]].rename(columns=g_cols)
    d = dpu_df[list(keys) + [c for c in keep_metric if c in dpu_df.columns]].rename(columns=d_cols)
    merged = a.merge(g, on=keys, how="inner").merge(d, on=keys, how="inner")
    merged["implied_aur_from_gpu_dpu"] = merged["total_abs_root_gpu"] - merged["total_abs_root_dpu"]
    merged["residual_aur"] = merged["total_abs_root_aur"] - merged["implied_aur_from_gpu_dpu"]
    abs_g = merged["total_abs_root_gpu"].abs()
    abs_d = merged["total_abs_root_dpu"].abs()
    denom = abs_g + abs_d
    tol = 1e-8
    w_g = np.where(denom > tol, abs_g / denom, 0.5)
    w_d = 1.0 - w_g
    res = merged["residual_aur"]
    merged["total_abs_root_gpu_adj"] = merged["total_abs_root_gpu"] + res * w_g
    merged["total_abs_root_dpu_adj"] = merged["total_abs_root_dpu"] - res * w_d
    return merged


def sales_gc_ac_bridge(df: pd.DataFrame, sales_col: str = "net_sales", gc_col: str = "gc", period_col: str = "year", base_period=2024, current_period=2025, split_interaction_half: bool = False) -> pd.DataFrame:
    base_sales = df.loc[df[period_col] == base_period, sales_col].sum()
    current_sales = df.loc[df[period_col] == current_period, sales_col].sum()
    base_gc = df.loc[df[period_col] == base_period, gc_col].sum()
    current_gc = df.loc[df[period_col] == current_period, gc_col].sum()
    base_ac = base_sales / base_gc if base_gc else np.nan
    current_ac = current_sales / current_gc if current_gc else np.nan

    delta_gc = current_gc - base_gc
    delta_ac = current_ac - base_ac
    delta_sales = current_sales - base_sales

    gc_abs = base_ac * delta_gc
    ac_abs = base_gc * delta_ac
    interaction_abs = delta_gc * delta_ac

    if split_interaction_half:
        gc_abs += interaction_abs / 2
        ac_abs += interaction_abs / 2
        interaction_abs = 0.0

    return pd.DataFrame([
        {"component": "sales_growth_total", "base_value": base_sales, "current_value": current_sales, "delta_abs": delta_sales, "pct_of_base_sales": delta_sales / base_sales if base_sales else np.nan},
        {"component": "gc_contribution_to_sales", "base_value": base_gc, "current_value": current_gc, "delta_abs": gc_abs, "pct_of_base_sales": gc_abs / base_sales if base_sales else np.nan},
        {"component": "ac_contribution_to_sales", "base_value": base_ac, "current_value": current_ac, "delta_abs": ac_abs, "pct_of_base_sales": ac_abs / base_sales if base_sales else np.nan},
        {"component": "interaction", "base_value": np.nan, "current_value": np.nan, "delta_abs": interaction_abs, "pct_of_base_sales": interaction_abs / base_sales if base_sales else np.nan},
    ])


def run_accounting_decomp(
    df: pd.DataFrame,
    *,
    period_col: str = "year",
    base_period=2024,
    current_period=2025,
    sales_metric: str = "net_sales",
    additive_metric_map: Optional[Dict[str, str]] = None,
    ratio_metric_map: Optional[Dict[str, Tuple[str, str]]] = None,
    hierarchy_map: Optional[Dict[str, List[List[str]]]] = None,
    auto_generate_orders_for: Optional[Dict[str, Sequence[str]]] = None,
    max_depth: Optional[int] = None,
    split_sales_interaction_half: bool = False,
) -> Dict[str, pd.DataFrame]:
    additive_metric_map = additive_metric_map or {"gc": "gc", "sales": sales_metric}
    ratio_metric_map = ratio_metric_map or {
        "ac": (sales_metric, "gc"),
        "upt": ("units", "gc"),
        "aur": (sales_metric, "units"),
        "original_price_per_unit": ("original_price_sales", "units"),
        "discount_per_unit": ("discount_amount", "units"),
    }
    hierarchy_map = hierarchy_map or {}
    auto_generate_orders_for = auto_generate_orders_for or {}

    for metric_name, dims in auto_generate_orders_for.items():
        hierarchy_map[metric_name] = build_hierarchy_orders(dims, max_depth=max_depth)

    outputs: Dict[str, pd.DataFrame] = {}
    outputs["sales_gc_ac_bridge"] = sales_gc_ac_bridge(
        df=df,
        sales_col=sales_metric,
        gc_col="gc",
        period_col=period_col,
        base_period=base_period,
        current_period=current_period,
        split_interaction_half=split_sales_interaction_half,
    )

    for metric_name, metric_col in additive_metric_map.items():
        hierarchies = hierarchy_map.get(metric_name, [])
        if not hierarchies:
            continue
        frames = []
        for hierarchy in hierarchies:
            frame = additive_hierarchy_decomp(df, metric_col, hierarchy, period_col, base_period, current_period)
            frame.insert(0, "metric_name", metric_name)
            frames.append(frame)
        outputs[f"additive__{metric_name}"] = pd.concat(frames, ignore_index=True)

    for metric_name, (num_col, den_col) in ratio_metric_map.items():
        hierarchies = hierarchy_map.get(metric_name, [])
        if not hierarchies:
            continue
        frames = []
        for hierarchy in hierarchies:
            frame = ratio_hierarchy_decomp(df, num_col, den_col, hierarchy, metric_name, period_col, base_period, current_period)
            frame.insert(0, "metric_name", metric_name)
            frames.append(frame)
        outputs[f"ratio__{metric_name}"] = pd.concat(frames, ignore_index=True)

    return outputs
