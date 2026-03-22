"""
Lightweight MLflow + panel OLS helpers (FE / Mundlak) for local experiments.

Derived from patterns in ``mlflow_structure.ipynb``. Requires:
  ``mlflow``, ``linearmodels``, ``statsmodels``, ``scikit-learn``.

Typical flow:
  1. ``setup_mlflow(...)``
  2. ``run_experiment_search(...)`` with ``time_split`` or ``time_splits`` and feature sets
     (``mlflow_structure`` pattern: one parent run, grid over time windows × feature sets × panel).
"""
from __future__ import annotations

import math
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional: seasonality / trend from your FE module
try:
    from feature_engineering_helper import add_seasonality_features, add_trend_features
except ImportError:  # pragma: no cover
    add_seasonality_features = None  # type: ignore
    add_trend_features = None  # type: ignore


# ---------------------------------------------------------------------------
# Data containers (aligned with mlflow_structure notebook)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeSplit:
    time_split_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class SimpleExperimentConfig:
    """One trial: target column, panel estimator, and driver feature list."""

    y_col: str
    panel_control: str  # "fe" | "mundlak"
    features: Tuple[str, ...]
    feature_block_set_id: str = "default"
    algorithm: str = "OLS"
    time_split: Optional[TimeSplit] = None
    seed: int = 42


@dataclass
class FitResult:
    model: object
    coef_df: pd.DataFrame
    predict_fn: Callable[[pd.DataFrame], np.ndarray]


def standardize_coef_df(
    feature_index,
    coef,
    t_stat=None,
    p_value=None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "feature": list(feature_index),
            "coef": np.asarray(coef, dtype=float),
            "t_stat": np.asarray(t_stat, dtype=float) if t_stat is not None else np.nan,
            "p_value": np.asarray(p_value, dtype=float) if p_value is not None else np.nan,
        }
    )
    df["abs_coef"] = df["coef"].abs()
    df["sign"] = np.sign(df["coef"]).astype(int)
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df["rank_abscoef"] = np.arange(1, len(df) + 1)
    df["is_significant_05"] = df["p_value"].apply(
        lambda x: (x < 0.05) if pd.notna(x) else False
    )
    return df


def fit_fe_linearmodels(
    df_train: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    entity_col: str = "store_id",
    time_col: str = "week_start",
    time_effects: bool = False,
) -> FitResult:
    df_tr = df_train.copy()
    df_tr[time_col] = pd.to_datetime(df_tr[time_col])
    df_tr = df_tr.set_index([entity_col, time_col]).sort_index()

    y = df_tr[y_col].astype(float)
    X = df_tr[x_cols].astype(float)

    mod = PanelOLS(
        y,
        X,
        entity_effects=True,
        time_effects=time_effects,
        check_rank=False,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    coef_df = standardize_coef_df(
        res.params.index,
        res.params.values,
        t_stat=res.tstats.reindex(res.params.index).values,
        p_value=res.pvalues.reindex(res.params.index).values,
    )

    def predict_fn(df_any: pd.DataFrame) -> np.ndarray:
        df_te = df_any.copy()
        df_te[time_col] = pd.to_datetime(df_te[time_col])
        df_te = df_te.set_index([entity_col, time_col]).sort_index()
        X_te = df_te[x_cols].astype(float)
        pred = res.predict(exog=X_te)
        return np.asarray(pred).ravel()

    return FitResult(model=res, coef_df=coef_df, predict_fn=predict_fn)


def fit_mundlak_linearmodels(
    df_train: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    entity_col: str = "store_id",
    time_col: str = "week_start",
    include_intercept: bool = True,
) -> FitResult:
    df_tr = df_train.copy()
    df_tr[time_col] = pd.to_datetime(df_tr[time_col])

    means_by_entity = df_tr.groupby(entity_col)[x_cols].mean()

    def make_X_aug(df_any: pd.DataFrame) -> pd.DataFrame:
        df_any = df_any.copy()
        m = df_any[[entity_col]].merge(
            means_by_entity.reset_index(),
            on=entity_col,
            how="left",
        )
        mean_cols = {c: f"{c}__mean_by_entity" for c in x_cols}
        for c in x_cols:
            df_any[mean_cols[c]] = m[c].values
        X = df_any[x_cols].astype(float)
        X_means = df_any[[mean_cols[c] for c in x_cols]].astype(float)
        X_aug = pd.concat([X, X_means], axis=1)
        if include_intercept:
            X_aug = sm.add_constant(X_aug, has_constant="add")
        return X_aug

    df_tr = df_tr.set_index([entity_col, time_col]).sort_index()
    y = df_tr[y_col].astype(float)

    df_tr_reset = df_tr.reset_index()
    X_aug = make_X_aug(df_tr_reset)
    X_aug.index = df_tr.index

    mod = RandomEffects(y, X_aug, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    coef_df = standardize_coef_df(
        res.params.index,
        res.params.values,
        t_stat=res.tstats.reindex(res.params.index).values,
        p_value=res.pvalues.reindex(res.params.index).values,
    )

    def predict_fn(df_any: pd.DataFrame) -> np.ndarray:
        df_te = df_any.copy()
        df_te[time_col] = pd.to_datetime(df_te[time_col])
        df_te = df_te.set_index([entity_col, time_col]).sort_index()
        df_te_reset = df_te.reset_index()
        X_te_aug = make_X_aug(df_te_reset)
        X_te_aug.index = df_te.index
        pred = res.predict(exog=X_te_aug)
        return np.asarray(pred).ravel()

    return FitResult(model=res, coef_df=coef_df, predict_fn=predict_fn)


def fit_panel_backend(
    df_train: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    panel_control: str,
) -> FitResult:
    pc = panel_control.lower().strip()
    if pc == "fe":
        return fit_fe_linearmodels(df_train, y_col, x_cols, time_effects=False)
    if pc == "mundlak":
        return fit_mundlak_linearmodels(df_train, y_col, x_cols, include_intercept=True)
    raise ValueError("panel_control must be 'fe' or 'mundlak'")


# ---------------------------------------------------------------------------
# Time split + search strategy (feature blocks / k-per-domain — mlflow_structure pattern)
# ---------------------------------------------------------------------------


def slice_by_time(df: pd.DataFrame, ts: TimeSplit) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test by ``week_start`` using inclusive bounds (compare as Timestamps)."""
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    train_start = pd.Timestamp(ts.train_start)
    train_end = pd.Timestamp(ts.train_end)
    test_start = pd.Timestamp(ts.test_start)
    test_end = pd.Timestamp(ts.test_end)
    train = df[(df["week_start"] >= train_start) & (df["week_start"] <= train_end)]
    test = df[(df["week_start"] >= test_start) & (df["week_start"] <= test_end)]
    return train, test


def search_strategy_table(
    feature_blocks: Dict[str, List[str]],
    k_per_block: Dict[str, int],
    *,
    n_stochastic_samples: int,
) -> pd.DataFrame:
    """
    One row per **domain** (feature block): how many features exist vs how many are drawn
    per stochastic sample (``k_per_block``), plus a row for the stochastic search size.

    Mirrors the intent of ``sample_k_per_block`` in ``mlflow_structure.ipynb``.
    """
    rows = []
    for domain in sorted(feature_blocks.keys()):
        feats = feature_blocks[domain]
        k = k_per_block.get(domain)
        rows.append(
            {
                "domain": domain,
                "n_features_in_block": len(feats),
                "k_candidates_per_sample": k,
            }
        )
    rows.append(
        {
            "domain": "_stochastic_search_",
            "n_features_in_block": np.nan,
            "k_candidates_per_sample": float(n_stochastic_samples),
        }
    )
    return pd.DataFrame(rows)


def time_splits_summary_table(time_splits: Sequence[TimeSplit]) -> pd.DataFrame:
    """
    Tabular view of train/test windows for display / logging (``mlflow_structure`` multi-split pattern).

    Columns: ``time_split_id``, ``train_start``, ``train_end``, ``test_start``, ``test_end``.
    """
    if not time_splits:
        return pd.DataFrame(
            columns=[
                "time_split_id",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
            ]
        )
    return pd.DataFrame([asdict(ts) for ts in time_splits])


def _normalize_time_splits_argument(
    time_split: Optional[Union[TimeSplit, Sequence[TimeSplit]]],
    time_splits: Optional[Sequence[TimeSplit]],
) -> List[TimeSplit]:
    if time_split is not None and time_splits is not None:
        raise ValueError("Pass only one of time_split or time_splits.")
    if time_splits is not None:
        out = list(time_splits)
        if not out:
            raise ValueError("time_splits must be a non-empty sequence.")
        for i, x in enumerate(out):
            if not isinstance(x, TimeSplit):
                raise TypeError(
                    f"time_splits[{i}] must be a TimeSplit instance, got {type(x)!r}."
                )
        return out
    if time_split is None:
        raise ValueError("Pass time_split or a non-empty time_splits sequence.")

    if isinstance(time_split, TimeSplit):
        return [time_split]

    # Common mistake: run_experiment_search(..., time_split=my_list) instead of time_splits=my_list
    if isinstance(time_split, (list, tuple)) and len(time_split) > 0:
        if all(isinstance(x, TimeSplit) for x in time_split):
            return list(time_split)
        bad = next(x for x in time_split if not isinstance(x, TimeSplit))
        raise TypeError(
            "If you pass a list of windows, use the keyword time_splits=..., not time_split=.... "
            f"(Found non-TimeSplit element {type(bad)!r}.)"
        )

    raise TypeError(
        "time_split must be a single TimeSplit. For multiple windows use "
        "time_splits=[TimeSplit(...), ...], not time_split=[...]. "
        f"(Got {type(time_split)!r}.)"
    )


def enumerate_block_feature_sets(
    feature_blocks: Dict[str, List[str]],
    *,
    include_all_union: bool = True,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Deterministic sets: one run per block (all features in that domain) and optionally
    one run with the union of all blocks — same idea as commented ``generate_feature_sets``
    in the notebook.
    """
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for block, feats in feature_blocks.items():
        out.append((f"block__{block}", tuple(sorted(feats))))
    if include_all_union and feature_blocks:
        all_feats = tuple(
            sorted({f for feats in feature_blocks.values() for f in feats})
        )
        out.append(("block__ALL", all_feats))
    return out


def sample_k_per_block(
    feature_blocks: Dict[str, List[str]],
    k_per_block: Dict[str, int],
    n_samples: int,
    seed: int = 42,
    allow_all_if_small: bool = True,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Stochastic search: each sample picks ``k`` features **from each domain** (block) without
    replacement within that block, then unions them. Same as ``mlflow_structure.ipynb``.

    Returns list of ``(feature_set_id, features_tuple)``.
    """
    rng = np.random.default_rng(seed)
    blocks = list(feature_blocks.keys())
    if set(k_per_block.keys()) != set(blocks):
        missing = set(blocks) - set(k_per_block.keys())
        extra = set(k_per_block.keys()) - set(blocks)
        raise ValueError(
            f"k_per_block keys must match feature_blocks keys. missing={missing}, extra={extra}"
        )

    results: List[Tuple[str, Tuple[str, ...]]] = []
    seen: Set[Tuple[str, ...]] = set()

    for i in range(n_samples):
        chosen: List[str] = []
        for b in blocks:
            feats = feature_blocks[b]
            k = k_per_block[b]
            if len(feats) < k:
                if allow_all_if_small:
                    pick = list(feats)
                else:
                    raise ValueError(
                        f"Block {b!r} has only {len(feats)} features but k={k}."
                    )
            else:
                pick = rng.choice(feats, size=k, replace=False).tolist()
            chosen.extend(pick)

        chosen_sorted = tuple(sorted(set(chosen)))
        if chosen_sorted in seen:
            continue
        seen.add(chosen_sorted)
        fs_id = f"kperblock__{i:04d}"
        results.append((fs_id, chosen_sorted))

    return results


def _dedupe_feature_sets(
    items: Sequence[Tuple[str, Tuple[str, ...]]],
) -> List[Tuple[str, Tuple[str, ...]]]:
    seen: Set[Tuple[str, ...]] = set()
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for fs_id, feats in items:
        key = tuple(sorted(feats))
        if key in seen:
            continue
        seen.add(key)
        out.append((fs_id, feats))
    return out


def build_feature_sets_from_blocks(
    feature_blocks: Dict[str, List[str]],
    k_per_block: Dict[str, int],
    *,
    n_stochastic_samples: int = 0,
    seed: int = 42,
    include_deterministic_block_sets: bool = True,
    allow_all_if_small: bool = True,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Combine deterministic per-domain (and ALL) sets with optional ``sample_k_per_block`` draws.
    """
    parts: List[Tuple[str, Tuple[str, ...]]] = []
    if include_deterministic_block_sets:
        parts.extend(enumerate_block_feature_sets(feature_blocks, include_all_union=True))
    if n_stochastic_samples > 0:
        parts.extend(
            sample_k_per_block(
                feature_blocks,
                k_per_block,
                n_samples=n_stochastic_samples,
                seed=seed,
                allow_all_if_small=allow_all_if_small,
            )
        )
    if not parts:
        raise ValueError(
            "No feature sets: enable include_deterministic_block_sets or n_stochastic_samples > 0."
        )
    return _dedupe_feature_sets(parts)


def build_control_columns(
    seasonality_mode: str,
    trend_mode: str,
) -> List[str]:
    """Column names produced by ``apply_fe_controls`` for given modes."""
    control_cols: List[str] = []
    if seasonality_mode == "fourier_1":
        control_cols += ["seas_sin1", "seas_cos1"]
    elif seasonality_mode == "fourier_2":
        control_cols += ["seas_sin1", "seas_cos1", "seas_sin2", "seas_cos2"]
    elif seasonality_mode == "woy_dummies":
        control_cols.append("seas_woy_*")  # placeholder; resolved on df in apply_fe_controls
    if trend_mode == "linear":
        control_cols += ["trend_t"]
    elif trend_mode == "linear_quadratic":
        control_cols += ["trend_t", "trend_t2"]
    return control_cols


def apply_fe_controls(
    df: pd.DataFrame,
    *,
    seasonality_mode: str = "fourier_1",
    trend_mode: str = "linear",
    date_col: str = "week_start",
    seas_prefix: str = "seas",
    trend_prefix: str = "trend",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add seasonality + trend columns using ``feature_engineering_helper``.

    Returns
    -------
    df_out : extended DataFrame
    control_cols : list of column names to append to driver features as X
    """
    if add_seasonality_features is None or add_trend_features is None:
        raise ImportError(
            "feature_engineering_helper not importable; add project root to PYTHONPATH "
            "or install the module."
        )
    df_out = add_seasonality_features(
        df.copy(), date_col, seasonality=seasonality_mode, prefix=seas_prefix
    )
    df_out = add_trend_features(
        df_out, date_col, trend=trend_mode, prefix=trend_prefix, unit="week"
    )

    control_cols: List[str] = []
    if seasonality_mode == "fourier_1":
        control_cols += [f"{seas_prefix}_sin1", f"{seas_prefix}_cos1"]
    elif seasonality_mode == "fourier_2":
        control_cols += [
            f"{seas_prefix}_sin1",
            f"{seas_prefix}_cos1",
            f"{seas_prefix}_sin2",
            f"{seas_prefix}_cos2",
        ]
    elif seasonality_mode == "woy_dummies":
        control_cols += [
            c for c in df_out.columns if c.startswith(f"{seas_prefix}_woy_")
        ]

    if trend_mode == "linear":
        control_cols += [f"{trend_prefix}_t"]
    elif trend_mode == "linear_quadratic":
        control_cols += [f"{trend_prefix}_t", f"{trend_prefix}_t2"]

    return df_out, control_cols


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_metrics_basic(prefix: str, y_true, y_pred) -> None:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        warnings.warn(f"No finite rows for metrics prefix={prefix!r}", UserWarning)
        return
    yt, yp = y_true[mask], y_pred[mask]
    rmse = math.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    mlflow.log_metrics(
        {
            f"{prefix}_rmse": rmse,
            f"{prefix}_mae": mae,
            f"{prefix}_r2": r2,
        }
    )


def log_dataframe_as_csv(df: pd.DataFrame, artifact_path: str, filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, filename)
        df.to_csv(fpath, index=False)
        mlflow.log_artifact(fpath, artifact_path=artifact_path)


# Single file on the **parent** search run: all successful trials' coefficients (mlflow_structure pattern).
CONSOLIDATED_COEFFICIENTS_ARTIFACT = "artifacts/consolidated_coefficients.csv"


def _read_coefficients_csv_for_run(run_id: str) -> pd.DataFrame:
    """Download a trial's ``artifacts/coefficients.csv`` (with legacy path fallback)."""
    rid = str(run_id)
    last_err: Optional[BaseException] = None
    for ap in ("artifacts/coefficients.csv", "artifacts/artifacts/coefficients.csv"):
        try:
            local = mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=ap)
            return pd.read_csv(local)
        except BaseException as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"Could not load coefficients for run_id={rid!r}; last error: {last_err!r}"
    ) from last_err


def log_consolidated_coefficients_for_parent(
    parent_run_id: str,
    experiment_id: str,
    *,
    filter_string: str = 'tags.run_status = "ok" AND tags.run_type = "trial"',
) -> Optional[pd.DataFrame]:
    """
    Concatenate all successful nested trials' ``coefficients.csv`` and log one CSV on the **parent** run.

    Expects each child row to include ``train_start`` / ``train_end`` / ``test_start`` / ``test_end``
    (added in ``run_one_experiment``). Adds ``run_id`` (child run id) if missing.

    Returns the consolidated DataFrame, or ``None`` if nothing to log.
    """
    pid = str(parent_run_id)
    flt = f'{filter_string} AND tags."mlflow.parentRunId" = "{pid}"'
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=flt)
    if runs is None or len(runs) == 0:
        return None

    parts: List[pd.DataFrame] = []
    for _, r in runs.iterrows():
        rid = r["run_id"]
        try:
            cdf = _read_coefficients_csv_for_run(rid)
            cdf = cdf.copy()
            if "run_id" not in cdf.columns:
                cdf["run_id"] = rid
            parts.append(cdf)
        except Exception:
            continue

    if not parts:
        return None

    big = pd.concat(parts, ignore_index=True)
    log_dataframe_as_csv(
        big,
        artifact_path="artifacts",
        filename="consolidated_coefficients.csv",
    )
    mlflow.log_param("consolidated_coefficients_path", CONSOLIDATED_COEFFICIENTS_ARTIFACT)
    mlflow.log_metric("n_runs_in_consolidated_coef", float(len(parts)))
    mlflow.log_metric("n_rows_in_consolidated_coef", float(len(big)))
    return big


def get_model_summary_text(fit_model: object) -> str:
    if hasattr(fit_model, "summary"):
        try:
            s = fit_model.summary
            if hasattr(s, "as_text"):
                return s.as_text()
            return str(s)
        except Exception:
            pass
    return "Model summary not available for this model type.\n" + repr(fit_model)


def log_config_snapshot(cfg: SimpleExperimentConfig, extra: Optional[dict] = None) -> None:
    payload = {
        "y_col": cfg.y_col,
        "panel_control": cfg.panel_control,
        "algorithm": cfg.algorithm,
        "feature_block_set_id": cfg.feature_block_set_id,
        "features": list(cfg.features),
        "seed": cfg.seed,
        "time_split": asdict(cfg.time_split) if cfg.time_split else None,
    }
    if extra:
        payload.update(extra)
    mlflow.log_dict(payload, artifact_file="run_config.json")


def log_run_params(cfg: SimpleExperimentConfig, n_features_x: int) -> None:
    mlflow.log_params(
        {
            "y_col": cfg.y_col,
            "panel_control": cfg.panel_control,
            "algorithm": cfg.algorithm,
            "feature_block_set_id": cfg.feature_block_set_id,
            "n_driver_features": len(cfg.features),
            "n_x_cols": n_features_x,
            "seed": cfg.seed,
        }
    )
    if cfg.time_split:
        mlflow.log_params(
            {
                "time_split_id": cfg.time_split.time_split_id,
                "train_start": cfg.time_split.train_start,
                "train_end": cfg.time_split.train_end,
                "test_start": cfg.time_split.test_start,
                "test_end": cfg.time_split.test_end,
            }
        )


# ---------------------------------------------------------------------------
# Single run + search grid
# ---------------------------------------------------------------------------


def run_one_experiment(
    df_pd: pd.DataFrame,
    cfg: SimpleExperimentConfig,
    *,
    apply_fe_controls_flag: bool = True,
    seasonality_mode: str = "fourier_1",
    trend_mode: str = "linear",
) -> None:
    """
    One MLflow child run: slice by time, optional FE controls, fit FE or Mundlak, log metrics/artifacts.

    Expects columns ``store_id``, ``week_start``, ``cfg.y_col``, and all ``cfg.features``.
    """
    if cfg.time_split is None:
        raise ValueError("cfg.time_split is required")

    run_name = (
        f"{cfg.y_col}__{cfg.panel_control}__{cfg.algorithm}"
        f"__{cfg.feature_block_set_id}__{cfg.time_split.time_split_id}"
    )

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("run_type", "trial")
        try:
            work = df_pd.copy()
            if apply_fe_controls_flag:
                work, control_cols = apply_fe_controls(
                    work,
                    seasonality_mode=seasonality_mode,
                    trend_mode=trend_mode,
                )
                mlflow.log_param("seasonality_mode", seasonality_mode)
                mlflow.log_param("trend_mode", trend_mode)
            else:
                control_cols = []

            train_df, test_df = slice_by_time(work, cfg.time_split)
            if len(train_df) == 0 or len(test_df) == 0:
                raise ValueError(
                    f"Empty train or test after time split: train={len(train_df)}, test={len(test_df)}. "
                    "Check TimeSplit vs df['week_start'] range."
                )
            ycol = cfg.y_col
            # Dedupe while preserving order (duplicate names break linearmodels / pandas exog)
            driver_cols = list(dict.fromkeys(cfg.features))
            x_cols = list(dict.fromkeys(driver_cols + control_cols))

            needed = list(set([ycol, "store_id", "week_start"] + x_cols))
            missing = [c for c in needed if c not in train_df.columns]
            if missing:
                raise ValueError(f"Missing columns after FE prep: {missing}")

            log_run_params(cfg, n_features_x=len(x_cols))
            log_config_snapshot(
                cfg,
                extra={
                    "seasonality_mode": seasonality_mode if apply_fe_controls_flag else None,
                    "trend_mode": trend_mode if apply_fe_controls_flag else None,
                    "control_cols": control_cols,
                    "x_cols": x_cols,
                },
            )

            fit_res = fit_panel_backend(train_df, ycol, x_cols, cfg.panel_control)

            pred_train = fit_res.predict_fn(train_df)
            pred_test = fit_res.predict_fn(test_df)

            y_train = train_df[ycol].astype(float).to_numpy()
            y_test = test_df[ycol].astype(float).to_numpy()

            log_metrics_basic("train", y_train, pred_train)
            log_metrics_basic("test", y_test, pred_test)

            cdf = fit_res.coef_df.copy()
            cdf["y_col"] = cfg.y_col
            cdf["panel_control"] = cfg.panel_control
            cdf["algorithm"] = cfg.algorithm
            cdf["feature_block_set_id"] = cfg.feature_block_set_id
            cdf["time_split_id"] = cfg.time_split.time_split_id
            ts = cfg.time_split
            cdf["train_start"] = ts.train_start
            cdf["train_end"] = ts.train_end
            cdf["test_start"] = ts.test_start
            cdf["test_end"] = ts.test_end

            if cdf["p_value"].notna().any():
                mlflow.log_metric(
                    "n_significant_05", float((cdf["p_value"] < 0.05).sum())
                )
                mlflow.log_metric(
                    "pct_significant_05", float((cdf["p_value"] < 0.05).mean())
                )

            log_dataframe_as_csv(cdf, artifact_path="artifacts", filename="coefficients.csv")
            mlflow.log_text(
                get_model_summary_text(fit_res.model),
                "artifacts/model_summary.txt",
            )
        except Exception as e:  # pragma: no cover
            mlflow.set_tag("run_status", "failed")
            mlflow.log_text(str(e), "error.txt")
            raise
        else:
            mlflow.set_tag("run_status", "ok")


def run_experiment_search(
    df: pd.DataFrame,
    *,
    y_col: str,
    time_split: Optional[Union[TimeSplit, Sequence[TimeSplit]]] = None,
    time_splits: Optional[Sequence[TimeSplit]] = None,
    feature_sets: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
    feature_blocks: Optional[Dict[str, List[str]]] = None,
    k_per_block: Optional[Dict[str, int]] = None,
    n_stochastic_samples: int = 0,
    seed: int = 42,
    include_deterministic_block_sets: bool = True,
    panel_controls: Sequence[str] = ("fe", "mundlak"),
    tracking_uri: Optional[str] = None,
    experiment_name: str = "panel_decomp_simple",
    apply_fe_controls_flag: bool = True,
    seasonality_mode: str = "fourier_1",
    trend_mode: str = "linear",
    parent_run_name: str = "search_run",
) -> str:
    """
    **Search strategy**: outer MLflow run, nested runs for each
    **(time window × feature_block_set_id × panel_control)**.

    Pass **either** ``time_split`` (one ``TimeSplit``) **or** ``time_splits`` (sequence of
    windows). If you pass several windows, prefer ``time_splits=[...]``; a list passed to
    ``time_split=`` is also accepted (same effect) to avoid the common typo
    ``time_split=time_splits``.

    Returns
    -------
    parent_run_id : str
        MLflow run id of the **parent** (outer) search run. After all trials, the parent logs
        ``artifacts/consolidated_coefficients.csv`` (all successful child coefficient tables plus
        ``train_*`` / ``test_*`` dates on each row). Stability helpers load that file first when present.

    Provide either:

    * **feature_sets** — explicit list of ``(feature_block_set_id, driver columns)``, or
    * **feature_blocks** + **k_per_block** — domain → feature columns, plus how many features
      to draw **per domain** in each stochastic sample (see ``sample_k_per_block`` in
      ``mlflow_structure.ipynb``), and **n_stochastic_samples** for how many such samples.

    Deterministic sets (``block__<domain>``, ``block__ALL``) are included when
    ``include_deterministic_block_sets`` is True.
    """
    ts_list = _normalize_time_splits_argument(time_split, time_splits)
    if tracking_uri is None:
        track_dir = os.path.abspath("./mlruns_experiment")
        tracking_uri = f"file:///{track_dir.replace(os.sep, '/')}"

    if feature_sets is not None and feature_blocks is not None:
        raise ValueError("Pass only one of feature_sets or feature_blocks.")

    if feature_blocks is not None:
        if k_per_block is None:
            raise ValueError("k_per_block is required when feature_blocks is set.")
        strategy_df = search_strategy_table(
            feature_blocks,
            k_per_block,
            n_stochastic_samples=n_stochastic_samples,
        )
        built = build_feature_sets_from_blocks(
            feature_blocks,
            k_per_block,
            n_stochastic_samples=n_stochastic_samples,
            seed=seed,
            include_deterministic_block_sets=include_deterministic_block_sets,
        )
        feature_sets = built
    elif feature_sets is None:
        raise ValueError("Provide feature_sets or feature_blocks.")

    # Materialize once (avoids exhausting a generator via len(list(...)) before the loop)
    feature_sets_list = list(feature_sets)

    setup_mlflow(tracking_uri, experiment_name)

    with mlflow.start_run(run_name=parent_run_name) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tag("run_type", "parent_search")
        mlflow.log_params(
            {
                "y_col": y_col,
                "n_time_splits": len(ts_list),
                "time_split_ids": ",".join(ts.time_split_id for ts in ts_list),
                "n_feature_sets": len(feature_sets_list),
                "panel_controls": ",".join(panel_controls),
                "n_stochastic_samples": n_stochastic_samples,
                "seed": seed,
            }
        )
        mlflow.log_dict(
            {"time_splits": [asdict(ts) for ts in ts_list]},
            artifact_file="time_splits.json",
        )
        log_dataframe_as_csv(
            time_splits_summary_table(ts_list),
            artifact_path="artifacts",
            filename="time_splits.csv",
        )
        if feature_blocks is not None and k_per_block is not None:
            log_dataframe_as_csv(
                strategy_df,
                artifact_path="artifacts",
                filename="search_strategy_by_domain.csv",
            )
            mlflow.log_dict(
                {
                    "feature_blocks": {k: list(v) for k, v in feature_blocks.items()},
                    "k_per_block": k_per_block,
                    "n_stochastic_samples": n_stochastic_samples,
                    "include_deterministic_block_sets": include_deterministic_block_sets,
                },
                artifact_file="search_strategy.json",
            )

        for ts in ts_list:
            for block_id, feats in feature_sets_list:
                feats_t = tuple(feats)
                for pc in panel_controls:
                    cfg = SimpleExperimentConfig(
                        y_col=y_col,
                        panel_control=pc,
                        features=feats_t,
                        feature_block_set_id=block_id,
                        algorithm="OLS",
                        time_split=ts,
                    )
                    run_one_experiment(
                        df,
                        cfg,
                        apply_fe_controls_flag=apply_fe_controls_flag,
                        seasonality_mode=seasonality_mode,
                        trend_mode=trend_mode,
                    )

        # One consolidated CSV on the parent run (stability + audit without re-downloading every child)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None:
            log_consolidated_coefficients_for_parent(
                parent_run_id,
                exp.experiment_id,
            )

    return parent_run_id


__all__ = [
    "TimeSplit",
    "SimpleExperimentConfig",
    "FitResult",
    "setup_mlflow",
    "slice_by_time",
    "search_strategy_table",
    "time_splits_summary_table",
    "CONSOLIDATED_COEFFICIENTS_ARTIFACT",
    "log_consolidated_coefficients_for_parent",
    "enumerate_block_feature_sets",
    "sample_k_per_block",
    "build_feature_sets_from_blocks",
    "apply_fe_controls",
    "fit_fe_linearmodels",
    "fit_mundlak_linearmodels",
    "fit_panel_backend",
    "run_one_experiment",
    "run_experiment_search",
]
