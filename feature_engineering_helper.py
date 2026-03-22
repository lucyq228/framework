"""
Feature engineering helpers for panel data (store × week).
Use from notebooks: %run feature_engineering_helper.py  then call make_tiny_df(),
or: from feature_engineering_helper import make_tiny_df

Also includes **MLflow coefficient stability** helpers aligned with ``mlflow_structure.ipynb``
(section “Retrieve results + feature stability summary”): load logged ``coefficients.csv``
artifacts and aggregate sign / magnitude stability across search runs.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Tuple, Optional, Union, Literal, Sequence, Any

def make_tiny_df():
    weeks = pd.date_range("2023-01-01", "2025-06-30", freq="W-SUN")
    n_weeks = len(weeks)
    store_ids = list(range(1, 11))  # 10 stores: 1, 2, ..., 10
    n_stores = len(store_ids)

    rng = np.random.default_rng(42)
    n = n_weeks * n_stores

    # Each store has the same week_start: cross product of store_id x week_start
    df = pd.DataFrame({
        "store_id": np.repeat(store_ids, n_weeks),
        "week_start": np.tile(weeks, n_stores),
        # target: raw GC (no log)
        "GC": rng.uniform(100, 10000, n),
    })

    # digital_promo features: within 0-1
    for c in [
        "digital_promo_1",
        "digital_promo_2",
        "digital_promo_3",
        "digital_promo_4",
        "digital_promo_5",
    ]:
        df[c] = rng.uniform(0, 1, n)

    # media features: within 100-100000
    for c in ["media_1", "media_2", "media_3", "media_4", "media_5"]:
        df[c] = rng.uniform(100, 100_000, n)

    return df


# =========================================================
# 1) Missing values: median (train-only friendly) + indicator
# =========================================================
def add_missing_imputation_features(
    df: pd.DataFrame,
    cols: Optional[Union[List[str], pd.Index]] = None,
    *,
    strategy: str = "median",                 # "median" or "median+indicator"
    suffix_imputed: str = "_imp",
    suffix_missing: str = "_miss",
    medians: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply simple missing handling on a pandas DataFrame.

    - strategy="median": create <col>_imp with NaNs filled by median
    - strategy="median+indicator": also create <col>_miss = 1 if NaN else 0
    - medians: pass precomputed medians (TRAIN) to avoid leakage; if None, compute on df.
    - cols: columns to impute. If None, uses all **numeric** columns (datetime/strings are skipped).
      Non-numeric columns in `cols` are skipped with a warning.

    Returns:
      df_out: df with new columns
      medians_used: dict(col -> median) you can reuse for test/production
    """
    if strategy not in {"median", "median+indicator"}:
        raise ValueError("strategy must be 'median' or 'median+indicator'")

    df_out = df.copy()
    medians_used: Dict[str, float] = {}

    if cols is None:
        cols = list(df_out.select_dtypes(include="number").columns)
    else:
        cols = list(cols)

    resolved_cols: List[str] = []
    for c in cols:
        if c not in df_out.columns:
            raise ValueError(f"Column not found: {c}")
        if not pd.api.types.is_numeric_dtype(df_out[c]):
            warnings.warn(
                f"Skipping non-numeric column {c!r} (median imputation requires numeric). "
                "Use cols=None or pass only numeric feature columns.",
                UserWarning,
                stacklevel=2,
            )
            continue
        resolved_cols.append(c)

    for c in resolved_cols:
        s = pd.to_numeric(df_out[c], errors="coerce").astype(float)
        med = medians[c] if medians is not None and c in medians else float(s.median(skipna=True))
        medians_used[c] = med

        if strategy == "median+indicator":
            df_out[f"{c}{suffix_missing}"] = s.isna().astype(int)

        df_out[f"{c}{suffix_imputed}"] = s.fillna(med)

    return df_out, medians_used


# =========================================================
# 2) Seasonality controls
#    - none
#    - fourier_1 (sin/cos)
#    - fourier_2 (sin/cos)
#    - week-of-year dummies (52)
# =========================================================
def add_seasonality_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    seasonality: str = "none",   # "none" | "fourier_1" | "fourier_2" | "woy_dummies"
    prefix: str = "seas",
    drop_first_woy: bool = True,
) -> pd.DataFrame:
    """
    Adds seasonality features derived from date_col.
    Uses ISO week-of-year for dummies and standard 52-week Fourier cycles.

    Notes:
      - Fourier features are compact and usually preferred for scale.
      - woy_dummies creates up to 52 columns (minus one if drop_first_woy=True).
    """
    if seasonality not in {"none", "fourier_1", "fourier_2", "woy_dummies"}:
        raise ValueError("seasonality must be one of: none, fourier_1, fourier_2, woy_dummies")

    df_out = df.copy()
    dt = pd.to_datetime(df_out[date_col])

    # ISO week number: 1..53 (some years have week 53)
    woy = dt.dt.isocalendar().week.astype(int)
    # Map week 53 -> 52 for stability (optional but keeps consistent dimensionality)
    woy = woy.clip(upper=52)

    if seasonality == "none":
        return df_out

    if seasonality in {"fourier_1", "fourier_2"}:
        # Convert week-of-year to angle in [0, 2pi)
        theta = 2.0 * np.pi * (woy - 1) / 52.0

        # Harmonic 1
        df_out[f"{prefix}_sin1"] = np.sin(theta)
        df_out[f"{prefix}_cos1"] = np.cos(theta)

        if seasonality == "fourier_2":
            # Harmonic 2
            df_out[f"{prefix}_sin2"] = np.sin(2 * theta)
            df_out[f"{prefix}_cos2"] = np.cos(2 * theta)

        return df_out

    # week-of-year dummies
    dummies = pd.get_dummies(woy, prefix=f"{prefix}_woy", dtype=int)
    # Ensure consistent column ordering (1..52)
    all_cols = [f"{prefix}_woy_{i}" for i in range(1, 53)]
    for col in all_cols:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[all_cols]

    if drop_first_woy:
        dummies = dummies.iloc[:, 1:]  # drop week 1 dummy

    df_out = pd.concat([df_out, dummies], axis=1)
    return df_out


# =========================================================
# 3) Trend controls
#    - none
#    - linear
#    - linear_quadratic
# =========================================================
def add_trend_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    trend: str = "none",         # "none" | "linear" | "linear_quadratic"
    prefix: str = "trend",
    unit: str = "week",          # "day" or "week"
) -> pd.DataFrame:
    """
    Adds global time trend features derived from date_col.

    Implementation:
      - trend_t: numeric index starting at 0
      - trend_t2: squared term (optional)

    unit:
      - "day": uses days since min(date)
      - "week": uses weeks since min(date) (recommended for weekly data)
    """
    if trend not in {"none", "linear", "linear_quadratic"}:
        raise ValueError("trend must be one of: none, linear, linear_quadratic")
    if unit not in {"day", "week"}:
        raise ValueError("unit must be 'day' or 'week'")

    df_out = df.copy()
    dt = pd.to_datetime(df_out[date_col])
    t_days = (dt - dt.min()).dt.days.astype(float)

    t = t_days if unit == "day" else (t_days / 7.0)

    if trend == "none":
        return df_out

    df_out[f"{prefix}_t"] = t

    if trend == "linear_quadratic":
        df_out[f"{prefix}_t2"] = df_out[f"{prefix}_t"] ** 2

    return df_out


def feature_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per column with counts, missingness, and distribution summaries.

    Columns
    -------
    count, unique, missing, missing_pct : all dtypes
    min, max, 1%, 25%, 50%, mean, 75%, 99% : numeric, datetime, and timedelta columns;
        None for other dtypes (e.g. string, categorical).
    """
    ordered_cols = [
        "feature",
        "count",
        "unique",
        "missing",
        "missing_pct",
        "min",
        "max",
        "1%",
        "25%",
        "50%",
        "mean",
        "75%",
        "99%",
    ]

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=ordered_cols)

    n_rows = len(df)
    stats: List[Dict] = []

    for c in df.columns:
        s = df[c]
        count = int(s.count())
        unique = int(s.nunique(dropna=True))
        missing = int(s.isna().sum())
        missing_pct = (missing / n_rows * 100.0) if n_rows else 0.0

        min_val = max_val = q1 = q25 = q50 = mean_val = q75 = q99 = None

        # Numeric (int/float/bool): min/max/mean/quantiles
        if pd.api.types.is_numeric_dtype(s.dtype):
            min_val = s.min(skipna=True)
            max_val = s.max(skipna=True)
            q1 = s.quantile(0.01)
            q25 = s.quantile(0.25)
            q50 = s.quantile(0.50)
            mean_val = s.mean()
            q75 = s.quantile(0.75)
            q99 = s.quantile(0.99)
        # Datetime: quantiles return timestamps; mean is NaT if not supported — use median only
        elif pd.api.types.is_datetime64_any_dtype(s.dtype):
            min_val = s.min(skipna=True)
            max_val = s.max(skipna=True)
            q1 = s.quantile(0.01)
            q25 = s.quantile(0.25)
            q50 = s.quantile(0.50)
            q75 = s.quantile(0.75)
            q99 = s.quantile(0.99)
            mean_val = s.mean(skipna=True)
        elif pd.api.types.is_timedelta64_dtype(s.dtype):
            min_val = s.min(skipna=True)
            max_val = s.max(skipna=True)
            q1 = s.quantile(0.01)
            q25 = s.quantile(0.25)
            q50 = s.quantile(0.50)
            mean_val = s.mean(skipna=True)
            q75 = s.quantile(0.75)
            q99 = s.quantile(0.99)

        stats.append(
            {
                "feature": c,
                "count": count,
                "unique": unique,
                "missing": missing,
                "missing_pct": missing_pct,
                "min": min_val,
                "max": max_val,
                "1%": q1,
                "25%": q25,
                "50%": q50,
                "mean": mean_val,
                "75%": q75,
                "99%": q99,
            }
        )

    table = pd.DataFrame(stats)
    return table[ordered_cols]


def feature_correlation_and_vif_diagnostics(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    *,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Diagnostics for modeling: pairwise correlations, each feature vs target, and VIF.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    features : list of str
        Feature column names (must not include ``target``).
    target : str
        Name of the target / dependent column.
    dropna : bool, default True
        If True, drop rows with any NaN in ``[target] + features`` before all calculations.

    Returns
    -------
    pairwise_corr : pd.DataFrame
        Square correlation matrix: index = feature names (rows), columns = feature names (cols).
    target_corr : pd.DataFrame
        Two columns: ``feature`` and ``corr_with_<target>`` (Pearson correlation with target).
    vif_table : pd.DataFrame
        Columns ``feature`` and ``vif`` (variance inflation factor vs other features).

    Notes
    -----
    - Pairwise and target correlations use Pearson correlation on numeric values.
    - VIF requires at least 2 features; with one feature, VIF is defined as 1.0.
    - Constant (zero-variance) columns raise ``ValueError`` for VIF.
    """
    if target in features:
        raise ValueError("`target` must not appear in `features`; list modeling features only.")

    # Preserve order, remove duplicates
    features = list(dict.fromkeys(features))

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    if not features:
        raise ValueError("`features` must be non-empty.")

    sub = df[[target] + list(features)].copy()
    for c in [target] + list(features):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    if dropna:
        sub = sub.dropna(axis=0, how="any")

    n = len(sub)
    if n < 2:
        warnings.warn(
            f"Only {n} row(s) after dropna; correlations may be empty or unreliable.",
            UserWarning,
            stacklevel=2,
        )

    X = sub[features]

    # 1) Pairwise correlation matrix (rows and cols = feature names)
    pairwise_corr = X.corr()

    # 2) Correlation of each feature with target
    corr_series = X.corrwith(sub[target])
    target_corr = pd.DataFrame(
        {
            "feature": corr_series.index.astype(str),
            f"corr_with_{target}": corr_series.values,
        }
    )

    # 3) VIF (requires numeric matrix, no constant columns)
    if len(features) == 1:
        vif_table = pd.DataFrame({"feature": features, "vif": [1.0]})
    else:
        nunique = X.nunique(dropna=False)
        zero_var = nunique[nunique <= 1].index.tolist()
        if zero_var:
            raise ValueError(
                f"Cannot compute VIF: zero-variance columns in features: {zero_var}"
            )
        X_np = X.astype(float).values
        vifs = [variance_inflation_factor(X_np, i) for i in range(len(features))]
        vif_table = pd.DataFrame({"feature": list(features), "vif": vifs})

    return pairwise_corr, target_corr, vif_table


def select_features_by_corr_and_vif(
    pairwise_corr: pd.DataFrame,
    target_corr: pd.DataFrame,
    vif_table: pd.DataFrame,
    max_pairwise_corr: float = 0.9,
    max_vif: float = 5.0,
    verbose: bool = True,
) -> list:
    """
    Given outputs from feature_correlation_and_vif_diagnostics, select features
    by removing those that exceed pairwise correlation or VIF thresholds.

    Steps:
      1. For any pair of features with abs(pairwise corr) > max_pairwise_corr, remove the one
         with smaller |correlation to target| (from target_corr).
      2. After resolving pairs, remove features with VIF > max_vif.
      3. Print removals as they occur.
      4. Return the list of retained feature names.

    Returns
    -------
    kept_features : list of strings

    Examples
    --------
    Use after :func:`feature_correlation_and_vif_diagnostics` on the same ``df`` / feature list::

        df = make_tiny_df()
        candidate_features = [
            c for c in df.columns
            if c.startswith("digital_promo_") or c.startswith("media_")
        ]

        pairwise_corr, target_corr, vif_table = feature_correlation_and_vif_diagnostics(
            df,
            features=candidate_features,
            target="GC",
            dropna=True,
        )

        kept = select_features_by_corr_and_vif(
            pairwise_corr,
            target_corr,
            vif_table,
            max_pairwise_corr=0.9,
            max_vif=5.0,
            verbose=True,
        )
        # kept is the subset to use as model inputs (e.g. pass into your panel / MLflow grid)
    """
    # Prepare initial list of features
    features = list(pairwise_corr.columns)
    # Make dict for fast lookup of correlation of feature to target
    target_corr_dict = target_corr.set_index("feature").iloc[:, 0].to_dict()
    features_set = set(features)
    removed = set()

    # 1. Remove features in highly correlated pairs
    corr_matrix = pairwise_corr.copy()
    # Only upper triangle, avoid self-corrs
    for i, f1 in enumerate(features):
        if f1 not in features_set:
            continue
        for j, f2 in enumerate(features):
            if j <= i:
                continue
            if f2 not in features_set:
                continue
            val = corr_matrix.loc[f1, f2]
            if abs(val) > max_pairwise_corr:
                c1 = abs(target_corr_dict.get(f1, 0))
                c2 = abs(target_corr_dict.get(f2, 0))
                if c1 < c2:
                    remove = f1
                    keep = f2
                else:
                    remove = f2
                    keep = f1
                features_set.discard(remove)  # remove from set
                removed.add(remove)
                if verbose:
                    print(
                        f"Remove '{remove}' due to high pairwise corr |{val:.3f}| > {max_pairwise_corr} "
                        f"with '{keep}' (corr with target: {remove}={c1:.3f}, {keep}={c2:.3f})"
                    )
    kept_features = list(features_set)

    # 2. Remove features with high VIF
    curr_vif_table = vif_table.set_index("feature").copy()
    final_keep = []
    for f in kept_features:
        vif_val = curr_vif_table.loc[f, "vif"]
        if vif_val > max_vif:
            removed.add(f)
            if verbose:
                print(
                    f"Remove '{f}' due to VIF={vif_val:.3f} > {max_vif}"
                )
        else:
            final_keep.append(f)

    if verbose:
        print(f"Selected features after filtering: {final_keep}")

    return final_keep




def _log_transform_array(
    x: np.ndarray,
    *,
    non_positive: Literal["nan", "log1p", "eps"] = "nan",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized log-related transform with explicit rules for non-positive and non-finite values.

    Input ``x`` is float array (NaN allowed).

    **+inf**  → ``+inf`` (``log(+inf)`` / ``log1p(+inf)`` is +inf).

    **-inf**  → ``nan`` (undefined).

    **nan**   → ``nan``.

    Modes
    -----
    ``nan`` (strict ``log``)
        * ``x > 0`` finite → ``log(x)``
        * ``x <= 0`` finite, ``x == 0`` → ``nan``
        * ``x < 0`` finite → ``nan``

    ``log1p``  (``log(1+x)``, domain ``x > -1`` for finite values)
        * ``-1 < x`` finite → ``log1p(x)``  (so ``x == 0`` → ``0``)
        * ``x <= -1`` finite → ``nan``
        * Use for nonnegative count-like data with zeros.

    ``eps``  (shifted log ``log(x + eps)``)
        * Requires ``eps > 0``.
        * ``x + eps > 0`` and finite → ``log(x + eps)`` (covers zero and small negatives above ``-eps``)
        * ``x <= -eps`` finite → ``nan``
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    if non_positive == "eps" and eps <= 0:
        raise ValueError("eps must be positive when non_positive='eps'.")

    if non_positive == "nan":
        # Strict natural log: only x > 0 finite; +inf -> +inf
        m = np.isfinite(x) & (x > 0)
        out[m] = np.log(x[m])
        out[np.isposinf(x)] = np.inf
        # x <= 0 finite, x == 0, x < 0, nan, -inf -> nan

    elif non_positive == "log1p":
        # log(1+x); finite domain x > -1 (at x == -1, log1p = -inf -> treat as invalid)
        m = np.isfinite(x) & (x > -1.0)
        out[m] = np.log1p(x[m])
        out[np.isposinf(x)] = np.inf
        # x <= -1 finite, nan, -inf -> nan

    elif non_positive == "eps":
        m = np.isfinite(x) & (x > -eps)
        out[m] = np.log(x[m] + eps)
        out[np.isposinf(x)] = np.inf
        # x <= -eps finite, nan, -inf -> nan

    else:
        raise ValueError("non_positive must be 'nan', 'log1p', or 'eps'.")

    return out


def add_log_features(
    df: pd.DataFrame,
    features: Union[List[str], pd.Index],
    *,
    non_positive: Literal["nan", "log1p", "eps"] = "nan",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Add log-transformed columns ``{feature}_log`` for each listed feature.

    Parameters
    ----------
    df
        Input data.
    features
        Column names to transform (must exist in ``df``).
    non_positive
        How to treat values that are not strictly positive (and edge cases):

        * ``nan`` (default): ``log(x)`` only for finite ``x > 0``; ``x <= 0`` → NaN.
          ``+inf`` → ``+inf``; ``-inf`` / invalid → NaN.
        * ``log1p``: ``log1p(x)`` for finite ``x > -1`` (zeros map to ``0.0``).
          ``+inf`` → ``+inf``.
        * ``eps``: ``log(x + eps)`` for finite ``x > -eps`` (requires ``eps > 0``).
          ``+inf`` → ``+inf``.

    eps
        Shift used when ``non_positive='eps'`` (ignored otherwise).

    Notes
    -----
    Values are coerced with ``pd.to_numeric(..., errors='coerce')`` first, so non-numeric
    entries become NaN.
    """
    df_out = df.copy()
    features = list(dict.fromkeys(list(features)))

    missing = [c for c in features if c not in df_out.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    for col in features:
        original_vals = pd.to_numeric(df_out[col], errors="coerce")
        arr = _log_transform_array(
            original_vals.to_numpy(dtype=float, copy=False),
            non_positive=non_positive,
            eps=eps,
        )
        df_out[f"{col}_log"] = arr

    return df_out


# =========================================================
# MLflow: coefficient stability (mlflow_structure.ipynb §10)
# =========================================================


def _require_mlflow():
    try:
        import mlflow  # noqa: WPS433
        return mlflow
    except ImportError as e:
        raise ImportError(
            "mlflow is required for coefficient stability helpers. Install: pip install mlflow"
        ) from e


def default_mlflow_tracking_uri() -> str:
    """Same default file store as ``experiment_helper.run_experiment_search`` (./mlruns_experiment)."""
    track_dir = os.path.abspath("./mlruns_experiment")
    return f"file:///{track_dir.replace(os.sep, '/')}"


def file_tracking_uri_to_local_root(tracking_uri: str) -> str:
    """
    Map a ``file:///...`` MLflow tracking URI to a local directory.

    Raises ``ValueError`` if ``tracking_uri`` is not a file backend (e.g. http server).
    """
    s = (tracking_uri or "").strip()
    if not s.lower().startswith("file:"):
        raise ValueError(
            f"Expected a file:// tracking URI for a local store, got {tracking_uri!r}. "
            "Use load_consolidated_coefficients_from_parent() for remote tracking."
        )
    if s.startswith("file:///"):
        rest = s[8:]
    elif s.startswith("file://"):
        rest = s[7:].lstrip("/")
    else:
        raise ValueError(f"Unrecognized file URI: {tracking_uri!r}")
    rest = rest.replace("/", os.sep)
    return os.path.normpath(rest)


def parent_consolidated_coefficients_disk_path(
    experiment_name: str,
    parent_run_id: str,
    *,
    tracking_uri: Optional[str] = None,
    artifact_filename: str = "consolidated_coefficients.csv",
) -> str:
    """
    Expected **local file-store** path to the parent's consolidated coefficients CSV.

    Layout::

        <store_root>/<experiment_id>/<parent_run_id>/artifacts/<artifact_filename>

    The segment after ``store_root`` is MLflow's **experiment_id** (often ``"1"``, ``"2"``, …),
    **not** the human-readable experiment name. That is why this works in the MLflow UI (which
    resolves the experiment for you) but ``os.path.join(\"mlruns_experiment\", EXPERIMENT_NAME, ...)``
    does not find the file on disk.
    """
    mlflow = _require_mlflow()
    if tracking_uri is None:
        tracking_uri = default_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    root = file_tracking_uri_to_local_root(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")
    return os.path.join(
        root,
        str(exp.experiment_id),
        str(parent_run_id),
        "artifacts",
        artifact_filename,
    )


def is_control_feature_name(
    feature_name: str,
    *,
    exclude_seasonality_trend: bool = True,
    exclude_mundlak_means: bool = True,
) -> bool:
    """
    True if ``feature_name`` is a seasonality/trend/Mundlak-mean/intercept column
    (not a marketing “driver” for stability summaries).
    """
    n = str(feature_name)
    if n in ("const", "Intercept"):
        return True
    if exclude_mundlak_means and "__mean_by_entity" in n:
        return True
    if exclude_seasonality_trend and (n.startswith("seas_") or n.startswith("trend_")):
        return True
    return False


def load_coefficients_for_run(
    run_id: str,
    *,
    artifact_path: str = "artifacts/coefficients.csv",
) -> pd.DataFrame:
    """
    Download coefficient CSV for an MLflow run and load as DataFrame.

    Tries the primary path (as logged by ``experiment_helper.run_one_experiment``), then
    ``artifacts/artifacts/coefficients.csv`` for older layouts where MLflow nested an extra
    ``artifacts`` folder.
    """
    mlflow = _require_mlflow()
    rid = str(run_id)
    candidates = [artifact_path, "artifacts/artifacts/coefficients.csv"]
    seen: set = set()
    ordered = []
    for ap in candidates:
        if ap not in seen:
            seen.add(ap)
            ordered.append(ap)
    last_err: Optional[BaseException] = None
    for ap in ordered:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=rid, artifact_path=ap
            )
            return pd.read_csv(local_path)
        except BaseException as e:
            last_err = e
            continue
    assert last_err is not None
    raise RuntimeError(
        f"Could not download coefficients for run_id={rid!r}; tried {ordered}. "
        f"Last error: {last_err!r}"
    ) from last_err


def get_parent_run_id_by_name(
    experiment_name: str,
    *,
    parent_run_name: str,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Resolve the **parent** (outer) search run id by ``attributes.run_name``.

    Use the same ``parent_run_name`` you passed to ``run_experiment_search(..., parent_run_name=...)``.
    """
    mlflow = _require_mlflow()
    if tracking_uri is None:
        tracking_uri = default_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")

    # MLflow 3+ search: use double-quoted strings (parentheses around AND clauses can fail to parse)
    name_esc = parent_run_name.replace('"', '\\"')
    flt = f'tags.run_type = "parent_search" AND attributes.run_name = "{name_esc}"'
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=flt,
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs is None or len(runs) == 0:
        raise ValueError(
            f"No parent_search run with attributes.run_name={parent_run_name!r} in {experiment_name!r}. "
            "Run the search cell first or pass a valid parent_run_id."
        )
    return str(runs.iloc[0]["run_id"])


# Logged by ``experiment_helper.run_experiment_search`` on the parent run (mlflow_structure pattern).
PARENT_CONSOLIDATED_COEFFICIENTS_PATH = "artifacts/consolidated_coefficients.csv"


def load_consolidated_coefficients_from_parent(
    parent_run_id: str,
    *,
    tracking_uri: Optional[str] = None,
    consolidated_artifact_path: str = PARENT_CONSOLIDATED_COEFFICIENTS_PATH,
) -> pd.DataFrame:
    """
    Load the parent run's single consolidated coefficient table (if logged).

    Returns an empty DataFrame if the artifact is missing or unreadable.
    """
    mlflow = _require_mlflow()
    if tracking_uri is None:
        tracking_uri = default_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=str(parent_run_id),
            artifact_path=consolidated_artifact_path,
        )
        df = pd.read_csv(local)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_combined_coefficient_tables(
    experiment_name: str,
    *,
    tracking_uri: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    filter_string: str = 'tags.run_status = "ok" AND tags.run_type = "trial"',
    artifact_path: str = "artifacts/coefficients.csv",
    prefer_parent_consolidated: bool = True,
    consolidated_artifact_path: str = PARENT_CONSOLIDATED_COEFFICIENTS_PATH,
) -> pd.DataFrame:
    """
    Long-format coefficients for stability summaries.

    If ``parent_run_id`` is set and ``prefer_parent_consolidated`` is True (default), loads
    **one** file from the parent run: ``artifacts/consolidated_coefficients.csv``, produced by
    ``run_experiment_search`` after all successful trials (see ``mlflow_structure`` consolidated
    artifact pattern). That file includes ``train_start`` / ``train_end`` / ``test_start`` /
    ``test_end`` on each row.

    Otherwise (or if the consolidated file is missing), finds matching trial runs and
    concatenates each child's ``artifacts/coefficients.csv`` (adding ``run_id`` per row).
    """
    mlflow = _require_mlflow()
    if tracking_uri is None:
        tracking_uri = default_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    if parent_run_id is not None and prefer_parent_consolidated:
        df = load_consolidated_coefficients_from_parent(
            parent_run_id,
            tracking_uri=tracking_uri,
            consolidated_artifact_path=consolidated_artifact_path,
        )
        if not df.empty:
            return df

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()

    flt = filter_string
    if parent_run_id is not None:
        pid = str(parent_run_id)
        flt = f'{filter_string} AND tags."mlflow.parentRunId" = "{pid}"'

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=flt,
    )
    if runs is None or len(runs) == 0:
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    for _, r in runs.iterrows():
        run_id = r["run_id"]
        try:
            cdf = load_coefficients_for_run(run_id, artifact_path=artifact_path)
            cdf = cdf.copy()
            cdf["run_id"] = run_id
            rows.append(cdf)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_combined_child_coefficient_tables(
    experiment_name: str,
    parent_run_id: str,
    *,
    tracking_uri: Optional[str] = None,
    filter_string: str = 'tags.run_status = "ok" AND tags.run_type = "trial"',
    artifact_path: str = "artifacts/coefficients.csv",
    prefer_parent_consolidated: bool = True,
    consolidated_artifact_path: str = PARENT_CONSOLIDATED_COEFFICIENTS_PATH,
) -> pd.DataFrame:
    """
    Coefficients for all trials under a parent search run.

    Prefers the parent's **consolidated** CSV (single download) when present; otherwise
    concatenates each child's ``coefficients.csv``.
    """
    return load_combined_coefficient_tables(
        experiment_name,
        tracking_uri=tracking_uri,
        parent_run_id=parent_run_id,
        filter_string=filter_string,
        artifact_path=artifact_path,
        prefer_parent_consolidated=prefer_parent_consolidated,
        consolidated_artifact_path=consolidated_artifact_path,
    )


def aggregate_feature_stability(
    experiment_name: str,
    *,
    tracking_uri: Optional[str] = None,
    filter_string: str = 'tags.run_status = "ok" AND tags.run_type = "trial"',
    parent_run_id: Optional[str] = None,
    exclude_control_features: bool = False,
    groupby_columns: Sequence[str] = ("feature",),
    prefer_parent_consolidated: bool = True,
    consolidated_artifact_path: str = PARENT_CONSOLIDATED_COEFFICIENTS_PATH,
) -> pd.DataFrame:
    """
    Aggregate coefficient **stability** across MLflow trial runs (``mlflow_structure`` pattern).

    For each feature (or ``(panel_control, feature)``), computes how often it appears,
    mean/median/std of ``coef``, sign consistency, mean rank, and optional inference stats.

    Parameters
    ----------
    experiment_name
        MLflow experiment name (e.g. ``analysis_work_flow_panel``).
    tracking_uri
        Defaults to ``default_mlflow_tracking_uri()`` (``./mlruns_experiment``).
    filter_string
        MLflow ``search_runs`` filter; default keeps successful nested trials only.
        Use double-quoted tag values (MLflow 3+); avoid wrapping compound clauses in ``(...)``.
    parent_run_id
        If set, restricts to runs nested under this parent — use the id returned by
        ``run_experiment_search``. When set, loads ``consolidated_coefficients.csv`` from the
        parent first if ``prefer_parent_consolidated`` is True.
    prefer_parent_consolidated
        If True (default) and ``parent_run_id`` is set, read the parent's consolidated file
        instead of downloading every child artifact.
    exclude_control_features
        If True, drops seasonality / trend / Mundlak mean / intercept rows before aggregating.
    groupby_columns
        Default ``("feature",)``. Use ``("panel_control", "feature")`` for stability **by**
        FE vs Mundlak (requires ``panel_control`` column in ``coefficients.csv``).

    Returns
    -------
    DataFrame sorted by ``n_runs_appeared``, ``mean_abscoef`` descending.
    """
    allc = load_combined_coefficient_tables(
        experiment_name,
        tracking_uri=tracking_uri,
        parent_run_id=parent_run_id,
        filter_string=filter_string,
        prefer_parent_consolidated=prefer_parent_consolidated,
        consolidated_artifact_path=consolidated_artifact_path,
    )
    if allc.empty:
        return pd.DataFrame()
    if "feature" not in allc.columns:
        raise ValueError("coefficients.csv must contain a 'feature' column")

    for c in groupby_columns:
        if c not in allc.columns:
            raise ValueError(
                f"Column {c!r} not in coefficient tables; cannot groupby. "
                f"Available: {list(allc.columns)}"
            )

    if exclude_control_features:
        mask = ~allc["feature"].map(
            lambda x: is_control_feature_name(str(x))
        )
        allc = allc.loc[mask].copy()

    if allc.empty:
        return pd.DataFrame()

    gcols = list(groupby_columns)
    g = allc.groupby(gcols, dropna=False)

    out = pd.DataFrame(
        {
            "n_runs_appeared": g["run_id"].nunique(),
            "mean_coef": g["coef"].mean(),
            "median_coef": g["coef"].median(),
            "std_coef": g["coef"].std(ddof=1),
            "mean_abscoef": g["abs_coef"].mean(),
            "mean_rank": g["rank_abscoef"].mean(),
            "pct_positive": g["sign"].apply(lambda s: (s > 0).mean()),
            "pct_negative": g["sign"].apply(lambda s: (s < 0).mean()),
        }
    ).reset_index()

    if "p_value" in allc.columns and allc["p_value"].notna().any():
        sig = (
            allc.groupby(gcols, dropna=False)["p_value"]
            .apply(lambda s: float((s < 0.05).mean()) if len(s) else np.nan)
            .reset_index(name="pct_significant_05")
        )
        out = out.merge(sig, on=gcols, how="left")

    # Coefficient of variation (use |mean| in denominator; mlflow_structure “from_study” pattern)
    out["coef_cv"] = out["std_coef"] / out["mean_coef"].abs().replace(0, np.nan)

    sort_cols = [("n_runs_appeared", False), ("mean_abscoef", False)]
    out = out.sort_values(
        [c for c, _ in sort_cols],
        ascending=[a for _, a in sort_cols],
    ).reset_index(drop=True)
    return out


def coefficient_stability_report_tables(
    experiment_name: str,
    *,
    parent_run_id: Optional[str] = None,
    parent_run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    prefer_parent_consolidated: bool = True,
    consolidated_artifact_path: str = PARENT_CONSOLIDATED_COEFFICIENTS_PATH,
) -> Dict[str, pd.DataFrame]:
    """
    Build the usual **coefficient stability report** tables after ``run_experiment_search``.

    Returns
    -------
    dict with keys:

    - ``drivers_only`` — seasonality/trend/Mundlak controls excluded (recommended).
    - ``all_model_terms`` — every regressor including controls.
    - ``by_panel_control`` — drivers only, grouped by ``panel_control`` × ``feature`` (FE vs Mundlak).

    Exactly one of ``parent_run_id`` or ``parent_run_name`` should be set (or pass
    ``parent_run_name`` for convenience).

    By default reads ``artifacts/consolidated_coefficients.csv`` from the parent run (written by
    ``run_experiment_search``); set ``prefer_parent_consolidated=False`` to rebuild from child CSVs.
    """
    if parent_run_id is None and parent_run_name is not None:
        parent_run_id = get_parent_run_id_by_name(
            experiment_name,
            parent_run_name=parent_run_name,
            tracking_uri=tracking_uri,
        )
    if parent_run_id is None:
        raise ValueError("Pass parent_run_id or parent_run_name.")

    pid = str(parent_run_id)
    drivers = aggregate_feature_stability(
        experiment_name,
        tracking_uri=tracking_uri,
        parent_run_id=pid,
        exclude_control_features=True,
        groupby_columns=("feature",),
        prefer_parent_consolidated=prefer_parent_consolidated,
        consolidated_artifact_path=consolidated_artifact_path,
    )
    all_terms = aggregate_feature_stability(
        experiment_name,
        tracking_uri=tracking_uri,
        parent_run_id=pid,
        exclude_control_features=False,
        groupby_columns=("feature",),
        prefer_parent_consolidated=prefer_parent_consolidated,
        consolidated_artifact_path=consolidated_artifact_path,
    )
    try:
        by_pc = aggregate_feature_stability(
            experiment_name,
            tracking_uri=tracking_uri,
            parent_run_id=pid,
            exclude_control_features=True,
            groupby_columns=("panel_control", "feature"),
            prefer_parent_consolidated=prefer_parent_consolidated,
            consolidated_artifact_path=consolidated_artifact_path,
        )
    except ValueError as e:
        warnings.warn(
            f"Skipping by_panel_control table: {e}",
            UserWarning,
            stacklevel=2,
        )
        by_pc = pd.DataFrame()

    return {
        "drivers_only": drivers,
        "all_model_terms": all_terms,
        "by_panel_control": by_pc,
    }


def save_coefficient_stability_report(
    tables: Dict[str, pd.DataFrame],
    *,
    export_dir: str = "reports",
    prefix: str = "coef_stability",
) -> Dict[str, str]:
    """
    Write stability tables to CSV under ``export_dir`` (created if missing).

    Returns paths written: ``{key: path}``.
    """
    os.makedirs(export_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for key, df in tables.items():
        if df is None or df.empty:
            continue
        safe = key.replace(" ", "_")
        path = os.path.join(export_dir, f"{prefix}__{safe}.csv")
        df.to_csv(path, index=False)
        paths[key] = path
    return paths


def plot_consolidated_coef_stability_by_window(
    consolidated_coef_df: pd.DataFrame,
    features: Sequence[str],
    *,
    panel_control: str,
    title: str,
    alpha: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    jitter_seed: int = 42,
    window_jitter_width: float = 0.18,
    point_jitter_std: float = 0.02,
    legend_fontsize: int = 9,
    xtick_fontsize: int = 9,
    scatter_size: float = 72.0,
    legend: str = "below",
    tight_layout_rect: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[Any, Any]:
    """
    Scatter plot: coefficient vs feature, with points spread by training window.

    Uses rows from ``consolidated_coef_df`` where ``panel_control`` matches ``panel_control``
    (case-insensitive, e.g. ``\"fe\"`` or ``\"mundlak\"``) and ``feature`` is in ``features``.
    Points are colored by train window when ``p_value < alpha``; otherwise gray.

    Parameters
    ----------
    consolidated_coef_df
        Long table from parent ``consolidated_coefficients.csv`` (needs ``feature``, ``coef``,
        ``panel_control``, ``train_start``, ``train_end``; optional ``p_value``).
    features
        Ordered feature names for the x-axis (duplicates removed, order preserved).
    panel_control
        Model / panel type to filter (must match values in ``panel_control`` column).
    title
        Axes title.
    alpha
        Significance threshold for coloring (vs gray).
    scatter_size
        Marker size for ``scatter``.
    legend
        ``"below"`` (default): legend under the plot so the axes span nearly the full figure width.
        ``"right"``: legend outside to the right (narrower axes).
    tight_layout_rect
        Passed to ``fig.tight_layout(rect=...)``. Default depends on ``legend``.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes. Caller typically ``plt.show()`` or ``fig.savefig(...)``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    if consolidated_coef_df is None or len(consolidated_coef_df) == 0:
        raise ValueError("consolidated_coef_df is empty.")

    need = {"feature", "coef", "panel_control", "train_start", "train_end"}
    miss = need - set(consolidated_coef_df.columns)
    if miss:
        raise ValueError(f"consolidated_coef_df missing columns: {sorted(miss)}")

    chart_features = list(dict.fromkeys(features))
    if not chart_features:
        raise ValueError("features must contain at least one name.")

    pc_target = str(panel_control).strip().lower()
    plot_df = consolidated_coef_df[
        consolidated_coef_df["panel_control"].astype(str).str.lower().eq(pc_target)
        & consolidated_coef_df["feature"].isin(chart_features)
    ].copy()
    if plot_df.empty:
        raise ValueError(
            f"No rows for panel_control={panel_control!r} and given features; "
            f"check panel_control values and feature names."
        )

    plot_df["train_window"] = (
        plot_df["train_start"].astype(str) + " → " + plot_df["train_end"].astype(str)
    )

    if "p_value" in plot_df.columns:
        plot_df["significant"] = plot_df["p_value"].notna() & (
            plot_df["p_value"] < float(alpha)
        )
    else:
        plot_df["significant"] = True

    x_index = {f: i for i, f in enumerate(chart_features)}
    plot_df["x0"] = plot_df["feature"].map(x_index)
    windows = sorted(plot_df["train_window"].unique())
    j_off = {
        w: (k - (len(windows) - 1) / 2.0) * float(window_jitter_width)
        for k, w in enumerate(windows)
    }
    plot_df["x"] = plot_df["x0"] + plot_df["train_window"].map(j_off)
    rng = np.random.default_rng(jitter_seed)
    plot_df["x"] = plot_df["x"] + rng.normal(0, float(point_jitter_std), size=len(plot_df))

    nfeat = len(chart_features)
    if figsize is None:
        # Wide per-feature width + tall enough for rotated labels; legend below uses full width
        figsize = (max(14.0, nfeat * 2.2), 7.8)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    cmap = plt.get_cmap("tab10")
    win_colors = {w: cmap(i % 10) for i, w in enumerate(windows)}

    for _, row in plot_df.iterrows():
        if row["significant"]:
            c = win_colors[row["train_window"]]
            ec = "0.25"
        else:
            c = "#b5b5b5"
            ec = "#777777"
        ax.scatter(
            row["x"],
            row["coef"],
            c=[c],
            edgecolors=ec,
            linewidths=0.5,
            s=scatter_size,
            zorder=3,
        )

    ax.axhline(0.0, color="black", linewidth=0.85, alpha=0.35, zorder=1)
    ax.set_xlim(-0.5, nfeat - 0.5)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(nfeat)))
    ax.set_xticks(np.arange(nfeat))
    ax.set_xticklabels(chart_features, rotation=55, ha="right", fontsize=xtick_fontsize)
    for lbl in ax.get_xticklabels():
        lbl.set_visible(True)
    ax.minorticks_off()
    ax.set_ylabel("Coefficient")
    ax.set_xlabel("Feature")
    ax.set_title(title)

    leg_el = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=win_colors[w],
            markeredgecolor="0.25",
            markersize=8,
            label=w,
        )
        for w in windows
    ]
    leg_el.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#b5b5b5",
            markeredgecolor="#777777",
            markersize=8,
            label=f"p ≥ {alpha:g} or missing p",
        )
    )
    _legend_mode = str(legend).strip().lower()
    if _legend_mode in ("below", "bottom", "under"):
        _ncol = min(5, max(2, len(leg_el)))
        ax.legend(
            handles=leg_el,
            title="Train window / significance",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=_ncol,
            fontsize=legend_fontsize,
            frameon=True,
            framealpha=0.95,
        )
        if tight_layout_rect is None:
            tight_layout_rect = (0.03, 0.28, 0.98, 0.96)
    else:
        ax.legend(
            handles=leg_el,
            title="Train window / significance",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=legend_fontsize,
        )
        if tight_layout_rect is None:
            tight_layout_rect = (0.02, 0.22, 0.82, 0.96)

    if tight_layout_rect is not None:
        fig.tight_layout(rect=list(tight_layout_rect))

    return fig, ax