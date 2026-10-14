"""Statistics helpers shared by the shelf tools."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

METHODS = ("pearson", "spearman", "kendall")


def clean_xy(frame: pd.DataFrame, x: str, y: str, extra: list[str] | None = None) -> pd.DataFrame:
    cols = [x, y] + [c for c in (extra or []) if c not in (x, y)]
    return frame[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])


def corr(x, y, method: str = "pearson") -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    fn = {"pearson": sps.pearsonr, "spearman": sps.spearmanr, "kendall": sps.kendalltau}[method]
    r, p = fn(x, y)
    return float(r), float(p)


def fisher_ci(r: float, n: int, level: float = .95) -> tuple[float, float]:
    if not np.isfinite(r) or n < 4:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -.9999, .9999))
    se = 1 / np.sqrt(n - 3)
    q = sps.norm.ppf(.5 + level / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def fisher_diff(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """z statistic and two-sided p for the difference between two independent correlations."""
    if min(n1, n2) < 4 or not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan, np.nan
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * sps.norm.sf(abs(z)))


def weighted_corr(x, y, w) -> float:
    x, y, w = (np.asarray(a, float) for a in (x, y, w))
    if w.sum() <= 0 or len(x) < 4:
        return np.nan
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def cluster_bootstrap(frame: pd.DataFrame, statistic, cluster: str | None = "glin", n_boot: int = 400,
                      seed: int = 27, level: float = .95) -> tuple[float, float]:
    """Percentile CI, resampling whole restaurants so repeated visits are not treated as independent."""
    rng = np.random.default_rng(seed)
    if cluster and cluster in frame.columns:
        groups = {k: idx for k, idx in frame.groupby(cluster).indices.items()}
        keys = np.array(list(groups))
        draws = []
        for _ in range(n_boot):
            pick = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in pick])
            draws.append(statistic(frame.iloc[idx]))
    else:
        draws = [statistic(frame.iloc[rng.integers(0, len(frame), len(frame))]) for _ in range(n_boot)]
    draws = np.array([d for d in draws if np.isfinite(d)])
    if len(draws) < n_boot * .5:
        return np.nan, np.nan
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def bh_fdr(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q.tolist()
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q.tolist()


def distance_corr(x, y) -> float:
    x, y = np.asarray(x, float)[:, None], np.asarray(y, float)[:, None]
    if len(x) > 3000:
        idx = np.random.default_rng(27).choice(len(x), 3000, replace=False)
        x, y = x[idx], y[idx]
    a, b = np.abs(x - x.T), np.abs(y - y.T)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov, dvx, dvy = (A * B).mean(), (A * A).mean(), (B * B).mean()
    return float(np.sqrt(dcov / np.sqrt(dvx * dvy))) if dvx > 0 and dvy > 0 else np.nan


def finding(label: str, measure: str | None, metric: str, estimate: float, n: int, ci=(np.nan, np.nan),
            p: float = np.nan, **extra) -> dict:
    """The standard testable statistic every tool reports. Review and FDR read these."""
    def r(v):
        return None if v is None or not np.isfinite(v) else round(float(v), 4)
    return {"label": label, "measure": measure, "metric": metric, "estimate": r(estimate), "n": int(n),
            "ci_low": r(ci[0]), "ci_high": r(ci[1]), "p": r(p) if p is not None else None, **extra}


def records(frame: pd.DataFrame, digits: int = 4) -> list[dict]:
    out = frame.copy()
    for c in out.select_dtypes(include="number").columns:
        out[c] = out[c].round(digits)
    return out.replace({np.nan: None}).to_dict("records")
