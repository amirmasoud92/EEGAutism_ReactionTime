"""Single-trial alpha reliability per child and its pooled summaries.

Inputs are the combined per-task feature tables; no behaviour column is read and nothing is split by group. Per
child x task (rows in the primary predictor set with a finite ``alpha``, i.e. ``in_predictor_set`` and ``valid``):

- **split-channel**: Pearson r of ``alpha_half_left`` and ``alpha_half_right`` (hemisphere halves PO7/PO3/O1/POz vs
  PO8/PO4/O2/Oz), Spearman-Brown ``2r / (1 + r)``; this is the child alpha reliability used as a model moderator;
- **split-time** (AVSRT): the same for ``alpha_time1`` / ``alpha_time2`` (the two 400-ms halves of the 800-ms window);
- with ``>= 20`` windows (``features.yaml`` ``reliability_min_windows``), computed by
  :func:`feature_extraction.decomposition.child_reliability`;
- **within-block** variant: both halves centred on their child x run x block means (the ``block`` column of the
  decomposition), i.e. the reliability of the ``a_wb``-type deviations.

Pooled summaries: median, IQR, p05 / p95, mean; child-level 95% CIs by a percentile bootstrap over children
(``n_boot``, fixed seed) for the median and the mean; per child a Fisher-z 95% CI of r mapped through
Spearman-Brown.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["SEED", "fisher_ci", "spearman_brown", "child_table", "pooled_summary", "within_block_centre",
           "true_score_bounds"]

SEED = 20261020


def spearman_brown(r):
    r = np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(r) & (r > -1), 2 * r / (1 + r), np.nan)


def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher-z CI of a Pearson r with n pairs (NaN for n <= 3 or |r| >= 1)."""
    from scipy import stats

    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    h = stats.norm.ppf(0.5 + level / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - h)), float(np.tanh(z + h))


def within_block_centre(df: pd.DataFrame, cols: Sequence[str], child_col: str = "subject",
                        block_cols: Sequence[str] = ("run", "block")) -> pd.DataFrame:
    """Copy with each column of ``cols`` centred on its child x run x block mean (finite rows)."""
    out = df.copy()
    keys = [child_col, *block_cols]
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce").astype(float)
        out[c] = v - v.groupby([out[k] for k in keys]).transform("mean")
    return out


def child_table(df: pd.DataFrame, *, dcfg=None, child_col: str = "subject", centre_blocks: bool = False
                ) -> pd.DataFrame:
    """Per-child r, Spearman-Brown reliability and its Fisher-z CI (split-channel and, if present, split-time)."""
    from feature_extraction.decomposition import child_reliability, predictor_mask

    d = df
    if centre_blocks:
        m = predictor_mask(df, "alpha", "in_predictor_set")
        cols = [c for c in ("alpha_half_left", "alpha_half_right", "alpha_time1", "alpha_time2") if c in df.columns]
        d = within_block_centre(df[m], cols, child_col)
    rel = child_reliability(d, child_col=child_col, dcfg=dcfg)
    for kind in ("split_channel", "split_time"):
        if f"r_{kind}" not in rel.columns:
            continue
        lo, hi = zip(*[fisher_ci(r, n) for r, n in zip(rel[f"r_{kind}"], rel[f"n_{kind}"])]) if len(rel) else ((), ())
        rel[f"rel_{kind}_ci_lo"] = spearman_brown(np.asarray(lo, float))
        rel[f"rel_{kind}_ci_hi"] = spearman_brown(np.asarray(hi, float))
    return rel


def pooled_summary(values: Sequence[float], *, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Median, IQR, p05 / p95, mean and child-bootstrap 95% CIs of the median and the mean."""
    a = np.asarray([v for v in values if np.isfinite(v)], float)
    if a.size == 0:
        return {"n_children": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    bs = a[idx]
    med_b, mean_b = np.median(bs, axis=1), bs.mean(axis=1)
    return {"n_children": int(a.size), "median": float(np.median(a)), "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)), "p05": float(np.quantile(a, 0.05)),
            "p95": float(np.quantile(a, 0.95)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "median_ci95_child_bootstrap": [float(np.quantile(med_b, 0.025)), float(np.quantile(med_b, 0.975))],
            "mean_ci95_child_bootstrap": [float(np.quantile(mean_b, 0.025)), float(np.quantile(mean_b, 0.975))],
            "n_boot": int(n_boot), "seed": int(seed)}


def true_score_bounds(rho_bar: float, sesoi: dict[str, float], r_sesoi: float = 0.03) -> dict:
    """True-score bound SESOI / sqrt(rho_bar) on the slope scale (and r_sesoi / sqrt(rho_bar))."""
    s = float(np.sqrt(rho_bar))
    return {"rho_bar": float(rho_bar), "sqrt_rho_bar": s, "r_observed": r_sesoi, "r_true_bound": r_sesoi / s,
            "slopes": {k: {"observed": v, "true_score_bound": v / s} for k, v in sesoi.items()}}
