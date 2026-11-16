"""S-REL: regression calibration with child-specific predictor reliability and its child-level bootstrap.

- **Calibrated predictor.** M1 (or M1-H2; M3 / M4 in FAST / IC) is refitted with ``sqrt(rho_i) * a_wb`` in place of
  ``a_wb`` (in every ``a_wb`` term and in the random slope): ``E[true | observed]`` in true-SD units for standardised
  variables, i.e. each child's slope divided by ``sqrt(rho_i)``. The final random structure of the main fit is
  refitted without the simplification path (``reduction=False, fixed_state=...``).
- **rho_i.** AVSRT: the child's split-time Spearman-Brown reliability (Pearson r of ``alpha_time1`` / ``alpha_time2``
  over the child's predictor-set windows with a valid window, >= 20 windows;
  :func:`feature_extraction.decomposition.split_half_r`). FAST / IC (split-time not computable at 500 ms): the
  child's split-channel value (``alpha_half_left`` / ``alpha_half_right``) times the pooled AVSRT ratio 0.741
  (unanchored correction, reported as such). A child whose rho_i is undefined (< 20 windows) or <= 0 takes the pooled
  value (AVSRT 0.642; FAST 0.587; IC 0.593); the count is reported. Values are capped at 1.
- **Bootstrap.** B = 499 resamples of children with replacement (``numpy.random.default_rng(20261020)``); a child
  drawn k times enters k times as distinct clusters (identifier suffix ``~b<k>``); each resampled child's rho_i is
  recomputed from its own windows inside the loop (:func:`child_rho` on that child's window rows); the same random
  structure is refitted; 95% (and, for S-DTOST, 90%) percentile intervals.

Predictor-only inputs (the predictor table); no behaviour is read here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extraction.decomposition import split_half_r, spearman_brown

__all__ = ["SEED", "B_DEFAULT", "RATIO_SPLIT_TIME_TO_CHANNEL", "POOLED_RHO", "child_rho", "rho_table",
           "load_rel_windows", "calibrated", "resample", "srel_bootstrap"]

SEED = 20261020
B_DEFAULT = 499
RATIO_SPLIT_TIME_TO_CHANNEL = 0.741
POOLED_RHO = {"AVSRT": 0.642, "FAST": 0.587, "IC": 0.593}
MIN_WINDOWS = 20


def child_rho(windows: pd.DataFrame, task: str) -> tuple[float, bool]:
    """``(rho_i, fallback_used)`` from one child's predictor-set windows (module docstring)."""
    if task == "AVSRT":
        r, _ = split_half_r(windows["alpha_time1"].to_numpy(float), windows["alpha_time2"].to_numpy(float),
                            MIN_WINDOWS)
        rho = spearman_brown(r)
    else:
        r, _ = split_half_r(windows["alpha_half_left"].to_numpy(float), windows["alpha_half_right"].to_numpy(float),
                            MIN_WINDOWS)
        rho = spearman_brown(r) * RATIO_SPLIT_TIME_TO_CHANNEL
    if not np.isfinite(rho) or rho <= 0:
        return POOLED_RHO[task], True
    return float(min(rho, 1.0)), False


def rho_table(windows: pd.DataFrame, task: str, child_col: str = "pid") -> pd.DataFrame:
    rows = []
    for c, g in windows.groupby(child_col, sort=True):
        rho, fb = child_rho(g, task)
        rows.append({child_col: c, "rho": rho, "fallback": fb, "n_windows": int(len(g))})
    return pd.DataFrame(rows)


def load_rel_windows(task: str, cohort: str, *, cfg=None, pseudonymizer) -> pd.DataFrame:
    """Predictor-set windows (``in_predictor_set`` and ``valid``) with the reliability halves, keyed by ``pid``."""
    if cfg is None:
        from utils.config import load_project_config

        cfg = load_project_config()
    coh = str(cohort).strip().lower()
    if coh not in ("dev", "conf"):
        raise ValueError(f"cohort must be 'dev' or 'conf', not {cohort!r}")
    path = cfg.paths.derivatives / "features" / task / f"{task}_{coh}_features_predictor.parquet"
    halves = ["alpha_time1", "alpha_time2"] if task == "AVSRT" else ["alpha_half_left", "alpha_half_right"]
    p = pd.read_parquet(path, columns=["subject", "in_predictor_set", "valid", "alpha", *halves])
    m = (pd.array(p["in_predictor_set"], dtype="boolean").fillna(False).to_numpy(bool)
         & pd.array(p["valid"], dtype="boolean").fillna(False).to_numpy(bool) & np.isfinite(p["alpha"].to_numpy(float)))
    p = p.loc[m].reset_index(drop=True)
    mp = {s: pseudonymizer(s) for s in sorted(set(p["subject"]))}
    p.insert(0, "pid", p["subject"].map(mp))
    return p.drop(columns=["subject"])


def calibrated(md, rho: Mapping[str, float]):
    """Copy of ``md`` with ``a_wb`` replaced by ``sqrt(rho_i) * a_wb`` (every ``a_wb`` term and the random slope)."""
    f = md.frame.copy()
    r = f["pid"].map(rho).astype(float)
    if not np.isfinite(r).all():
        raise ValueError("rho missing for some children")
    f["a_wb_uncalibrated"] = f["a_wb"].to_numpy(float)
    f["a_wb"] = np.sqrt(r.to_numpy()) * f["a_wb"].to_numpy(float)
    return replace(md, frame=f, meta={**md.meta, "s_rel": {"calibrated": True}})


def resample(md, windows: pd.DataFrame, draws: Sequence[int], task: str) -> tuple:
    """Bootstrap data: the children at positions ``draws`` (with replacement) as distinct clusters, each with its
    rho_i recomputed from its own windows. Returns ``(md_b, rho_b, n_fallback)``."""
    kids = sorted(md.frame["pid"].unique())
    fr_by = {k: g for k, g in md.frame.groupby("pid", sort=True)}
    win_by = {k: g for k, g in windows.groupby("pid", sort=True)}
    parts, rho, nfb = [], {}, 0
    count: dict[str, int] = {}
    for j in draws:
        k = kids[j]
        count[k] = count.get(k, 0) + 1
        new = f"{k}~b{count[k]}"
        g = fr_by[k].copy()
        g["pid"] = new
        g["pid_run_block"] = new + "|" + g["block"].astype(str)
        parts.append(g)
        rr, fb = child_rho(win_by.get(k, windows.iloc[0:0]), task)     # recomputed inside the loop
        rho[new] = rr
        nfb += int(fb)
    f = pd.concat(parts, ignore_index=True).sort_values(["pid", "run", "sample"], kind="stable").reset_index(drop=True)
    return replace(md, frame=f), rho, nfb


def srel_bootstrap(md, windows: pd.DataFrame, *, fit: Callable[[object, str], Mapping], target: str,
                   B: int = B_DEFAULT, seed: int = SEED, workers: int = 8, level: float = 0.95) -> dict:
    """S-REL estimate (calibrated refit of the full data) and its child-level percentile bootstrap.

    ``fit(md, label) -> result`` must fit the final random structure of the main model (e.g.
    ``r_bridge.fit(..., reduction=False, fixed_state=final_state)``) and return the engine result.
    """
    task = md.task
    rt = rho_table(windows[windows["pid"].isin(md.frame["pid"].unique())], task)
    rho = dict(zip(rt["pid"], rt["rho"]))
    for k in md.frame["pid"].unique():
        if k not in rho:
            rho[k] = POOLED_RHO[task]
    full = fit(calibrated(md, rho), "srel_full")
    est = full["targets"][target]["estimate"]
    kids = sorted(md.frame["pid"].unique())
    G = len(kids)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, G, G) for _ in range(B)]

    def one(b: int):
        mb, rb, nfb = resample(md, windows, draws[b], task)
        try:
            r = fit(calibrated(mb, rb), f"srel_b{b + 1:04d}")
            return float(r["targets"][target]["estimate"]), nfb, None
        except Exception as e:  # noqa: BLE001 - recorded
            return float("nan"), nfb, str(e)[:200]

    with ThreadPoolExecutor(max(1, workers)) as ex:
        out = list(ex.map(one, range(B)))
    bs = np.array([o[0] for o in out])
    ok = np.isfinite(bs)
    a = (1 - level) / 2
    ci = [float(np.quantile(bs[ok], a)), float(np.quantile(bs[ok], 1 - a))] if ok.any() else [math.nan, math.nan]
    ci90 = [float(np.quantile(bs[ok], 0.05)), float(np.quantile(bs[ok], 0.95))] if ok.any() else [math.nan] * 2
    rho_bar = float(np.median(list(rho.values())))
    prim = None
    return {"target": target, "estimate": float(est), "se_model": float(full["targets"][target]["se"]),
            "ci95_percentile": ci, "ci90_percentile": ci90, "B": int(B), "B_ok": int(ok.sum()),
            "B_errors": [o[2] for o in out if o[2]][:5], "seed": int(seed),
            "bootstrap_sd": float(np.std(bs[ok], ddof=1)) if ok.sum() > 1 else math.nan,
            "rho_children": int(len(rt)), "rho_fallback_children": int(rt["fallback"].sum()),
            "rho_median": rho_bar, "rho_min": float(rt["rho"].min()), "rho_max": float(rt["rho"].max()),
            "mean_bootstrap_fallbacks": float(np.mean([o[1] for o in out])),
            "final_state": full.get("final_state", {}).get("re"), "flag_full": full.get("flag"),
            "primary": prim,
            "bootstrap_estimates": bs.tolist()}


def dtost(srel: Mapping, sesoi: float = 0.0102) -> dict:
    """S-DTOST: the S-REL child-bootstrap 90% CI inside +-SESOI (true-score scale)."""
    lo, hi = srel["ci90_percentile"]
    return {"ci90_percentile": [lo, hi], "sesoi": sesoi, "equivalent": bool(-sesoi < lo and hi < sesoi)}


__all__.append("dtost")
_ = Path  # keep import for type hints in callers
