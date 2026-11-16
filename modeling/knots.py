"""Spline knots and the trial-timing covariates of the RT and miss models.

The knot table (``configs/model_knots.yaml``) is computed from the trial tables only (behaviour timing and stimulus
timing; no EEG, no alpha, no group): the model rows are the trial-table analysis subsets of ``configs/events.yaml``.
The EEG window rules are not applied here (they need the predictor tables; removing about 2-3% of windows does not
move pooled quantiles materially).

Covariate definitions
---------------------
- ``log_tsp`` = ln(min(``t_since_prev_press_ms_concat`` / 1000, 10)) (s; capped at 10 s); the join indicator
  ``prev_press_crosses_join`` is a separate term. Rows without an earlier press in the run are missing.
- ``soa`` (AVSRT) = ``soa_prev_ms_concat`` / 1000 (s); FAST ``soa_prev`` the same (linear, centred at the median).
- ``t_run`` = z-score within child x task (mean, SD ddof 1 over the model rows) of ``trial_index_in_run``.
- ``tib`` = z-score within child x task of the 1-based position in the task-native block: AVSRT ``trial_in_block``
  (201-delimited block, every A/V/AV row); FAST the position among all images of the half-block
  (``block_status`` x ``half_status``, rank of ``trial_index_in_run``); IC the position among the flips of the
  180-contour sequence (``seq180_index``, rank of ``trial_index_in_run``). Ranks run over the trial table, not over
  the model rows, so they do not depend on outcomes.
- ``log_tsfl`` (IC) = ln(``t_since_prev_flip_ms_concat`` / 1000).
- ``t_since_prev_contour`` (IC) = ``t_since_prev_contour_ms_concat`` / 1000 (s).
- ``win_start_lag`` (S-LAG) = ``soa_prev_ms_concat`` / 1000 + window start (s; -0.85 AVSRT, -0.55 FAST / IC), the
  time from the previous onset (any stimulus code) to the start of the prestimulus window.

Quantile rule
-------------
For a term ``ns(x, df)`` (``splines::ns``, no intercept) the knots are those ``ns(x, df)`` would choose on the pooled
model rows of all children: ``df - 1`` interior knots at the pooled quantiles with probabilities ``k / df``
(k = 1 .. df - 1; R ``quantile`` type 7 = numpy ``linear``) and the boundary knots at the pooled minimum and maximum,
over the rows with a finite value. The model code calls ``ns(x, knots = <interior>, Boundary.knots = <boundary>)``;
on the pooled rows this is exactly ``ns(x, df)``. A value outside the boundary is extrapolated linearly by ``ns``
(possible only for a subset model, which uses the same knots).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "TSP_CAP_S", "WINDOW_START_S", "TermSpec", "ModelSpec", "MODEL_SPECS", "TRIAL_COLUMNS", "interior_probs",
    "ns_knots", "zscore_within", "block_position", "covariates",
]

TSP_CAP_S = 10.0
#: Prestimulus window start (s before onset) of the primary window, for ``win_start_lag``.
WINDOW_START_S = {"AVSRT": -0.85, "FAST": -0.55, "IC": -0.55}

#: Trial-table columns read: keys, timing, block structure (no response column except through the subset flags).
TRIAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "AVSRT": ("subject", "run", "sample", "trial_index_in_run", "trial_in_block", "block_index_in_run",
              "t_since_prev_press_ms_concat", "soa_prev_ms_concat"),
    "FAST": ("subject", "run", "sample", "trial_index_in_run", "block_status", "half_status",
             "t_since_prev_press_ms_concat", "soa_prev_ms_concat"),
    "IC": ("subject", "run", "sample", "trial_index_in_run", "seq180_index", "t_since_prev_press_ms_concat",
           "soa_prev_ms_concat", "t_since_prev_flip_ms_concat", "t_since_prev_contour_ms_concat"),
}


@dataclass(frozen=True)
class TermSpec:
    variable: str
    df: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    subset: str                 # configs/events.yaml analysis subset of the model rows
    min_rows: int               # >= 30 analysis rows per child x task
    terms: tuple[TermSpec, ...]
    nonperformance: bool        # behavioural non-performance rule applied (FAST, IC)
    note: str = ""


#: Every ns() term of the models (plus the S-LAG sensitivity analysis) with the rows of its model.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("M1", "AVSRT", "primary_rt", 30,
              (TermSpec("log_tsp", 4), TermSpec("soa", 4), TermSpec("t_run", 3), TermSpec("tib", 3)), False,
              "also M1-H2, M1-hist (rows where a_hist5 is defined use the same knots), M1-raw, A3-NL "
              "(ns(t_run, 3), ns(tib, 3)), S-CENS, H1-A, S5, S6 and the AVSRT part of M5h"),
    ModelSpec("M2", "AVSRT", "predictor_set", 30,
              (TermSpec("log_tsp", 3), TermSpec("soa", 3), TermSpec("t_run", 3), TermSpec("tib", 3)), False,
              "all stimuli with a valid predictor (M2); S1-U, S1-A, E2"),
    ModelSpec("M3", "FAST", "primary_rt", 30,
              (TermSpec("log_tsp", 3), TermSpec("t_run", 3), TermSpec("tib", 3)), True,
              "also M3-H2 and the FAST part of M5h; soa_prev is linear (median below)"),
    ModelSpec("M4", "IC", "primary_rt", 30,
              (TermSpec("log_tsfl", 3), TermSpec("t_since_prev_contour", 3), TermSpec("log_tsp", 3),
               TermSpec("tib", 3), TermSpec("t_run", 3)), True,
              "also M4-H2 and the IC part of M5h"),
    ModelSpec("S-LAG_AVSRT", "AVSRT", "primary_rt", 30, (TermSpec("win_start_lag", 3),), False,
              "added to M1 (S-LAG); in AVSRT win_start_lag = soa - 0.85 s exactly"),
    ModelSpec("S-LAG_FAST", "FAST", "primary_rt", 30, (TermSpec("win_start_lag", 3),), True,
              "added to M3; win_start_lag = soa_prev - 0.55 s"),
    ModelSpec("S-LAG_IC", "IC", "primary_rt", 30, (TermSpec("win_start_lag", 3),), True,
              "added to M4; previous onset = previous contour or flip"),
)


def interior_probs(df: int) -> np.ndarray:
    """Probabilities of the ``df - 1`` interior knots of ``splines::ns(x, df)`` (no intercept)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    return np.arange(1, df) / df


def ns_knots(x: Sequence[float] | np.ndarray, df: int) -> dict:
    """Interior and boundary knots that ``splines::ns(x, df)`` chooses (R quantile type 7) on the finite ``x``."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size < df + 1:
        raise ValueError(f"too few finite values ({v.size}) for df = {df}")
    probs = interior_probs(df)
    interior = np.quantile(v, probs, method="linear") if probs.size else np.zeros(0)
    return {"df": int(df), "probs": [float(p) for p in probs], "interior_knots": [float(k) for k in interior],
            "boundary_knots": [float(v.min()), float(v.max())], "n": int(v.size),
            "n_distinct": int(np.unique(v).size)}


def zscore_within(values: pd.Series, groups: pd.Series) -> pd.Series:
    """z-score of ``values`` within ``groups`` (mean, SD ddof 1 over the finite values of each group)."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    mu = v.groupby(groups).transform("mean")
    sd = v.groupby(groups).transform(lambda s: s.std(ddof=1))
    return ((v - mu) / sd).where(sd > 0)


def block_position(trials: pd.DataFrame, task: str) -> pd.Series:
    """1-based position of each row in its task-native block, over all rows of the trial table (module docstring)."""
    if task == "AVSRT":
        return pd.to_numeric(trials["trial_in_block"], errors="coerce").astype(float)
    keys = {"FAST": ["subject", "run", "block_status", "half_status"], "IC": ["subject", "run", "seq180_index"]}[task]
    t = trials.copy()
    for k in keys:
        t[k] = t[k].astype("float").fillna(-1.0) if k not in ("subject",) else t[k]
    pos = t.groupby(keys, sort=False)["trial_index_in_run"].rank(method="first")
    blank = np.zeros(len(t), bool)
    for k in keys[2:]:
        blank |= trials[k].isna().to_numpy()
    return pos.where(~blank).astype(float)


def covariates(trials: pd.DataFrame, task: str, model_rows: pd.Series | np.ndarray) -> pd.DataFrame:
    """Model covariates (module docstring) for the ``model_rows`` of a trial table; z-scores over those rows."""
    rows = np.asarray(model_rows, bool)
    pos = block_position(trials, task)
    d = trials.loc[rows, ["subject", "run", "sample"]].copy()
    tsp = pd.to_numeric(trials.loc[rows, "t_since_prev_press_ms_concat"], errors="coerce").astype(float) / 1000.0
    d["log_tsp"] = np.log(np.minimum(tsp, TSP_CAP_S))
    soa = pd.to_numeric(trials.loc[rows, "soa_prev_ms_concat"], errors="coerce").astype(float) / 1000.0
    d["soa"] = soa
    d["win_start_lag"] = soa + WINDOW_START_S[task]
    d["t_run"] = zscore_within(trials.loc[rows, "trial_index_in_run"], d["subject"])
    d["tib"] = zscore_within(pos[rows], d["subject"])
    d["t_run_raw"] = pd.to_numeric(trials.loc[rows, "trial_index_in_run"], errors="coerce").astype(float)
    d["tib_raw"] = pos[rows].to_numpy()
    if task == "IC":
        tsf = pd.to_numeric(trials.loc[rows, "t_since_prev_flip_ms_concat"], errors="coerce").astype(float)
        d["log_tsfl"] = np.log(tsf / 1000.0)
        tc = pd.to_numeric(trials.loc[rows, "t_since_prev_contour_ms_concat"], errors="coerce").astype(float)
        d["t_since_prev_contour"] = tc / 1000.0
    return d
