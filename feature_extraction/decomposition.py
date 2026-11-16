"""Within-child decomposition of the prestimulus alpha table.

Operates on the per-recording tables of :mod:`feature_extraction.prestimulus_alpha`, concatenated over the runs of
each child x task. Everything is within child x task; nothing here reads behaviour, press counts or group. The context
counts ``n_stim10`` / ``n_press10`` that accompany ``a_ctx10`` are computed in :mod:`trials.context_counts`.
Parameters: ``configs/features.yaml`` ``decomposition``.

**Predictor set of a variant.** Row ``i`` enters the centring and scaling of a column (``alpha`` or a sensitivity
variant) when its stimulus-defined set flag is True (``in_predictor_set``; the window variants use their own
``in_s_w500_predictor_set`` etc.) **and** the column is finite (the column is NaN when its window fails the edge /
segment / BAD / ROI-channel / artefact rules).

Definitions (child i, task fixed):

- ``sd_i``: SD (ddof = 1) of the column around the child x task mean.
- Blocks: AVSRT ``block_index_in_run`` (stimuli between consecutive 201 markers); FAST half-block (``block_status`` x
  ``half_status``, label ``1000 x block_status + half_status``); IC 180-contour sequence ``seq180_index``. A missing
  block label is its own block (label -1) per run. **Merge rule** (per task, ``decomposition.merge_mode`` in
  ``configs/features.yaml``). AVSRT and IC (``preceding``): within each run, in block order, the first block with
  fewer than 10 predictor-set windows is merged with the preceding block (with the following block when it is the
  run's first), and this repeats until every block has >= 10 windows or the run has a single block. FAST
  (``greedy``): the half-blocks are accumulated in order into a group until it has >= 10 windows; a smaller
  remainder joins the preceding group (the ``preceding`` rule would collapse every FAST run into one block). Every
  merge is logged (:attr:`Decomposition.merges`).
- ``a_wb`` = (value - child x run x merged-block mean) / ``sd_i`` (the within-block predictor); ``a_slow`` = (block
  mean - child mean) / ``sd_i`` (the block-mean covariate); ``a_wb_raw`` = value - block mean (unscaled).
  ``a_wb + a_slow`` = (value - child mean) / ``sd_i``.
- **Running-mean variant** ``a_harm``: (value - centred running mean over the predictor-set windows of the same child
  and **run** whose onsets lie within +-60 s on the recording clock ``onset_s``, the window itself included) /
  ``sd_i``. The mean **spans joins and bad spans** (``decomposition.harm_scope: run``): each window is already a
  valid window on its own, and only the averaging crosses a join. ``harm_scope: segment`` averages only within the
  same child, run and valid span ``segment_id``.
- **a_hist5**: ``h`` = mean of the column over the (up to) 5 previous predictor-set windows of the same child, run and
  valid span (``segment_id``: between joins and bad spans), requiring >= 3 of them, else missing; previous means
  strictly earlier onsets, so ``h`` of a window never uses that window or any later one. ``a_hist5`` = (h -
  mean_i(h)) / SD_i(h) over the predictor-set windows where h is defined. ``hist5_span_s`` is the onset minus the
  onset of the earliest contributing window.
- **a_ctx10** (``a_ctx20`` sensitivity): ``actx10_log`` z-scored within child x task over the predictor-set windows
  of the primary column where it is defined.
- Child alpha reliability: split-channel (hemisphere halves) and split-time (two half-windows, AVSRT) Pearson r
  across a child's predictor-set windows, Spearman-Brown ``2r / (1 + r)``; >= 20 windows.
- Post-stimulus desynchronisation and time-on-task checks: per-child mean of ``alpha_post_pc1 - alpha`` over valid
  V/AV predictor-set trials, the frequency of its minimum, and the per-child slope of z(alpha) on
  z(``trial_index_in_run``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHILD = "subject"
#: Running-mean averaging scopes (``decomposition.harm_scope``).
HARM_SCOPES: tuple[str, ...] = ("run", "segment")

#: Decomposed predictor columns: variant -> (value column, predictor-set column).
VARIANTS: dict[str, tuple[str, str]] = {
    "primary": ("alpha", "in_predictor_set"),
    "s_w500": ("alpha_s_w500", "in_s_w500_predictor_set"),
    "s_w1000": ("alpha_s_w1000", "in_s_w1000_predictor_set"),
    "s_w400": ("alpha_s_w400", "in_s_w400_predictor_set"),
    "s_est1": ("alpha_s_est1", "in_predictor_set"),
    "s_est2": ("alpha_s_est2", "in_predictor_set"),
    "s_band": ("alpha_s_band", "in_predictor_set"),
    "e9": ("alpha_e9", "in_predictor_set"),
    "s_roi": ("alpha_s_roi", "in_predictor_set"),
    "e7_mu": ("alpha_e7_mu", "in_predictor_set"),
    "s_ica": ("alpha_s_ica", "in_predictor_set"),
    "s_erp": ("alpha_s_erp", "in_predictor_set"),
    "s_iaf": ("alpha_s_iaf", "in_predictor_set"),
}


def _dcfg(dcfg):
    if dcfg is None:
        from utils.config_features import load_features_config

        dcfg = load_features_config().decomposition
    return getattr(dcfg, "decomposition", dcfg)


# --------------------------------------------------------------------------- helpers
def predictor_mask(df: pd.DataFrame, value_col: str, set_col: str) -> np.ndarray:
    """``set_col`` True and ``value_col`` finite."""
    if value_col not in df.columns or set_col not in df.columns:
        return np.zeros(len(df), bool)
    s = pd.array(df[set_col], dtype="boolean").fillna(False).to_numpy(bool)
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float)
    return s & np.isfinite(v)


def native_block(df: pd.DataFrame, task: str, dcfg=None) -> np.ndarray:
    """Task-native block label (int64; -1 = missing) of every row: ``sum_k col_k x 1000^(K-1-k)`` over the
    configured block columns (AVSRT / IC one column; FAST ``1000 x block_status + half_status``)."""
    d = _dcfg(dcfg)
    if task not in d.blocks:
        raise ValueError(f"no block definition for task {task!r}")
    cols = d.blocks[task]
    lab = np.zeros(len(df), np.int64)
    miss = np.zeros(len(df), bool)
    for c in cols:
        if c not in df.columns:
            return np.full(len(df), -1, np.int64)
        v = pd.array(df[c], dtype="Int64")
        na = v.isna()
        x = v.fillna(-1).to_numpy(np.int64)
        if (x >= 1000).any():
            raise ValueError(f"block column {c} has values >= 1000")
        miss |= na | (x < 0)
        lab = lab * 1000 + x
    return np.where(miss, -1, lab)


def merge_small_blocks(labels: Sequence[int], counts: Sequence[int], min_n: int, mode: str = "preceding"
                       ) -> tuple[dict[int, int], list[dict]]:
    """Block merge rule for the blocks of ONE run, given in block order.

    ``mode="preceding"``: repeatedly merge the first block with < ``min_n`` windows into the preceding
    block (the following one for the first block). When every block is small (FAST half-blocks, ~6 windows) this
    cascades into ONE block per run. ``mode="greedy"``: walk the blocks in order and accumulate them
    into a group until it has >= ``min_n`` windows; a small remainder at the end joins the preceding group.

    Returns ``({native label: merged id}, merge log)``; merged ids are 0-based positions of the surviving groups.
    """
    groups: list[list[int]] = [[int(lb)] for lb in labels]
    n = [int(c) for c in counts]
    log: list[dict] = []
    if mode == "greedy":
        out_g: list[list[int]] = []
        out_n: list[int] = []
        cur: list[int] = []
        cn = 0
        for g, c in zip(groups, n):
            cur, cn = cur + g, cn + c
            if cn >= min_n:
                out_g.append(cur)
                out_n.append(cn)
                cur, cn = [], 0
        if cur:
            if out_g:
                out_g[-1] += cur
                out_n[-1] += cn
            else:
                out_g.append(cur)
                out_n.append(cn)
        for g, c in zip(out_g, out_n):
            if len(g) > 1:
                log.append({"merged": list(g[1:]), "into": [g[0]], "n_small": None, "n_target": c,
                            "direction": "greedy"})
        return {lb: k for k, g in enumerate(out_g) for lb in g}, log
    if mode != "preceding":
        raise ValueError(f"unknown merge mode {mode!r}")
    while len(groups) > 1:
        small = [i for i, c in enumerate(n) if c < min_n]
        if not small:
            break
        i = small[0]
        j = i - 1 if i > 0 else 1
        lo, hi = min(i, j), max(i, j)
        log.append({"merged": list(groups[i]), "into": list(groups[j]), "n_small": n[i], "n_target": n[j],
                    "direction": "preceding" if j < i else "following"})
        groups[lo] = groups[lo] + groups[hi]
        n[lo] = n[lo] + n[hi]
        del groups[hi], n[hi]
    return {lb: k for k, g in enumerate(groups) for lb in g}, log


@dataclass
class Decomposition:
    """Columns added by :func:`decompose` and the merge log."""

    columns: pd.DataFrame
    merges: list[dict] = field(default_factory=list)
    sd: dict = field(default_factory=dict)


def decompose(df: pd.DataFrame, value_col: str = "alpha", set_col: str = "in_predictor_set", *, task: str,
              suffix: str = "", child_col: str = CHILD, dcfg=None, merge_mode: str | None = None) -> Decomposition:
    """``sd_i``, merged blocks, ``a_wb``, ``a_slow``, ``a_wb_raw``; one row per row of ``df``.

    ``merge_mode`` overrides the task's configured block-merge mode (``decomposition.merge_mode``).
    """
    d = _dcfg(dcfg)
    m = predictor_mask(df, value_col, set_col)
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float) if value_col in df.columns else \
        np.full(len(df), np.nan)
    blk = native_block(df, task, d)
    mode = merge_mode if merge_mode is not None else d.merge_mode(task)
    run = df["run"].to_numpy(np.int64)
    child = df[child_col].astype(str).to_numpy()
    sd_i = np.full(len(df), np.nan)
    cmean = np.full(len(df), np.nan)
    bmean = np.full(len(df), np.nan)
    bid = np.full(len(df), "", dtype=object)
    merges: list[dict] = []
    sds: dict = {}
    for c in np.unique(child):
        ci = (child == c) & m
        if ci.sum() < 2:
            continue
        mu = v[ci].mean()
        sd = v[ci].std(ddof=d.sd_ddof)
        sds[c] = float(sd)
        sel = child == c
        sd_i[sel] = sd
        cmean[sel] = mu
        for r in np.unique(run[ci]):
            ri = ci & (run == r)
            labs = np.unique(blk[ri])                      # sorted: block order (-1 first)
            counts = [int((ri & (blk == lb)).sum()) for lb in labs]
            mp, lg = merge_small_blocks(labs, counts, d.min_block_windows, mode)
            merges.extend({"child": c, "run": int(r), **e} for e in lg)
            for lb in labs:
                members = [k for k, g in mp.items() if g == mp[lb]]
                rows = (child == c) & (run == r) & np.isin(blk, members)
                bmean[rows] = v[rows & m].mean()
                bid[rows] = f"{int(r)}:" + "+".join(str(int(x)) for x in sorted(members))
    with np.errstate(invalid="ignore", divide="ignore"):
        a_wb = np.where(m, (v - bmean) / sd_i, np.nan)
        a_slow = np.where(m, (bmean - cmean) / sd_i, np.nan)
        a_raw = np.where(m, v - bmean, np.nan)
    s = suffix
    cols = pd.DataFrame({f"sd_i{s}": np.where(m, sd_i, np.nan), f"block{s}": pd.array(bid, dtype="string"),
                         f"a_wb{s}": a_wb, f"a_slow{s}": a_slow, f"a_wb_raw{s}": a_raw}, index=df.index)
    cols.loc[~m, f"block{s}"] = pd.NA
    return Decomposition(cols, merges, sds)


def decompose_sd(df: pd.DataFrame, value_col: str, set_col: str, child_col: str = CHILD, ddof: int = 1
                 ) -> np.ndarray:
    """``sd_i`` broadcast to every row (NaN for children with < 2 predictor-set windows)."""
    m = predictor_mask(df, value_col, set_col)
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float)
    out = np.full(len(df), np.nan)
    child = df[child_col].astype(str).to_numpy()
    for c in np.unique(child[m]):
        ci = (child == c) & m
        if ci.sum() >= 2:
            out[child == c] = v[ci].std(ddof=ddof)
    return out


def harmonised(df: pd.DataFrame, value_col: str = "alpha", set_col: str = "in_predictor_set", *,
               time_col: str = "onset_s", seg_col: str = "segment_id", child_col: str = CHILD,
               sd_i: np.ndarray | None = None, dcfg=None, scope: str | None = None) -> pd.Series:
    """Running-mean variant: deviation from the centred +-60-s running mean / sd_i (see the module docstring).

    ``scope`` (default: ``decomposition.harm_scope`` of ``configs/features.yaml``): ``"run"`` averages over the
    predictor-set windows of the same child and run, across joins; ``"segment"`` only within the same valid span
    ``seg_col``.
    """
    d = _dcfg(dcfg)
    half = d.harm_half_window_s
    scope = scope if scope is not None else getattr(d, "harm_scope", "run")
    if scope not in HARM_SCOPES:
        raise ValueError(f"harm scope must be one of {HARM_SCOPES}, not {scope!r}")
    keys = [child_col, "run"] + ([seg_col] if scope == "segment" else [])
    m = predictor_mask(df, value_col, set_col)
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float)
    t = df[time_col].to_numpy(float)
    rm = np.full(len(df), np.nan)
    sub = df.loc[m, keys].copy()
    sub["_i"] = np.flatnonzero(m)
    for _, g in sub.groupby(keys, sort=False):
        ii = g["_i"].to_numpy()
        ii = ii[np.argsort(t[ii], kind="stable")]
        tt, vv = t[ii], v[ii]
        cs = np.concatenate([[0.0], np.cumsum(vv)])
        lo = np.searchsorted(tt, tt - half - 1e-9, side="left")
        hi = np.searchsorted(tt, tt + half + 1e-9, side="right")
        rm[ii] = (cs[hi] - cs[lo]) / (hi - lo)
    if sd_i is None:
        sd_i = decompose_sd(df, value_col, set_col, child_col, d.sd_ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(np.where(m, (v - rm) / sd_i, np.nan), index=df.index, name="a_harm")


def history(df: pd.DataFrame, value_col: str = "alpha", set_col: str = "in_predictor_set", *,
            seg_col: str = "segment_id", child_col: str = CHILD, order_col: str = "sample",
            time_col: str = "onset_s", dcfg=None) -> pd.DataFrame:
    """``h_hist5``, ``a_hist5``, ``n_hist5`` and ``hist5_span_s`` (see module docstring)."""
    d = _dcfg(dcfg)
    n_prev, min_prev = d.hist_n_prev, d.hist_min_prev
    m = predictor_mask(df, value_col, set_col)
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float)
    order = df[order_col].to_numpy(np.int64)
    t = df[time_col].to_numpy(float)
    h = np.full(len(df), np.nan)
    nn = np.zeros(len(df), np.int64)
    span = np.full(len(df), np.nan)
    sub = df.loc[m, [child_col, "run", seg_col]].copy()
    sub["_i"] = np.flatnonzero(m)
    for _, g in sub.groupby([child_col, "run", seg_col], sort=False):
        ii = g["_i"].to_numpy()
        ii = ii[np.argsort(order[ii], kind="stable")]
        vv = v[ii]
        k = np.arange(len(ii))
        lo = np.maximum(k - n_prev, 0)
        cnt = k - lo                                          # windows strictly before k
        ok = cnt >= min_prev
        hk = np.full(len(ii), np.nan)
        for j in np.flatnonzero(ok):                          # explicit slice: no cumulative-sum carry-over
            hk[j] = vv[lo[j]:j].mean()
        h[ii] = hk
        nn[ii] = cnt
        span[ii] = np.where(ok, t[ii] - t[ii[lo]], np.nan)
    z = zscore_within_child(df, h, m & np.isfinite(h), child_col, d.sd_ddof)
    return pd.DataFrame({"h_hist5": h, "n_hist5": nn, "a_hist5": z, "hist5_span_s": span}, index=df.index)


def zscore_within_child(df: pd.DataFrame, x: pd.Series | np.ndarray, mask: np.ndarray, child_col: str = CHILD,
                        ddof: int = 1) -> np.ndarray:
    """(x - child mean) / child SD over ``mask`` rows; NaN outside ``mask``."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    child = df[child_col].astype(str).to_numpy()
    for c in np.unique(child[mask]):
        ci = (child == c) & mask
        if ci.sum() < 2:
            continue
        sd = x[ci].std(ddof=ddof)
        if sd > 0:
            out[ci] = (x[ci] - x[ci].mean()) / sd
    return out


def context_z(df: pd.DataFrame, ctx_col: str = "actx10_log", value_col: str = "alpha",
              set_col: str = "in_predictor_set", child_col: str = CHILD, ddof: int = 1) -> np.ndarray:
    """``a_ctx10`` (or ``a_ctx20``): ``ctx_col`` z-scored within child x task over predictor-set windows."""
    if ctx_col not in df.columns:
        return np.full(len(df), np.nan)
    c = pd.to_numeric(df[ctx_col], errors="coerce").to_numpy(float)
    m = predictor_mask(df, value_col, set_col) & np.isfinite(c)
    return zscore_within_child(df, c, m, child_col, ddof)


# --------------------------------------------------------------------------- table builder
def build_decomposed(df: pd.DataFrame, task: str, variants: Mapping[str, tuple[str, str]] | None = None,
                     child_col: str = CHILD, dcfg=None) -> tuple[pd.DataFrame, dict]:
    """All derived predictors for a child x task table (any number of children / runs).

    Adds, per variant ``v`` present in the table, ``sd_i``/``block``/``a_wb``/``a_slow``/``a_wb_raw`` with suffix
    ``""`` for the primary and ``__<v>`` otherwise; ``a_harm`` (running-mean variant) for every variant with suffix; ``a_hist5``
    (primary); ``a_ctx10`` / ``a_ctx20``. Returns ``(table, log)``.
    """
    d = _dcfg(dcfg)
    variants = dict(VARIANTS if variants is None else variants)
    out = df.copy()
    logd: dict = {"merges": {}, "n_predictor_set": {}}
    for name, (vc, sc) in variants.items():
        if vc not in out.columns or sc not in out.columns or not np.isfinite(
                pd.to_numeric(out[vc], errors="coerce").to_numpy(float)).any():
            continue
        sfx = "" if name == "primary" else f"__{name}"
        dd = decompose(out, vc, sc, task=task, suffix=sfx, child_col=child_col, dcfg=d)
        for c in dd.columns.columns:
            out[c] = dd.columns[c]
        out[f"a_harm{sfx}"] = harmonised(out, vc, sc, child_col=child_col, dcfg=d,
                                         sd_i=decompose_sd(out, vc, sc, child_col, d.sd_ddof)).to_numpy()
        logd["merges"][name] = dd.merges
        logd["n_predictor_set"][name] = int(predictor_mask(out, vc, sc).sum())
    if "alpha" in out.columns:
        hh = history(out, "alpha", "in_predictor_set", child_col=child_col, dcfg=d)
        for c in hh.columns:
            out[c] = hh[c].to_numpy()
        for L in d.context_lengths_s:
            tag = f"{L:g}"
            out[f"a_ctx{tag}"] = context_z(out, f"actx{tag}_log", child_col=child_col, ddof=d.sd_ddof)
    return out, logd


# --------------------------------------------------------------------------- child alpha reliability
def spearman_brown(r: float) -> float:
    """``2r / (1 + r)`` (NaN for r <= -1 or non-finite r)."""
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def split_half_r(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> tuple[float, int]:
    """Pearson r over the rows where both are finite (NaN with fewer than ``min_n`` rows)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < min_n or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan, n
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), n


def child_reliability(df: pd.DataFrame, set_col: str = "in_predictor_set", child_col: str = CHILD, dcfg=None
                      ) -> pd.DataFrame:
    """Per child: split-channel and (when present) split-time r and Spearman-Brown reliability."""
    d = _dcfg(dcfg)
    rows = []
    m = predictor_mask(df, "alpha", set_col)
    for c, g in df[m].groupby(child_col, sort=True):
        nanc = pd.Series(np.nan, index=g.index)
        r1, n1 = split_half_r(g.get("alpha_half_left", nanc).to_numpy(float),
                              g.get("alpha_half_right", nanc).to_numpy(float), d.reliability_min_windows)
        rec = {child_col: c, "n_windows": int(len(g)), "n_split_channel": n1, "r_split_channel": r1,
               "rel_split_channel": spearman_brown(r1)}
        if "alpha_time1" in g.columns:
            r2, n2 = split_half_r(g["alpha_time1"].to_numpy(float), g["alpha_time2"].to_numpy(float),
                                  d.reliability_min_windows)
            rec.update(n_split_time=n2, r_split_time=r2, rel_split_time=spearman_brown(r2))
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- desynchronisation and time-on-task checks
def time_on_task_slopes(df: pd.DataFrame, value_col: str = "alpha", set_col: str = "in_predictor_set",
                        t_col: str = "trial_index_in_run", child_col: str = CHILD) -> pd.DataFrame:
    """Per child, the OLS slope of z(value) on z(t_run), both z-scored within child x task."""
    m = predictor_mask(df, value_col, set_col)
    zv = zscore_within_child(df, pd.to_numeric(df[value_col], errors="coerce").to_numpy(float), m, child_col)
    zt = zscore_within_child(df, pd.to_numeric(df[t_col], errors="coerce").to_numpy(float), m, child_col)
    rows = []
    child = df[child_col].astype(str).to_numpy()
    for c in np.unique(child[m]):
        ci = (child == c) & m & np.isfinite(zv) & np.isfinite(zt)
        if ci.sum() < 3:
            continue
        x, y = zt[ci], zv[ci]
        slope = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else np.nan
        rows.append({child_col: c, "n": int(ci.sum()), "slope": slope})
    return pd.DataFrame(rows)


def erd_per_child(df: pd.DataFrame, set_col: str = "in_predictor_set", child_col: str = CHILD,
                  modalities: Sequence[str] = ("V", "AV")) -> pd.DataFrame:
    """Per child, mean over valid V/AV predictor-set trials of ``alpha_post_pc1 - alpha`` (log10 units)."""
    need = {"alpha_post_pc1", "alpha", "modality"}
    if not need <= set(df.columns):
        return pd.DataFrame(columns=[child_col, "n", "erd"])
    m = (predictor_mask(df, "alpha", set_col) & np.isfinite(df["alpha_post_pc1"].to_numpy(float))
         & df["modality"].isin(list(modalities)).to_numpy(bool))
    dd = df[m].assign(_erd=df.loc[m, "alpha_post_pc1"] - df.loc[m, "alpha"])
    g = dd.groupby(child_col, sort=True)["_erd"]
    return pd.DataFrame({child_col: g.mean().index, "n": g.size().to_numpy(), "erd": g.mean().to_numpy()})


def erd_minimum_frequency(freqs: np.ndarray, mean_log_post: np.ndarray, mean_log_pre: np.ndarray, iaf_hz: float,
                          fmin: float = 3.0, fmax: float = 30.0, half_width: float = 2.0) -> tuple[float, bool]:
    """Frequency of the minimum of the ERD spectrum (post - pre, log10) in [fmin, fmax] Hz and whether it lies in
    [IAF - 2, IAF + 2]."""
    f = np.asarray(freqs, float)
    e = np.asarray(mean_log_post, float) - np.asarray(mean_log_pre, float)
    k = (f >= fmin) & (f <= fmax) & np.isfinite(e)
    if not k.any():
        return np.nan, False
    fm = float(f[k][np.argmin(e[k])])
    return fm, bool(iaf_hz - half_width - 1e-9 <= fm <= iaf_hz + half_width + 1e-9)


def mean_ci(x: Sequence[float], level: float = 0.95) -> dict:
    """Mean and t-based CI of child-level values (pooled; descriptive)."""
    from scipy import stats

    a = np.asarray([v for v in x if np.isfinite(v)], float)
    n = len(a)
    if n < 2:
        return {"n": n, "mean": float(a.mean()) if n else np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    se = a.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return {"n": n, "mean": float(a.mean()), "ci_lo": float(a.mean() - q * se), "ci_hi": float(a.mean() + q * se)}
