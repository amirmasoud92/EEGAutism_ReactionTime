"""Validation of the EOG eye-closure and gaze-deviation indices against detected blinks and saccades.

Computed on the dev cohort (unaffected siblings, used to develop the processing) from ocular and alpha data only.
The criterion is a pooled AUC >= 0.80; the indices reached 0.61 (``closure_z``, blinks) and 0.65 (``gaze_dev_z``,
saccades) and were therefore not used as controls (they enter one descriptive variant only).

Per task and pooled over the tasks:

1. **Events.** Blinks = the blink detector :func:`preprocessing.ocular_indices.detect_blinks`
   (``configs/ocular.yaml`` ``blinks:``; peak rule on the 15-Hz Fp - mastoid VEOG proxy). Saccades = the HEOG
   (EXG1 - EXG2) step rule of ``configs/ocular.yaml`` ``saccades:``, detected by
   :func:`preprocessing.ocular_indices.detect_saccades` (this module has no detector of its own): ``step_like``
   (|mean(x[t, t+75 ms)) - mean(x[t-75 ms, t))| > 64 uV on the 15-Hz HEOG), and an event overlapping a blink extent
   widened by ``blink_pad_ms`` (100 ms) is dropped (``exclude_blinks``). Without a ``saccades:`` section the default
   rule applies (``SPEC_DEFAULT_SACCADES``: a change > 30 uV within 50 ms, ``step`` family, no blink exclusion).
   Saccades are detected only in recordings whose HEOG is the EXG pair (``gaze_dev_z`` is 0 under the EXG fallback,
   so it is not validated there) and, when the rule excludes blinks, only where the blink detector can run (VEOG proxy
   available): elsewhere a blink could not be told apart from a saccade, so the windows are not saccade-detectable. A
   window is saccade-detectable only if every event that could overlap it was detectable (guard = the 15-Hz filter
   half-length + the rule's reach; with blink exclusion also the blink detector's guard).
2. **Event windows.** A predictor-set window "contains" an event if the event's extent overlaps the window over which
   the index is computed (``event_window='index'``, primary: the closure / gaze window [-1.5, -0.05] s); the alpha
   window variant (``'alpha'``) is reported as well. A window counts only if detection was possible over all of it
   (inside one valid piece, at least the filter half-lengths from the piece edges).
3. **Matching.** Each event window is matched to the nearest-in-time event-free window of the same child, task and
   run (time = onset sample; ties: the earlier control), **with replacement** (primary, :data:`PRIMARY_REPLACE`):
   most index windows contain an event (blinks about 24/min, HEOG steps more often), so 1:1 matching without
   replacement would leave most event windows unmatched and the AUC would describe a selected subset. The number of
   distinct controls used is reported. Sensitivity: 1:1 without replacement, greedy by increasing time distance
   (``*_index_1to1``; unmatched event windows counted).
4. **AUC** of ``closure_z`` (blinks) and of ``gaze_dev_z`` (saccades): Mann-Whitney AUC of event vs matched control
   windows (P(event > control) + 0.5 P(tie); "higher in event windows" is the hypothesis), pooled over children, with
   a child-level percentile bootstrap CI (resampling children; ``n_boot``, fixed seed); per-child AUCs (>=
   :data:`MIN_CHILD_PAIRS` pairs) summarised by median and IQR. Pass: pooled AUC >= 0.80 (point estimate).
5. **Adjacent-window test-retest** of each index (``heog_z``, ``vfp_z``, ``closure_z``, ``gaze_dev_z``): Pearson r of
   consecutive predictor-set windows of the same run (finite values; ``gaze_dev_z`` without fallback rows), per child
   x task (>= :data:`MIN_PAIRS_RETEST` pairs), median and IQR; pooled r over all pairs; and the fraction of pairs whose
   index windows overlap in time (a reason for inflated r at short SOAs).
6. **Within-child association with alpha**: Spearman rho of each index with prestimulus alpha over the predictor set,
   per child x task (>= :data:`MIN_PAIRS_RETEST` windows), median, IQR and the fraction > 0. The alpha input is a
   DataFrame argument (keys + ``alpha_ij``, or the ``alpha`` column of :mod:`feature_extraction.prestimulus_alpha`;
   rows with ``valid`` False are dropped when that column is present).

Outputs are pooled (counts, medians, quantiles); no child identifier leaves :func:`validate_ocular_indices`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from utils.config_ocular import OcularConfig, SaccadeConfig, as_ocular_config

__all__ = [
    "saccade_rule", "saccade_rule_record", "saccade_guard_samples", "piece_bounds", "window_bounds", "recording_events",
    "match_nearest", "auc_mann_whitney", "index_auc", "adjacent_retest", "alpha_association",
    "validate_ocular_indices", "collect_validation_inputs", "alpha_input", "AUC_PASS", "MIN_CHILD_PAIRS", "MIN_PAIRS_RETEST", "SEED",
    "EVENT_COLUMNS", "ALPHA_COLUMN_CANDIDATES", "PRIMARY_REPLACE",
]

AUC_PASS = 0.80
#: Primary matching: nearest event-free window with replacement (module docstring, item 3).
PRIMARY_REPLACE = True
MIN_CHILD_PAIRS = 5
MIN_PAIRS_RETEST = 20
SEED = 20261020
EVENT_COLUMNS = ("n_blinks_index", "n_blinks_alpha", "det_valid_blink_index", "det_valid_blink_alpha",
                 "n_saccades_index", "n_saccades_alpha", "det_valid_saccade_index", "det_valid_saccade_alpha")
KEYS = ("subject", "task", "run", "sample")


# --------------------------------------------------------------------------- saccade rule
def saccade_rule(cfg: OcularConfig | Mapping | None = None) -> SaccadeConfig:
    """The HEOG step rule in force: ``configs/ocular.yaml`` ``saccades:`` or, without that section, the default
    rule (:data:`preprocessing.ocular_indices.SPEC_DEFAULT_SACCADES`)."""
    from preprocessing.ocular_indices import saccade_config

    return saccade_config(as_ocular_config(cfg))


def saccade_guard_samples(rule: SaccadeConfig, sfreq: float) -> int:
    """Samples beyond the filter guard that a window must keep from a piece edge so that every event of ``rule``
    that could overlap the window was detectable.

    ``step``: extent ``[t, t + k]`` with ``t`` detectable on ``[lo, hi - k)`` -> ``k``. ``step_like``: extent
    ``[t - h, t + h - 1]`` with ``t`` detectable on ``[lo + h, hi - h]`` -> ``2 h``.
    """
    if rule.rule == "step":
        return max(1, int(math.floor(rule.within_ms * sfreq / 1000.0 + 1e-9)))
    return 2 * max(1, int(round(rule.within_ms / 2.0 * sfreq / 1000.0)))


def saccade_rule_record(cfg: OcularConfig | Mapping | None = None) -> dict:
    """Record of the rule in force (fields of :class:`~utils.config_ocular.SaccadeConfig` + source)."""
    c = as_ocular_config(cfg)
    r = saccade_rule(c)
    src = ("configs/ocular.yaml saccades:" if c.saccades is not None
           else "default rule (no saccades: section in configs/ocular.yaml)")
    return {**asdict(r), "source": src,
            "detector": "preprocessing.ocular_indices.detect_saccades"}


# --------------------------------------------------------------------------- signal helpers
def piece_bounds(piece: np.ndarray) -> list[tuple[int, int, int]]:
    """``(piece id, start, stop)`` of each run of equal non-negative ``OC_PIECE`` values (relative indices)."""
    from preprocessing.ocular_indices import _piece_bounds

    return _piece_bounds(np.asarray(piece, dtype=float))


def window_bounds(onsets: np.ndarray, ms: tuple[float, float], orig_sfreq: float, decim: int, first_oc: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Relative ocular indices [lo, hi) of the samples whose BDF time lies in [onset + start, onset + stop)
    (the convention of the ocular indices)."""
    on = np.asarray(onsets, dtype=float)
    lo = np.ceil((on + ms[0] * orig_sfreq / 1000.0) / decim - 1e-9).astype(np.int64) - first_oc
    hi = np.ceil((on + ms[1] * orig_sfreq / 1000.0) / decim - 1e-9).astype(np.int64) - first_oc
    return lo, hi


def _count_overlaps(lo: np.ndarray, hi: np.ndarray, ev_start: np.ndarray, ev_stop: np.ndarray) -> np.ndarray:
    """Number of events [start, stop] (inclusive, fractional allowed) overlapping each window [lo, hi)."""
    if ev_start.size == 0:
        return np.zeros(lo.size, dtype=int)
    order = np.argsort(ev_start)
    s, t = ev_start[order], ev_stop[order]
    out = np.zeros(lo.size, dtype=int)
    for i in range(lo.size):
        j = np.searchsorted(s, hi[i], side="left")       # events starting before hi
        out[i] = int(np.sum(t[:j] >= lo[i]))
    return out


def _det_valid(lo: np.ndarray, hi: np.ndarray, valid: np.ndarray, piece: np.ndarray,
               bounds_by_id: Mapping[int, tuple[int, int]], guard: int) -> np.ndarray:
    n = valid.size
    out = np.zeros(lo.size, dtype=bool)
    for i in range(lo.size):
        a, b = int(lo[i]), int(hi[i])
        if b <= a or a < 0 or b > n or not valid[a:b].all():
            continue
        p = piece[a:b]
        if p.min() != p.max():
            continue
        pa, pb = bounds_by_id[int(p[0])]
        out[i] = (a - pa >= guard) and (pb - b >= guard)
    return out


def recording_events(ocular_raw: Any, onsets_samples: Sequence[int] | np.ndarray, alpha_window_ms: tuple[float, float],
                     cfg: OcularConfig | Mapping | None = None, detect_saccades_here: bool = True
                     ) -> tuple[pd.DataFrame, dict]:
    """Per-onset event counts and detection validity for one recording (columns :data:`EVENT_COLUMNS`).

    Blinks: :func:`~preprocessing.ocular_indices.detect_blinks`; saccades:
    :func:`~preprocessing.ocular_indices.detect_saccades` with the configured rule (:func:`saccade_rule`)
    and, if the rule excludes blinks, the blinks of the same copy.

    Returns
    -------
    events, info
        ``info``: pooled counts (n blinks, n saccades, n saccade candidates before the blink exclusion, valid
        minutes, fraction of candidate saccade events overlapping an unpadded blink extent) and the rule used.
    """
    from dataclasses import replace

    from preprocessing.ocular import PIECE_CH, VALID_CH, fir_half_length_s, ocular_meta
    from preprocessing.ocular_indices import SACCADE_COLUMNS, detect_blinks, detect_saccades

    c = as_ocular_config(cfg)
    rule = saccade_rule(c)
    meta = ocular_meta(ocular_raw)
    sf = float(ocular_raw.info["sfreq"])
    orig_sf, decim, first_oc = float(meta["orig_sfreq"]), int(meta["decim"]), int(ocular_raw.first_samp)
    names = ocular_raw.ch_names
    data = ocular_raw.get_data(picks=[names.index(ch) for ch in ("VEOG_lp15", rule.signal, VALID_CH, PIECE_CH)])
    v15, heog = data[0] * 1e6, data[1] * 1e6
    valid = data[2] > 0.5
    piece = np.where(valid, np.round(data[3]), -1).astype(int)
    bounds = piece_bounds(piece)
    by_id = {pid: (a, b) for pid, a, b in bounds}
    src = meta["sources"]
    has_v = bool(src.get("fp")) and src.get("ref") is not None
    has_h_exg = tuple(src.get("heog") or ()) == ("EXG1", "EXG2")
    g_fast = int(math.ceil(meta["half_length_fast_s"] * sf))
    g_blink = g_fast + int(math.ceil(fir_half_length_s(sf, *c.blinks.band_hz) * sf))
    g_sacc = g_fast + saccade_guard_samples(rule, sf)
    if rule.exclude_blinks:
        g_sacc = max(g_sacc, g_blink)          # every blink that could remove an event must have been detectable
    # saccade-detectable recording: EXG HEOG, the caller allows it, and the blink exclusion can be applied
    sacc_ok = has_h_exg and detect_saccades_here and (has_v or not rule.exclude_blinks)

    if has_v:
        blinks, binfo = detect_blinks(v15, sf, c, bounds)
    else:
        blinks, binfo = pd.DataFrame(columns=["start", "stop", "piece"]), {}
    empty = pd.DataFrame(columns=SACCADE_COLUMNS)
    if sacc_ok:
        sacc = detect_saccades(heog, sf, bounds, rule, edge_guard=g_fast, blinks=blinks)
        cand = (detect_saccades(heog, sf, bounds, replace(rule, exclude_blinks=False), edge_guard=g_fast)
                if rule.exclude_blinks else sacc)
    else:
        sacc, cand = empty, empty

    on = np.asarray(onsets_samples, dtype=float).ravel()
    out = pd.DataFrame({"sample": on.astype(np.int64)})
    for wname, ms in (("index", c.indices.closure_window_ms), ("alpha", alpha_window_ms)):
        lo, hi = window_bounds(on, (float(ms[0]), float(ms[1])), orig_sf, decim, first_oc)
        bst = blinks["start"].to_numpy(float) if len(blinks) else np.array([])
        bsp = blinks["stop"].to_numpy(float) if len(blinks) else np.array([])
        out[f"n_blinks_{wname}"] = _count_overlaps(lo, hi, bst, bsp)
        out[f"det_valid_blink_{wname}"] = _det_valid(lo, hi, valid, piece, by_id, g_blink) & has_v
        sst = sacc["start"].to_numpy(float) if len(sacc) else np.array([])
        ssp = sacc["stop"].to_numpy(float) if len(sacc) else np.array([])
        out[f"n_saccades_{wname}"] = _count_overlaps(lo, hi, sst, ssp)
        out[f"det_valid_saccade_{wname}"] = _det_valid(lo, hi, valid, piece, by_id, g_sacc) & sacc_ok
    # diagnostic: candidate saccade events overlapping an (unpadded) blink extent (blink leakage into the HEOG)
    if len(cand) and len(blinks):
        bst, bsp = blinks["start"].to_numpy(float), blinks["stop"].to_numpy(float)
        ov = np.array([np.any((bsp >= s) & (bst <= t)) for s, t in zip(cand["start"], cand["stop"])])
        frac_ov = float(ov.mean())
    else:
        frac_ov = float("nan")
    minutes = valid.sum() / sf / 60.0
    info = {"n_blinks": int(len(blinks)), "n_saccades": int(len(sacc)), "n_saccade_candidates": int(len(cand)),
            "valid_minutes": float(minutes), "has_veog": has_v, "heog_is_exg": has_h_exg,
            "saccades_detectable": bool(sacc_ok), "frac_saccades_overlapping_blink": frac_ov,
            "blink_threshold_uv": binfo.get("threshold_uv"), "saccade_rule": saccade_rule_record(c),
            "saccade_guard_samples": int(g_sacc)}
    return out, info


# --------------------------------------------------------------------------- matching and AUC
def match_nearest(t_event: np.ndarray, t_control: np.ndarray, replace: bool = PRIMARY_REPLACE
                  ) -> list[tuple[int, int]]:
    """Nearest-in-time matching of event to control windows.

    ``replace=True`` (primary): every event window gets its nearest control (ties: the earlier control); controls
    may repeat. ``replace=False`` (sensitivity): 1:1 without replacement, greedy by |dt| (ties by event, then
    control index), so event windows beyond the number of controls stay unmatched.

    Returns ``(event index, control index)`` pairs, sorted.
    """
    te, tc = np.asarray(t_event, float), np.asarray(t_control, float)
    if te.size == 0 or tc.size == 0:
        return []
    if replace:
        return [(i, int(np.argmin(np.abs(tc - t)))) for i, t in enumerate(te)]
    d = np.abs(te[:, None] - tc[None, :])
    ei, ci = np.meshgrid(np.arange(te.size), np.arange(tc.size), indexing="ij")
    order = np.lexsort((ci.ravel(), ei.ravel(), d.ravel()))
    used_e = np.zeros(te.size, bool)
    used_c = np.zeros(tc.size, bool)
    pairs = []
    for o in order:
        e, c = int(ei.ravel()[o]), int(ci.ravel()[o])
        if used_e[e] or used_c[c]:
            continue
        used_e[e] = used_c[c] = True
        pairs.append((e, c))
        if used_e.all() or used_c.all():
            break
    return sorted(pairs)


def auc_mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(pos == neg) (NaN if either set is empty)."""
    from scipy.stats import rankdata

    p, n = np.asarray(pos, float), np.asarray(neg, float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([p, n]))
    return float((r[: p.size].sum() - p.size * (p.size + 1) / 2.0) / (p.size * n.size))


def _matched_pairs(df: pd.DataFrame, index: str, n_col: str, valid_col: str, replace: bool = PRIMARY_REPLACE
                   ) -> pd.DataFrame:
    """Matched (event, control) pairs per child x task x run; columns subject, task, x_event, x_control, dt,
    control_key. ``attrs``: n_unmatched, n_event_windows, n_control_windows."""
    use = df[df["in_predictor_set"].astype(bool) & df[valid_col].astype(bool) & np.isfinite(df[index].astype(float))]
    rows = []
    n_unmatched = n_ev = n_ct = 0
    for (sub, task, run), g in use.groupby(["subject", "task", "run"], sort=True):
        g = g.sort_values("sample", kind="stable")         # "earlier control" tie rule needs time order
        ev = g[g[n_col] >= 1]
        ct = g[g[n_col] == 0]
        n_ev += len(ev)
        n_ct += len(ct)
        pairs = match_nearest(ev["sample"].to_numpy(float), ct["sample"].to_numpy(float), replace=replace)
        n_unmatched += len(ev) - len(pairs)
        xe, xc = ev[index].to_numpy(float), ct[index].to_numpy(float)
        se, sc = ev["sample"].to_numpy(float), ct["sample"].to_numpy(float)
        for e, c in pairs:
            rows.append((sub, task, xe[e], xc[c], abs(se[e] - sc[c]), f"{sub}|{task}|{run}|{int(sc[c])}"))
    out = pd.DataFrame(rows, columns=["subject", "task", "x_event", "x_control", "dt_samples", "control_key"])
    out.attrs.update({"n_unmatched": n_unmatched, "n_event_windows": n_ev, "n_control_windows": n_ct})
    return out


def _q(x: Iterable[float]) -> dict:
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    if a.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {"n": int(a.size), "median": float(np.median(a)), "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)), "min": float(a.min()), "max": float(a.max())}


def index_auc(df: pd.DataFrame, index: str, n_col: str, valid_col: str, *, n_boot: int = 2000, seed: int = SEED,
              orig_sfreq: float = 512.0, replace: bool = PRIMARY_REPLACE) -> dict:
    """Pooled AUC (matched windows), child-bootstrap CI and per-child AUC summary for one index / event type."""
    pairs = _matched_pairs(df, index, n_col, valid_col, replace=replace)
    res: dict[str, Any] = {"index": index, "event_count_column": n_col, "matching": "nearest, with replacement"
                           if replace else "nearest, 1:1 without replacement", "n_pairs": int(len(pairs)),
                           "n_event_windows": int(pairs.attrs.get("n_event_windows", 0)),
                           "n_control_windows": int(pairs.attrs.get("n_control_windows", 0)),
                           "n_distinct_controls_used": int(pairs["control_key"].nunique()) if len(pairs) else 0,
                           "n_event_windows_unmatched": int(pairs.attrs.get("n_unmatched", 0)),
                           "n_children": int(pairs["subject"].nunique()) if len(pairs) else 0}
    if not len(pairs):
        res.update({"auc_pooled": None, "ci95": None, "pass": None, "per_child_auc": _q([])})
        return res
    auc = auc_mann_whitney(pairs["x_event"].to_numpy(), pairs["x_control"].to_numpy())
    res["auc_pooled"] = auc
    res["paired_concordance"] = float(np.mean((pairs["x_event"] > pairs["x_control"])
                                              + 0.5 * (pairs["x_event"] == pairs["x_control"])))
    res["median_dt_s"] = float(np.median(pairs["dt_samples"]) / orig_sfreq)
    by = {s: g for s, g in pairs.groupby("subject", sort=True)}
    per_child = [auc_mann_whitney(g["x_event"].to_numpy(), g["x_control"].to_numpy())
                 for g in by.values() if len(g) >= MIN_CHILD_PAIRS]
    res["per_child_auc"] = _q(per_child)
    res["per_child_min_pairs"] = MIN_CHILD_PAIRS
    keys = list(by)
    rng = np.random.default_rng(seed)
    boots = []
    if len(keys) >= 2 and n_boot > 0:
        xe = {k: by[k]["x_event"].to_numpy() for k in keys}
        xc = {k: by[k]["x_control"].to_numpy() for k in keys}
        for _ in range(n_boot):
            pick = rng.integers(0, len(keys), len(keys))
            boots.append(auc_mann_whitney(np.concatenate([xe[keys[i]] for i in pick]),
                                          np.concatenate([xc[keys[i]] for i in pick])))
        res["ci95"] = [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]
    else:
        res["ci95"] = None
    res["n_boot"] = n_boot
    res["pass"] = bool(auc >= AUC_PASS)
    return res


# --------------------------------------------------------------------------- test-retest and alpha association
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def adjacent_retest(df: pd.DataFrame, index: str, window_len_samples: float | None = None,
                    min_pairs: int = MIN_PAIRS_RETEST) -> dict:
    """Adjacent-window test-retest of ``index`` (see the module docstring)."""
    use = df[df["in_predictor_set"].astype(bool) & np.isfinite(df[index].astype(float))]
    if index == "gaze_dev_z" and "gaze_dev_fallback" in use.columns:
        use = use[~use["gaze_dev_fallback"].astype(bool)]
    per, allx, ally, overl = [], [], [], []
    for (sub, task), g in use.groupby(["subject", "task"], sort=True):
        xs, ys = [], []
        for _, gr in g.groupby("run", sort=True):
            gr = gr.sort_values("sample")
            v = gr[index].to_numpy(float)
            s = gr["sample"].to_numpy(float)
            if v.size < 2:
                continue
            xs.append(v[:-1])
            ys.append(v[1:])
            if window_len_samples is not None:
                overl.append(np.diff(s) < window_len_samples)
        if not xs:
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        allx.append(x)
        ally.append(y)
        if x.size >= min_pairs:
            per.append(_pearson(x, y))
    X = np.concatenate(allx) if allx else np.array([])
    Y = np.concatenate(ally) if ally else np.array([])
    return {"index": index, "per_child_task_r": _q(per), "pooled_r": _pearson(X, Y) if X.size else None,
            "n_pairs": int(X.size), "min_pairs": min_pairs,
            "frac_pairs_windows_overlap": float(np.concatenate(overl).mean()) if overl else None}


#: Alpha columns accepted by :func:`alpha_association` (in this order).
ALPHA_COLUMN_CANDIDATES = ("alpha_ij", "alpha")


def alpha_input(alpha: pd.DataFrame, alpha_col: str | None = None) -> pd.DataFrame:
    """Keys + ``alpha_ij`` from an alpha predictor table (``alpha_col`` or the first of
    :data:`ALPHA_COLUMN_CANDIDATES`); rows with ``valid`` False become NaN."""
    col = alpha_col or next((c for c in ALPHA_COLUMN_CANDIDATES if c in alpha.columns), None)
    if col is None or col not in alpha.columns:
        raise KeyError(f"alpha table needs one of {ALPHA_COLUMN_CANDIDATES} (or alpha_col)")
    out = alpha[list(KEYS)].copy()
    a = pd.to_numeric(alpha[col], errors="coerce").astype(float)
    if "valid" in alpha.columns:
        a = a.where(alpha["valid"].astype("boolean").fillna(False).astype(bool).to_numpy())
    out["alpha_ij"] = a.to_numpy()
    return out


def alpha_association(df: pd.DataFrame, alpha: pd.DataFrame, index: str, min_rows: int = MIN_PAIRS_RETEST,
                      alpha_col: str | None = None) -> dict:
    """Within-child Spearman rho of ``index`` with prestimulus alpha over the predictor set."""
    from scipy.stats import spearmanr

    a = alpha_input(alpha, alpha_col)
    m = df.merge(a, on=list(KEYS), how="inner", validate="one_to_one")
    m = m[m["in_predictor_set"].astype(bool) & np.isfinite(m[index].astype(float))
          & np.isfinite(m["alpha_ij"].astype(float))]
    if index == "gaze_dev_z" and "gaze_dev_fallback" in m.columns:
        m = m[~m["gaze_dev_fallback"].astype(bool)]
    rhos = []
    for _, g in m.groupby(["subject", "task"], sort=True):
        if len(g) >= min_rows:
            r = spearmanr(g[index].to_numpy(float), g["alpha_ij"].to_numpy(float)).statistic
            rhos.append(float(r))
    a = np.asarray([r for r in rhos if np.isfinite(r)])
    return {"index": index, "per_child_task_rho": _q(rhos), "frac_positive": float((a > 0).mean()) if a.size else None,
            "n_rows": int(len(m)), "min_rows": min_rows}


# --------------------------------------------------------------------------- driver
def validate_ocular_indices(features: pd.DataFrame, events: pd.DataFrame, *, alpha: pd.DataFrame | None = None,
                            n_boot: int = 2000, ocular_cfg: OcularConfig | Mapping | None = None,
                            alpha_windows_ms: Mapping[str, Sequence[float]] | None = None,
                            orig_sfreq: float = 512.0, alpha_col: str | None = None) -> dict:
    """AUCs, test-retest and alpha association of the ocular indices (pooled; see the module docstring).

    Parameters
    ----------
    features
        :func:`feature_extraction.ocular_trial.assemble_ocular_features` output (one or more tasks).
    events
        Keys + :data:`EVENT_COLUMNS` (from :func:`recording_events`), aligned one-to-one on the keys.
    alpha
        Optional alpha predictor table: keys + ``alpha_ij`` (or ``alpha``; see :func:`alpha_input`). Without it the
        alpha association is reported as not run.
    """
    alpha_tab = None
    if alpha is not None:
        alpha_tab = alpha_input(alpha, alpha_col)
    c = as_ocular_config(ocular_cfg)
    df = features.merge(events[[*KEYS, *EVENT_COLUMNS]], on=list(KEYS), how="left", validate="one_to_one")
    if df[list(EVENT_COLUMNS)].isna().any().any():
        raise ValueError("events do not cover every feature row")
    df = df.copy()
    sac_ok = ~df["gaze_dev_fallback"].astype(bool)
    for w in ("index", "alpha"):
        df[f"det_valid_saccade_{w}"] = df[f"det_valid_saccade_{w}"].astype(bool) & sac_ok
    cl = c.indices.closure_window_ms
    idx_len = (cl[1] - cl[0]) * orig_sfreq / 1000.0
    out: dict[str, Any] = {"criterion_auc": AUC_PASS, "seed": SEED,
                           "n_boot": n_boot, "event_window_primary": "index",
                           "matching_primary": "nearest, with replacement" if PRIMARY_REPLACE else "1:1",
                           "closure_window_ms": list(cl), "saccade_rule": saccade_rule_record(c),
                           "blink_config": asdict(c.blinks), "tasks": {}}
    for task in [*sorted(df["task"].unique()), "ALL"]:
        d = df if task == "ALL" else df[df["task"] == task]
        r: dict[str, Any] = {"n_children": int(d["subject"].nunique()), "n_rows": int(len(d)),
                             "n_predictor_set": int(d["in_predictor_set"].sum()),
                             "frac_predictor_with_blink_index": float(
                                 (d.loc[d["in_predictor_set"], "n_blinks_index"] >= 1).mean())
                             if d["in_predictor_set"].any() else None}
        for w in ("index", "alpha"):
            r[f"closure_z_blinks_{w}"] = index_auc(d, "closure_z", f"n_blinks_{w}", f"det_valid_blink_{w}",
                                                   n_boot=n_boot if w == "index" else 0, orig_sfreq=orig_sfreq)
            r[f"gaze_dev_z_saccades_{w}"] = index_auc(d, "gaze_dev_z", f"n_saccades_{w}", f"det_valid_saccade_{w}",
                                                      n_boot=n_boot if w == "index" else 0, orig_sfreq=orig_sfreq)
        # sensitivity: 1:1 matching without replacement
        r["closure_z_blinks_index_1to1"] = index_auc(d, "closure_z", "n_blinks_index", "det_valid_blink_index",
                                                     n_boot=0, orig_sfreq=orig_sfreq, replace=not PRIMARY_REPLACE)
        r["gaze_dev_z_saccades_index_1to1"] = index_auc(d, "gaze_dev_z", "n_saccades_index",
                                                        "det_valid_saccade_index", n_boot=0,
                                                        orig_sfreq=orig_sfreq, replace=not PRIMARY_REPLACE)
        aw = None
        if alpha_windows_ms is not None and task in alpha_windows_ms:
            a = alpha_windows_ms[task]
            aw = (float(a[1]) - float(a[0])) * orig_sfreq / 1000.0
        r["retest"] = {z: adjacent_retest(d, z, idx_len if z in ("closure_z", "gaze_dev_z") else aw)
                       for z in ("heog_z", "vfp_z", "closure_z", "gaze_dev_z")}
        if alpha_tab is not None:
            r["alpha_association"] = {z: alpha_association(d, alpha_tab, z)
                                      for z in ("heog_z", "vfp_z", "closure_z", "gaze_dev_z")}
        else:
            r["alpha_association"] = "not run (no alpha table given)"
        out["tasks"][task] = r
    allr = out["tasks"]["ALL"]
    out["verdict"] = {
        "closure_z": {"auc_pooled": allr["closure_z_blinks_index"].get("auc_pooled"),
                      "pass": allr["closure_z_blinks_index"].get("pass")},
        "gaze_dev_z": {"auc_pooled": allr["gaze_dev_z_saccades_index"].get("auc_pooled"),
                       "pass": allr["gaze_dev_z_saccades_index"].get("pass")},
    }
    return out


def collect_validation_inputs(task: str, subjects: Iterable[str], *, events_cfg: Any,
                              trials_path: str | os.PathLike[str], preproc_root: str | os.PathLike[str],
                              cfg: OcularConfig | Mapping | None = None, window: str = "primary", strict: bool = False,
                              extra_valid: pd.DataFrame | None = None, valid_col: str | None = None
                              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Features (:mod:`feature_extraction.ocular_trial`) and event counts for the validation, reading each ocular
    copy once.

    ``extra_valid`` (keys + ``window_valid`` / the alpha table's ``valid``) restricts the predictor set to windows
    that also pass the EEG window rules, as in the feature tables.

    Returns
    -------
    features, events, info
        ``info``: pooled counts (recordings, skipped, blinks, saccades, blink-saccade overlap); no IDs.
    """
    from feature_extraction.ocular_trial import (
        KEY_COLUMNS,
        _extra_valid_mask,
        assemble_ocular_features,
        exg_status_from_qc,
        iter_recordings,
        load_predictor_trials,
        missingness_summary,
        recording_ocular_indices,
        window_spec,
    )

    c = as_ocular_config(cfg)
    tc = events_cfg.task(task)
    window_ms, _ = window_spec(tc, window, c if window == "primary" else None)
    trials = load_predictor_trials(task, events_cfg, trials_path, subjects=subjects, window=window)
    feats, evs, infos = [], [], []
    counter: dict = {}
    for _, _, g, oc, qc in iter_recordings(trials, task, preproc_root, strict=strict, counter=counter):
        feats.append(recording_ocular_indices(oc, g, window_ms, qc=qc, cfg=c))
        ev, info = recording_events(oc, g["sample"].to_numpy(), window_ms, c,
                                    detect_saccades_here=exg_status_from_qc(qc).heog_exg_ok)
        for k in KEY_COLUMNS:
            ev[k] = g[k].to_numpy()
        evs.append(ev)
        infos.append(info)
    if not feats:
        raise FileNotFoundError(f"no {task} recordings with ocular copies for the given subjects")
    rows = pd.concat(feats, ignore_index=True)
    ev_all = pd.concat(evs, ignore_index=True)
    features = assemble_ocular_features(rows, extra_valid=_extra_valid_mask(rows, extra_valid, valid_col))
    minutes = sum(i["valid_minutes"] for i in infos)
    ov = [i["frac_saccades_overlapping_blink"] for i in infos if np.isfinite(i["frac_saccades_overlapping_blink"])]
    nb, ns = sum(i["n_blinks"] for i in infos), sum(i["n_saccades"] for i in infos)
    nc = sum(i["n_saccade_candidates"] for i in infos)
    min_s = sum(i["valid_minutes"] for i in infos if i["saccades_detectable"])
    pooled = {"task": task, "n_recordings": len(feats), "n_skipped": counter.get("n_skipped", 0),
              "n_blinks": int(nb), "n_saccades": int(ns), "n_saccade_candidates": int(nc),
              "frac_saccade_candidates_removed_as_blink": (1.0 - ns / nc) if nc else None,
              "blinks_per_min": (nb / minutes) if minutes else None,
              "saccades_per_min": (ns / min_s) if min_s else None,
              "n_recordings_heog_exg": int(sum(i["heog_is_exg"] for i in infos)),
              "n_recordings_saccades_detectable": int(sum(i["saccades_detectable"] for i in infos)),
              "median_frac_saccades_overlapping_blink": float(np.median(ov)) if ov else None,
              "saccade_rule": infos[0]["saccade_rule"], **missingness_summary(features)}
    return features, ev_all, pooled


