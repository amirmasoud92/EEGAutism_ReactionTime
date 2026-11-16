"""Per-recording prestimulus alpha table.

One row per trial-table row that belongs to any of the task's stimulus-defined predictor sets
(``configs/events.yaml``: ``predictor_set`` and the ``*_predictor_set`` sets of the window variants), with one boolean
``in_<set>`` column per set. The predictor sets carry the trial-level window rules that need only the event markers
(no stimulus, press or block marker within 700 ms (FAST, IC) or 1000 ms (AVSRT) before onset; built in
:mod:`trials`). The trial table is read with a fixed list of columns (keys, stimulus codes, block structure and the
subset flags); no response column is read.

Inputs per recording: the cleaned continuous data (:func:`preprocessing.clean.load_clean` with ``interpolate=False``,
so bad ROI channels stay marked and are dropped from the ROI average; the ICA-corrected copy for everything except
the no-ICA variant, which uses ``apply_ica=False``), the preprocessing :class:`~preprocessing.segments.SegmentInfo`,
the events.tsv stimulus onsets (evoked-response variant only) and the child's IAF (:mod:`feature_extraction.iaf`).
Parameters: ``configs/features.yaml``.

Columns (feature values are NaN where their window is invalid; units log10(uV^2/Hz); see
:mod:`feature_extraction.spectral` for the validity rules and estimators):

=================================  ========================================================================
``alpha``                          primary: DPSS single taper, IAF +- 2 Hz, 8-channel ROI, task window
``valid``, ``invalid_reason``      primary window validity (edge flag, segment, BAD annotation, >= 5 ROI
                                   channels, ROI peak-to-peak <= 200 uV); ``n_roi``, ``ptp_uv``,
                                   ``win_start``, ``win_stop``, ``segment_id``
``alpha_s_w500`` / ``_s_w1000``    AVSRT 500-ms / 1000-ms windows, ``alpha_s_w400`` (FAST, IC 400-ms window),
                                   each with its own ``valid_<w>``, ``invalid_reason_<w>``, ``ptp_uv_<w>``
``alpha_s_est1``, ``_s_est2``      Welch; 4-Hz multitaper; primary window and validity
``alpha_s_band``, ``alpha_e9``     fixed 8-12 Hz, 7-13 Hz
``ap_2p5_5``, ``ap_15_25``         log10 power 2.5-5 Hz and 15-25 Hz (same estimator, window and ROI)
``alpha_s_roi``                    the data owners' 7-channel ROI, own validity ``valid_s_roi``
``alpha_e7_mu`` (``_left``,        centro-parietal mu ROI and its left (C3, CP3) / right (C4, CP4) pairs,
``_right``)                        own validity
``alpha_half_left`` / ``_right``   split-channel hemisphere halves, primary validity + >= 3 channels
``alpha_time1`` / ``_time2``       split-time halves of the AVSRT 800-ms window (first / last 204 samples)
``alpha_s_iaf``                    IAF +- 2 Hz around the main-day-rest IAF, when that IAF is given
``alpha_s_ica``                    no ICA correction; own artefact rule on the uncorrected data
``alpha_s_erp``                    FAST, IC: previous evoked response removed (:func:`erp_corrected_windows`);
                                   ``erp_lag_ms``, ``erp_n_template_min``, ``erp_frac_corrected``
``alpha_post_pc1``                 AVSRT: IAF +- 2 Hz power in [200, 600) ms after onset, own validity
``actx10_log`` (``actx20_log``)    log10 mean IAF +- 2 Hz Hilbert power of the ROI signal over the 10 s (20 s)
                                   before the window start, same valid span, >= 50% valid samples, filtered and
                                   Hilbert-transformed on data before the current onset only
                                   (:func:`~feature_extraction.spectral.context_band_power`); ``actx10_frac``
                                   (z-scored in decomposition)
=================================  ========================================================================

The within-child decomposition (``a_wb``, ``a_slow``), ``a_hist5``, the running-mean variant and ``a_ctx10`` are
computed on the table by :mod:`feature_extraction.decomposition`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from feature_extraction import spectral as sp

log = logging.getLogger(__name__)

#: Columns read from the trial table besides the subset flags (stimulus and block structure only).
KEY_COLUMNS: tuple[str, ...] = ("subject", "task", "run", "sample", "onset_s", "trial_key", "trial_index_in_run",
                                "value", "condition", "is_target")
STRUCT_COLUMNS: dict[str, tuple[str, ...]] = {
    "AVSRT": ("block_index_in_run", "n_joins_before", "modality"),
    "FAST": ("block_index_in_run", "n_joins_before", "block_status", "half_status", "category"),
    "IC": ("block_index_in_run", "n_joins_before", "seq180_index"),
}


def _features_cfg(fcfg):
    """``FeaturesConfig`` (loaded from ``configs/features.yaml`` when None)."""
    if fcfg is None:
        from utils.config_features import load_features_config

        fcfg = load_features_config()
    return fcfg


# --------------------------------------------------------------------------- trial rows
def predictor_subsets(task_cfg) -> dict[str, str]:
    """Outcome-independent ``*predictor_set`` subsets of the task and the prestimulus window each one uses.

    ``predictor_set`` and the other ``*_predictor_set`` sets use the ``primary`` window unless the subset's name
    starts with a configured window name (``s_w500_predictor_set`` -> ``s_w500``).
    """
    out: dict[str, str] = {}
    wins = set(task_cfg.prestim_windows)
    for name, sd in task_cfg.analysis_subsets.items():
        if not name.endswith("predictor_set") or not sd.get("outcome_independent", False):
            continue
        w = "primary"
        for wn in wins:
            if wn != "primary" and name.startswith(wn + "_"):
                w = wn
        out[name] = w
    if "predictor_set" not in out:
        raise ValueError(f"{task_cfg.name}: no outcome-independent 'predictor_set' configured")
    return out


def predictor_columns(task_cfg) -> list[str]:
    """Trial-table columns read for the alpha table of ``task_cfg``."""
    cols = list(KEY_COLUMNS) + list(STRUCT_COLUMNS.get(task_cfg.name, ()))
    for name in predictor_subsets(task_cfg):
        sd = task_cfg.analysis_subsets[name]
        for c in list(sd["require_true"]) + list(sd["require_false"]):
            if c not in cols:
                cols.append(c)
    for wn in task_cfg.prestim_windows:
        c = f"edge_valid_{wn}"
        if c not in cols:
            cols.append(c)
    return cols


def select_predictor_rows(trials: pd.DataFrame, task_cfg) -> pd.DataFrame:
    """Rows of ``trials`` in any predictor set, with ``in_<set>`` flags; only the listed columns are kept."""
    from trials.schema import apply_subset

    cols = [c for c in predictor_columns(task_cfg) if c in trials.columns]
    t = trials[cols].copy()
    sets = predictor_subsets(task_cfg)
    any_set = pd.Series(False, index=t.index)
    for name in sets:
        m = apply_subset(t, task_cfg.analysis_subsets[name]).astype(bool)
        t[f"in_{name}"] = m.to_numpy()
        any_set |= m
    return t[any_set.to_numpy()].sort_values("sample", kind="stable").reset_index(drop=True)


def load_predictor_rows(trials_path: str | os.PathLike[str], task_cfg, subject: str, run: int) -> pd.DataFrame:
    """Read the listed columns of one recording's rows from the combined ``<task>_trials.parquet``."""
    cols = predictor_columns(task_cfg)
    df = pd.read_parquet(trials_path, columns=cols, filters=[("subject", "==", subject), ("run", "==", int(run))])
    return select_predictor_rows(df, task_cfg)


def recording_stimulus_samples(events_path: str | os.PathLike[str], task_cfg) -> np.ndarray:
    """Sorted stimulus onsets (samples) of the task's ``stimulus_codes`` from events.tsv (no press is read)."""
    from trials.io import read_events_tsv

    ev = read_events_tsv(events_path)
    return np.sort(ev.loc[ev["value"].isin(list(task_cfg.stimulus_codes)), "sample"].to_numpy(np.int64))


# --------------------------------------------------------------------------- previous evoked response removed
def erp_leave_out_stop(onsets: np.ndarray, event_samples: np.ndarray, rt_hi_samples: int) -> np.ndarray:
    """End of each trial's own post-onset period: ``max(onset + rt_hi, first stimulus after onset)``.

    With the window start, ``[win_start, stop)`` is the span of samples that may never enter the trial's own evoked-response
    template (its window, its response window and everything until the next stimulus).
    """
    on = np.asarray(onsets, np.int64)
    ev = np.unique(np.asarray(event_samples, np.int64))
    k = np.searchsorted(ev, on, side="right")
    nxt = np.where(k < len(ev), ev[np.clip(k, 0, max(len(ev) - 1, 0))] if len(ev) else on, on)
    return np.maximum(on + int(rt_hi_samples), nxt)


def erp_corrected_windows(x: np.ndarray, starts: np.ndarray, n: int, onsets: np.ndarray, event_samples: np.ndarray,
                          seg, sfreq: float, *, epoch_s: float, leave_out_stop: np.ndarray, min_epochs: int = 10,
                          bad_spans: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Windows with the child x run average evoked response to the previous stimulus subtracted.

    Definition (``configs/features.yaml`` ``alpha.s_erp``):

    - **Events** ``s_j``: every stimulus onset of the run (task ``stimulus_codes``: FAST every image, IC contours and
      flips), one pooled template whatever the stimulus type ("the" child x run average evoked response).
    - **Epoch support** of event j: lags ``tau = 0 .. len_j - 1`` with ``len_j = min(T, s_{j+1} - s_j, end of the
      valid span containing s_j - s_j, first BAD annotation after s_j - s_j)``, ``T = epoch_s x sfreq`` (2 s). The
      epoch stops at the next stimulus because the window to be corrected always lies before the next stimulus
      (the current onset), so the template is the response to a stimulus not yet followed by another one. No
      baseline (the data are 1-40 Hz band-passed).
    - **Leave-out (other trials' data only)**: for trial i, sample ``t`` of any epoch is excluded from the template
      when ``t`` lies in ``[win_start_i, leave_out_stop_i)`` (the window itself and the trial's own post-onset
      period, :func:`erp_leave_out_stop`). The template is summed directly over the remaining samples (no
      subtract-from-total), so data in that interval cannot change trial i's corrected window at all.
    - **Correction**: ``p_i`` = the last stimulus strictly before ``onset_i``; it must lie in the valid span of the
      window start. Window sample t (lag ``tau = t - p_i``) becomes ``x(t) - E_i(tau)`` where ``E_i(tau)`` is the mean
      over the epochs that support ``tau`` and survive the leave-out, if at least ``min_epochs`` do; samples with
      ``tau < 0`` or ``tau >= T`` or too few epochs are left unchanged.

    Returns
    -------
    windows : ``(n_windows, n_channels, n)`` corrected data
    lag_ms : window-start lag after ``p_i`` (ms; NaN without any correction)
    n_template_min : smallest number of epochs behind a corrected sample (0 without correction)
    frac_corrected : fraction of the window's samples that were corrected
    """
    x = np.asarray(x, float)
    st = np.asarray(starts, np.int64)
    on = np.asarray(onsets, np.int64)
    stop = np.asarray(leave_out_stop, np.int64)
    T = int(round(epoch_s * sfreq))
    ev = np.unique(np.asarray(event_samples, np.int64))
    out = sp.extract_windows(x, st, n)
    lag_ms = np.full(len(st), np.nan)
    n_min = np.zeros(len(st), np.int64)
    frac = np.zeros(len(st))
    if len(ev) == 0 or len(st) == 0:
        return out, lag_ms, n_min, frac
    # epoch support
    k_ev = seg.segment_of(ev)
    span_end = np.where(k_ev >= 0, seg.valid_spans[np.clip(k_ev, 0, None), 1] if len(seg.valid_spans) else ev, ev)
    nxt = np.r_[ev[1:], np.iinfo(np.int64).max]
    end = np.minimum.reduce([ev + T, nxt, span_end])
    if bad_spans is not None and len(bad_spans):
        bs = np.asarray(bad_spans, np.int64).reshape(-1, 2)
        for s0, s1 in bs:
            inside = (ev >= s0) & (ev < s1)
            end[inside] = ev[inside]
            after = (ev < s0) & (end > s0)
            end[after] = s0
    supp = np.where(k_ev >= 0, np.maximum(end - ev, 0), 0)          # len_j
    use = supp > 0
    ev_u, supp_u = ev[use], supp[use]
    seg_st = seg.segment_of(st)
    for i in range(len(st)):
        kk = np.searchsorted(ev, on[i], side="left") - 1
        if kk < 0 or seg_st[i] < 0:
            continue
        p = int(ev[kk])
        if seg.segment_of(p)[0] != seg_st[i]:
            continue
        lags = st[i] - p + np.arange(n)
        inl = (lags >= 0) & (lags < T)
        if not inl.any() or len(ev_u) == 0:
            continue
        li = lags[inl]
        t_abs = ev_u[:, None] + li[None, :]                          # (n_ev, n_l)
        M = (li[None, :] < supp_u[:, None]) & ~((t_abs >= st[i]) & (t_abs < stop[i]))
        cnt = M.sum(axis=0)
        okl = cnt >= int(min_epochs)
        if not okl.any():
            continue
        idx = np.where(M, t_abs, 0)
        G = np.where(M[None, :, :], x[:, idx], 0.0)                  # excluded samples are exact zeros
        E = G.sum(axis=1) / np.maximum(cnt, 1)[None, :]
        cols = np.flatnonzero(inl)[okl]
        out[i][:, cols] -= E[:, okl]
        lag_ms[i] = (st[i] - p) * 1000.0 / sfreq
        n_min[i] = int(cnt[okl].min())
        frac[i] = len(cols) / float(n)
    return out, lag_ms, n_min, frac


# --------------------------------------------------------------------------- main computation
@dataclass
class AlphaResult:
    """Predictor table, mean spectra and metadata of one recording."""

    table: pd.DataFrame
    spectra: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _picks(raw, names: Sequence[str]) -> list[str]:
    have = set(raw.ch_names)
    return [c for c in names if c in have]


def compute_alpha_features(raw, trials: pd.DataFrame, task_cfg, iaf_hz: float, *, seg, fcfg=None, raw_noica=None,
                           stim_samples: np.ndarray | None = None, iaf_hz_s_iaf: float | None = None
                           ) -> AlphaResult:
    """Compute the alpha predictor columns for the rows of ``trials`` (see the module docstring).

    Parameters
    ----------
    raw
        Cleaned continuous data (ICA-corrected), bad channels in ``info['bads']`` (not interpolated), volts.
    trials
        Output of :func:`select_predictor_rows` (needs ``sample`` and ``edge_valid_<window>``).
    task_cfg
        :class:`~utils.config.TaskEventsConfig` (windows, RT window end for the evoked-response leave-out).
    iaf_hz
        The child's IAF.
    seg
        The recording's :class:`~preprocessing.segments.SegmentInfo`.
    fcfg
        :class:`~utils.config_features.FeaturesConfig` (default: ``configs/features.yaml``).
    raw_noica
        The same data without ICA (no-ICA variant); skipped when ``None``.
    stim_samples
        events.tsv stimulus onsets of the run (evoked-response template); that variant is skipped when ``None``.
    iaf_hz_s_iaf
        Main-day-rest IAF; ``alpha_s_iaf`` is NaN when ``None``.
    """
    fcfg = _features_cfg(fcfg)
    A = fcfg.alpha
    task = task_cfg.name
    sf = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    if seg.n_times != n_times:
        raise ValueError(f"segment info ({seg.n_times}) and raw ({n_times}) differ in length")
    rois = sp.rois_from_config(A)
    names = _picks(raw, sp.all_roi_channels(rois.values()))
    X = raw.get_data(picks=names) * A.data_scale if names else np.zeros((0, n_times))
    bads = list(raw.info["bads"])
    bad_sp = sp.bad_annotation_spans(raw, A.bad_annotation_prefix)
    margin = int(round(A.edge_margin_s * sf))
    thr = A.artefact_ptp_uv * A.data_scale / 1e6                       # threshold in data units
    tol = A.band_tol_hz
    band = sp.iaf_band(iaf_hz, A.iaf_half_width_hz)
    onsets = trials["sample"].to_numpy(np.int64)
    n_rows = len(onsets)
    out = trials.copy()
    out["iaf_hz"] = float(iaf_hz)
    out["band_lo_hz"], out["band_hi_hz"] = band
    avail = {r: sp.available_channels(spec, names, bads) for r, spec in rois.items()}
    roi_idx = {r: [names.index(c) for c in ch] for r, ch in avail.items()}
    prim = rois["primary"]
    meta: dict = {"task": task, "sfreq": sf, "n_rows": n_rows, "iaf_hz": float(iaf_hz), "band_hz": list(band),
                  "artefact_rule": A.artefact_rule, "artefact_uv": float(A.artefact_ptp_uv),
                  "edge_margin_samples": margin, "units": "log10(uV^2/Hz)" if A.data_scale == 1e6 else
                  f"log10((V x {A.data_scale:g})^2/Hz)",
                  "roi_available": {r: list(v) for r, v in avail.items()},
                  "bad_roi_channels": [c for c in prim.channels if c in bads],
                  "n_bad_annotation_spans": int(len(bad_sp)), "windows": {}, "estimators": {}}
    spectra_rows: list[pd.DataFrame] = []
    windows = dict(task_cfg.prestim_windows)
    if "primary" not in windows:
        raise ValueError(f"{task}: no primary prestimulus window configured")
    want = A.windows_ms.get(task)
    if want is not None and {k: tuple(map(float, v)) for k, v in windows.items()} != want:
        raise ValueError(f"{task}: events.yaml windows {windows} != configs/features.yaml {want}")
    f_grid = np.fft.rfftfreq(A.nfft, 1 / sf)

    def psd(W, est="dpss"):
        if len(W) == 0:
            return np.zeros(W.shape[:2] + (len(f_grid),)), f_grid
        return sp.psd_estimate(W, sf, est, A)

    def log_bp(P, f, channels, bnd):
        if len(P) == 0:
            return np.zeros(0)
        if not channels:
            return np.full(len(P), np.nan)
        return sp.roi_log_band_power(P, f, names, channels, bnd, tol)

    W = P = f = a = b = ok = None
    for wn, (w0, w1) in windows.items():
        a_w, b_w = sp.window_bounds(onsets, w0, w1, sf)
        n = sp.window_length(w0, w1, sf)
        sfx = "" if wn == "primary" else f"_{wn}"
        edge_col = f"edge_valid_{wn}"
        if A.use_edge_flags and edge_col not in trials.columns:
            raise KeyError(f"trial table lacks {edge_col}")
        base = sp.window_validity(a_w, b_w, n_times=n_times, seg=seg if A.check_segments else None, margin=margin,
                                  bad_spans=bad_sp, edge_flag=trials[edge_col].to_numpy() if A.use_edge_flags
                                  else None)
        meta["windows"][wn] = {"ms": [w0, w1], "n_samples": n,
                               "offset_samples": [sp.ms_to_samples(w0, sf), sp.ms_to_samples(w1, sf)]}
        meta["estimators"][wn] = sp.estimator_info(n, sf, A)
        W_w = sp.extract_windows(X, a_w, n) if n_rows else np.zeros((0, len(names), n))
        P_w, f_w = psd(W_w)
        ptp = sp.peak_to_peak(W_w[:, roi_idx["primary"], :]) if n_rows else np.zeros(0)
        reason = sp.roi_validity(base, len(avail["primary"]), prim.min_channels, ptp, thr, A.artefact_keep_equal)
        ok_w = reason == ""
        sp.assert_no_bad_overlap(a_w, b_w, ok_w, bad_sp, f"{task} window {wn}")
        out[f"alpha{sfx}"] = np.where(ok_w, log_bp(P_w, f_w, avail["primary"], band), np.nan)
        out[f"valid{sfx}"] = ok_w
        out[f"invalid_reason{sfx}"] = pd.array(reason.astype(str), dtype="string")
        out[f"ptp_uv{sfx}"] = ptp * 1e6 / A.data_scale
        meta["windows"][wn]["band_bins"] = int(sp.band_mask(f_w, *band, tol=tol).sum())
        if wn == "primary":
            W, P, f, a, b, ok = W_w, P_w, f_w, a_w, b_w, ok_w
    n = W.shape[-1]
    out["win_start"], out["win_stop"] = a, b
    out["n_roi"] = len(avail["primary"])
    out["segment_id"] = seg.segment_of(a)
    # ---- fixed bands and estimators on the primary window, primary validity
    for col, key in (("alpha_s_band", "s_band"), ("alpha_e9", "e9"), ("ap_2p5_5", "ap_2p5_5"),
                     ("ap_15_25", "ap_15_25")):
        out[col] = np.where(ok, log_bp(P, f, avail["primary"], A.fixed_bands[key]), np.nan)
    for col, est in (("alpha_s_est1", "welch"), ("alpha_s_est2", "mt4")):
        Pe, fe = psd(W, est)
        out[col] = np.where(ok, log_bp(Pe, fe, avail["primary"], band), np.nan)
    if iaf_hz_s_iaf is not None and np.isfinite(iaf_hz_s_iaf):
        out["alpha_s_iaf"] = np.where(ok, log_bp(P, f, avail["primary"],
                                                 sp.iaf_band(iaf_hz_s_iaf, A.iaf_half_width_hz)), np.nan)
        out["iaf_hz_s_iaf"] = float(iaf_hz_s_iaf)
    else:
        out["alpha_s_iaf"] = np.nan
        out["iaf_hz_s_iaf"] = np.nan
    # ---- ROI variants: own channel count and artefact rule (same window rules otherwise)
    base_p = sp.window_validity(a, b, n_times=n_times, seg=seg if A.check_segments else None, margin=margin,
                                bad_spans=bad_sp, edge_flag=trials["edge_valid_primary"].to_numpy()
                                if A.use_edge_flags else None)
    for col, r in (("s_roi", "ptheta"), ("e7_mu", "mu"), ("e7_mu_left", "mu_left"), ("e7_mu_right", "mu_right")):
        spec = rois[r]
        ptp_r = sp.peak_to_peak(W[:, roi_idx[r], :]) if n_rows else np.zeros(0)
        rr = sp.roi_validity(base_p, len(avail[r]), spec.min_channels, ptp_r, thr, A.artefact_keep_equal)
        okr = rr == ""
        out[f"alpha_{col}"] = np.where(okr, log_bp(P, f, avail[r], band), np.nan)
        out[f"valid_{col}"] = okr
        out[f"invalid_reason_{col}"] = pd.array(rr.astype(str), dtype="string")
    # ---- split-channel halves (subsets of the primary ROI: primary validity + own channel minimum)
    for r in ("half_left", "half_right"):
        okh = ok & (len(avail[r]) >= rois[r].min_channels)
        out[f"alpha_{r}"] = np.where(okh, log_bp(P, f, avail[r], band), np.nan)
    # ---- split-time halves (AVSRT 800 ms)
    if task in A.split_time_tasks:
        s1, s2 = sp.split_time_slices(n)
        meta["estimators"]["split_time_half"] = sp.estimator_info(s1.stop - s1.start, sf, A)
        for col, sl in (("alpha_time1", s1), ("alpha_time2", s2)):
            Ph, fh = psd(np.ascontiguousarray(W[:, :, sl]))
            out[col] = np.where(ok, log_bp(Ph, fh, avail["primary"], band), np.nan)
    # ---- mean prestimulus spectrum (predictor set, valid windows)
    ps = ok & out["in_predictor_set"].to_numpy(bool)
    keep = f <= A.spectra_fmax_hz
    if ps.any() and avail["primary"]:
        ls = sp.roi_log_spectrum(P[ps], names, avail["primary"])
        spectra_rows.append(pd.DataFrame({"kind": "pre_predictor_set", "freq_hz": f[keep],
                                          "mean_log10_psd": ls[:, keep].mean(axis=0), "n_windows": int(ps.sum())}))
    # ---- no ICA correction
    if raw_noica is not None:
        if raw_noica.n_times != n_times:
            raise ValueError("raw_noica has a different sample axis")
        names_n = _picks(raw_noica, avail["primary"])
        Xn = raw_noica.get_data(picks=names_n) * A.data_scale
        Wn = sp.extract_windows(Xn, a, n) if n_rows else np.zeros((0, len(names_n), n))
        ptp_n = sp.peak_to_peak(Wn) if n_rows else np.zeros(0)
        # the no-ICA copy's own BAD* annotations (load_clean adds BAD_icafit to both variants), united
        # with the primary copy's, so a window near an ICA fit-rejected chunk is invalid in both variants
        bad_sp_n = sp.bad_annotation_spans(raw_noica, A.bad_annotation_prefix)
        base_n = sp.window_validity(a, b, n_times=n_times, seg=seg if A.check_segments else None, margin=margin,
                                    bad_spans=np.concatenate([bad_sp, bad_sp_n]).reshape(-1, 2),
                                    edge_flag=trials["edge_valid_primary"].to_numpy() if A.use_edge_flags else None)
        rn = sp.roi_validity(base_n, len(names_n), prim.min_channels, ptp_n, thr, A.artefact_keep_equal)
        okn = rn == ""
        sp.assert_no_bad_overlap(a, b, okn, bad_sp_n, f"{task} no-ICA window")
        meta["n_bad_annotation_spans_s_ica"] = int(len(bad_sp_n))
        if n_rows and names_n:
            Pn, fn = sp.psd_estimate(Wn, sf, "dpss", A)
            v = sp.roi_log_band_power(Pn, fn, names_n, names_n, band, tol)
        else:
            v = np.full(n_rows, np.nan)
        out["alpha_s_ica"] = np.where(okn, v, np.nan)
        out["valid_s_ica"] = okn
        out["invalid_reason_s_ica"] = pd.array(rn.astype(str), dtype="string")
        out["ptp_uv_s_ica"] = ptp_n * 1e6 / A.data_scale
        tmp = raw_noica.info.get("temp")
        meta["s_ica_noica_copy_ica_applied"] = bool(tmp.get("ica_applied", False)) if isinstance(tmp, dict) else None
    # ---- previous evoked response removed
    if task in A.erp_tasks and stim_samples is not None:
        rt_hi = sp.ms_to_samples(task_cfg.primary.hi_ms, sf)
        stop = erp_leave_out_stop(onsets, stim_samples, rt_hi)
        We, lag_ms, n_tpl, frac = erp_corrected_windows(
            X[roi_idx["primary"]], a, n, onsets, stim_samples, seg, sf, epoch_s=A.erp_epoch_s,
            leave_out_stop=stop, min_epochs=A.erp_min_epochs, bad_spans=bad_sp)
        if n_rows and avail["primary"]:
            Pe, fe = sp.psd_estimate(We, sf, "dpss", A)
            v = sp.roi_log_band_power(Pe, fe, avail["primary"], avail["primary"], band, tol)
        else:
            v = np.full(n_rows, np.nan)
        out["alpha_s_erp"] = np.where(ok, v, np.nan)
        out["erp_lag_ms"] = lag_ms
        out["erp_n_template_min"] = n_tpl
        out["erp_frac_corrected"] = frac
        meta["s_erp"] = {"epoch_s": A.erp_epoch_s, "min_epochs": A.erp_min_epochs,
                         "leave_out": A.erp_leave_out_until, "rt_hi_ms": float(task_cfg.primary.hi_ms),
                         "n_events": int(len(np.unique(stim_samples))),
                         "n_rows_corrected": int(np.isfinite(lag_ms).sum())}
    # ---- context power a_ctx
    xc = X[roi_idx["primary"]] if len(avail["primary"]) >= prim.min_channels else np.zeros((0, n_times))
    ctx = sp.context_band_power(xc, sf, a, onsets, seg, *band, A.context_lengths_s, A.context_min_valid_frac,
                                **A.context_filter)
    for Ls in A.context_lengths_s:
        tag = f"{Ls:g}"
        m, frc = ctx[float(Ls)]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"actx{tag}_log"] = np.where(m > 0, np.log10(m), np.nan)
        out[f"actx{tag}_frac"] = frc
    # ---- post-stimulus window (AVSRT; post-stimulus alpha desynchronisation)
    if task in A.pc1_tasks:
        w0, w1 = A.pc1_window_ms
        ap_, bp_ = sp.window_bounds(onsets, w0, w1, sf)
        npost = sp.window_length(w0, w1, sf)
        basep = sp.window_validity(ap_, bp_, n_times=n_times, seg=seg, margin=margin, bad_spans=bad_sp)
        meta["windows"]["post_pc1"] = {"ms": [w0, w1], "n_samples": npost}
        meta["estimators"]["post_pc1"] = sp.estimator_info(npost, sf, A)
        Wp = sp.extract_windows(X, ap_, npost) if n_rows else np.zeros((0, len(names), npost))
        ptp = sp.peak_to_peak(Wp[:, roi_idx["primary"], :]) if n_rows else np.zeros(0)
        rp = sp.roi_validity(basep, len(avail["primary"]), prim.min_channels, ptp, thr, A.artefact_keep_equal)
        Pp, fp = psd(Wp)
        okp = rp == ""
        out["alpha_post_pc1"] = np.where(okp, log_bp(Pp, fp, avail["primary"], band), np.nan)
        out["valid_post_pc1"] = okp
        out["invalid_reason_post_pc1"] = pd.array(np.asarray(rp).astype(str), dtype="string")
        if n_rows and "modality" in out.columns and avail["primary"]:
            va = okp & ps & out["modality"].isin(["V", "AV"]).to_numpy(bool)
            if va.any():
                lpost = sp.roi_log_spectrum(Pp[va], names, avail["primary"])
                lpre = sp.roi_log_spectrum(P[va], names, avail["primary"])
                for kind, ls in (("post_pc1_va", lpost), ("pre_pc1_va", lpre)):
                    spectra_rows.append(pd.DataFrame({"kind": kind, "freq_hz": f[keep],
                                                      "mean_log10_psd": ls[:, keep].mean(axis=0),
                                                      "n_windows": int(va.sum())}))
    meta["n_valid"] = {c: int(out[c].sum()) for c in out.columns if c.startswith("valid")}
    spectra = pd.concat(spectra_rows, ignore_index=True) if spectra_rows else \
        pd.DataFrame(columns=["kind", "freq_hz", "mean_log10_psd", "n_windows"])
    return AlphaResult(out, spectra, meta)


# --------------------------------------------------------------------------- recording level


