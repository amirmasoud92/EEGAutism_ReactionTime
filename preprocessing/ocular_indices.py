"""Trial-level ocular indices, blink and saccade detection on the stored ocular copy.

This code reads the stored ocular copy (``<stem>_desc-ocular_raw.fif``). The functions are re-exported by
:mod:`preprocessing.ocular`.

:func:`compute_ocular_indices` turns the copy into one row of ocular indices per stimulus onset. It needs only
onset samples and a window definition; it never sees a behaviour column. :func:`detect_blinks` and
:func:`detect_saccades` are the fixed blink and horizontal-saccade rules (``configs/ocular.yaml`` ``blinks:`` and
``saccades:``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from utils.config_ocular import OcularConfig, SaccadeConfig, as_ocular_config
from preprocessing.ocular import DERIVED, PIECE_CH, VALID_CH, _filter_piece, fir_half_length_s, ocular_meta

__all__ = ["INDEX_COLUMNS", "detect_blinks", "running_median_by_piece", "compute_ocular_indices", "detect_saccades",
           "saccade_config", "SPEC_DEFAULT_SACCADES"]

INDEX_COLUMNS = [
    "onset_sample", "heog_logsd", "vfp_logsd", "valid_window", "window_invalid_reason",
    "closure", "gaze_dev", "rm_coverage", "valid_closure", "closure_invalid_reason",
    "blink_in_window", "n_blinks_in_window", "time_since_last_blink_s", "tslb_censored", "valid_blink",
    "heog_source", "veog_ref",
]


# --------------------------------------------------------------------------- helpers on the copy
def _piece_bounds(piece: np.ndarray) -> list[tuple[int, int, int]]:
    """(piece id, start, stop) of each run of equal non-negative piece ids (relative indices)."""
    out = []
    p = np.round(piece).astype(int)
    if p.size == 0:
        return out
    edges = np.flatnonzero(np.diff(p)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [p.size]])
    for a, b in zip(starts, stops):
        if p[a] >= 0:
            out.append((int(p[a]), int(a), int(b)))
    return out


def running_median_by_piece(x: np.ndarray, bounds: Sequence[tuple[int, int, int]], half: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Centred running median over ``2*half+1`` samples, computed inside each piece only.

    Returns the median and the coverage (fraction of the nominal window that
    holds samples of the same piece). Outside the pieces both are NaN.
    """
    med = np.full(x.size, np.nan)
    cov = np.full(x.size, np.nan)
    w = 2 * half + 1
    for _, a, b in bounds:
        s = pd.Series(np.asarray(x[a:b], dtype=np.float64))
        med[a:b] = s.rolling(w, center=True, min_periods=1).median().to_numpy()
        cnt = s.rolling(w, center=True, min_periods=1).count().to_numpy()
        cov[a:b] = cnt / w
    return med, cov


def detect_blinks(veog_uv: np.ndarray, sfreq: float, cfg: OcularConfig | Mapping[str, Any] | None = None,
                  bounds: Sequence[tuple[int, int, int]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Detect blinks on the VEOG proxy.

    Rule: band-pass 1-10 Hz (the band of
    ``mne.preprocessing.find_eog_events``) inside each piece; keep peaks of
    the configured polarity whose prominence (measured within a
    ``local_wlen_s`` window, so distant troughs cannot inflate it) exceeds ``clip(k_mad * sigma,
    min_prominence_uv, max_threshold_uv)``, where ``sigma`` is the MAD-based
    background SD re-estimated ``sigma_iterations`` times with +-0.25 s around
    detected blinks excluded; full width at half prominence within
    ``width_s``; peaks at least ``min_separation_s`` apart. Each candidate
    must also be sharp in the 15-Hz signal itself: its prominence within a
    ``local_wlen_s`` window must exceed the same threshold, with full width at
    half that prominence <= ``local_max_width_s`` (this rejects slow eye
    movements and drifts that leak through the 1-Hz band-pass edge). MNE's own ``(max - min) / 4``
    threshold is not used: on DC-coupled data with slow drifts it is
    dominated by the largest excursion.

    The blink *extent* is the half-prominence interval of the band-passed
    peak padded by ``extent_pad_fwhm`` x FWHM on each side (for a
    raised-cosine blink of duration D, FWHM = D/2 and the extent is ~D).

    Parameters
    ----------
    veog_uv
        VEOG proxy (15-Hz copy) in microvolts, at ``sfreq``.
    bounds
        ``(piece id, start, stop)`` runs (default: the whole signal).

    Returns
    -------
    blinks
        One row per blink: ``peak`` (index), ``start``/``stop`` (extent,
        fractional indices), ``prominence_uv`` (band-passed),
        ``local_prominence_uv`` (15-Hz signal, ``local_wlen_s`` window),
        ``width_s`` (FWHM), ``piece``, ``edge_dist_s`` (distance of the peak
        to the nearer piece edge).
    info
        ``sigma_uv``, ``threshold_uv``, ``half_length_s`` (band-pass FIR),
        ``n_candidates`` and ``n_rejected_local`` (failed the sharpness check).
    """
    from scipy.signal import find_peaks, peak_prominences, peak_widths

    b = as_ocular_config(cfg).blinks
    x = np.asarray(veog_uv, dtype=np.float64)
    if bounds is None:
        bounds = [(0, 0, x.size)]
    sign = 1.0 if b.polarity == "positive" else -1.0
    hl = fir_half_length_s(sfreq, b.band_hz[0], b.band_hz[1])
    y = np.full(x.size, np.nan)
    for _, a, c in bounds:
        if c - a >= 2 * int(hl * sfreq) + 1:
            y[a:c] = sign * _filter_piece(x[a:c], sfreq, b.band_hz[0], b.band_hz[1], "hamming")
    good = np.isfinite(y)
    cols = ["peak", "start", "stop", "prominence_uv", "local_prominence_uv", "width_s", "piece", "edge_dist_s"]
    if good.sum() < 10:
        return pd.DataFrame(columns=cols), {"sigma_uv": np.nan, "threshold_uv": np.nan, "half_length_s": hl,
                                            "n_candidates": 0, "n_rejected_local": 0}
    dist = max(1, int(round(b.min_separation_s * sfreq)))
    wlim = (b.width_s[0] * sfreq, b.width_s[1] * sfreq)
    wlen = 2 * int(round(b.local_wlen_s * sfreq / 2)) + 1  # prominence measured within +-local_wlen_s/2

    def peaks(thr: float) -> list[tuple[int, np.ndarray, dict, int, int]]:
        out = []
        for pid, a, c in bounds:
            seg = y[a:c]
            if not np.isfinite(seg).all() or seg.size < 3:
                continue
            pk, pr = find_peaks(seg, prominence=thr, distance=dist, width=wlim, rel_height=0.5, wlen=wlen)
            out.append((pid, pk, pr, a, c))
        return out

    excl = np.zeros(x.size, dtype=bool)
    pad = int(round(0.25 * sfreq))
    thr = b.min_prominence_uv
    sigma = np.nan
    for _ in range(max(1, b.sigma_iterations)):
        sel = good & ~excl
        if sel.sum() < 10:
            break
        v = y[sel]
        sigma = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        thr = float(np.clip(b.k_mad * sigma, b.min_prominence_uv, b.max_threshold_uv))
        excl[:] = False
        for _, pk, _, a, _ in peaks(thr):
            for p in pk:
                excl[max(0, a + p - pad): a + p + pad + 1] = True
    rows = []
    n_cand = n_rej = 0
    shift = max(1, int(round(0.05 * sfreq)))
    for pid, pk, pr, a, c in peaks(thr):
        if pk.size == 0:
            continue
        n_cand += pk.size
        xs = sign * x[a:c]
        # sharpness in the 15-Hz signal: local maximum within +-50 ms of the band-passed peak
        lp = np.array([max(0, p - shift) + int(np.argmax(xs[max(0, p - shift): p + shift + 1])) for p in pk])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lprom, lb, rb = peak_prominences(xs, lp, wlen=wlen)
            lw = peak_widths(xs, lp, rel_height=0.5, prominence_data=(lprom, lb, rb))[0] / sfreq
        keep = (lprom >= thr) & (lw <= b.local_max_width_s)
        n_rej += int((~keep).sum())
        fwhm = pr["widths"]
        lo = pr["left_ips"] - b.extent_pad_fwhm * fwhm
        hi = pr["right_ips"] + b.extent_pad_fwhm * fwhm
        for i in np.flatnonzero(keep):
            p = int(pk[i])
            rows.append((a + p, a + float(lo[i]), a + float(hi[i]), float(pr["prominences"][i]), float(lprom[i]),
                         float(fwhm[i]) / sfreq, int(pid), min(p, c - a - 1 - p) / sfreq))
    df = pd.DataFrame(rows, columns=cols)
    return df, {"sigma_uv": sigma, "threshold_uv": thr, "half_length_s": hl, "n_candidates": n_cand,
                "n_rejected_local": n_rej}


# --------------------------------------------------------------------------- trial indices
def _window_idx(onsets: np.ndarray, ms: tuple[float, float], orig_sfreq: float, decim: int,
                first_oc: int) -> tuple[np.ndarray, np.ndarray]:
    """Relative ocular indices [lo, hi) of samples whose BDF time lies in [onset+start, onset+stop)."""
    lo = np.ceil((onsets + ms[0] * orig_sfreq / 1000.0) / decim - 1e-9).astype(np.int64) - first_oc
    hi = np.ceil((onsets + ms[1] * orig_sfreq / 1000.0) / decim - 1e-9).astype(np.int64) - first_oc
    return lo, hi


def _log(x: float, base: float) -> float:
    return math.log(x) / math.log(base) if x > 0 else np.nan


def compute_ocular_indices(ocular_raw: Any, onsets_samples: Sequence[int] | np.ndarray,
                           cfg: OcularConfig | Mapping[str, Any] | None = None, *, task: str | None = None,
                           window_ms: tuple[float, float] | None = None) -> pd.DataFrame:
    """Trial-level ocular indices for stimulus onsets (no behaviour is read or needed).

    Parameters
    ----------
    ocular_raw
        Output of :func:`build_ocular_copy` (or its saved FIF re-read).
    onsets_samples
        Stimulus onsets as BDF samples at the original rate (512 Hz), e.g. the
        ``sample`` column of a trial table.
    cfg
        Ocular config.
    task, window_ms
        The alpha window: explicit ``window_ms`` ([start, stop) ms relative to
        onset), else ``cfg.indices.windows_ms[task]``.

    Returns
    -------
    pandas.DataFrame
        One row per onset, in input order, with :data:`INDEX_COLUMNS`:

        - ``heog_logsd``, ``vfp_logsd``: log SD (base ``log_base``, uV) of the
          15-Hz HEOG / VEOG after removing the window's linear trend, in the
          alpha window (``heog_z`` / ``vfp_z`` before z-scoring);
        - ``closure``: mean over the closure window ([-1.5, -0.05] s) of the
          5-Hz VEOG minus its +-30-s running median (uV;
          ``closure_z`` before z-scoring);
        - ``gaze_dev``: mean absolute deviation over the same window of the
          5-Hz HEOG from its +-30-s running median (uV);
        - ``blink_in_window``: a blink extent overlaps the alpha window;
          ``n_blinks_in_window``; ``time_since_last_blink_s``: onset minus the
          last blink peak before the onset in the same piece (NaN and
          ``tslb_censored`` if none);
        - validity: ``valid_window`` (alpha window inside one valid piece and
          at least the 15-Hz FIR half-length from its edges),
          ``valid_closure`` (closure window inside one piece, 5-Hz FIR
          half-length from the edges, running-median coverage >=
          ``running_median_min_coverage``), ``valid_blink`` (alpha window
          at least the blink band-pass + 15-Hz half-lengths from the edges),
          with the first failing reason in ``*_invalid_reason``;
        - ``heog_source``, ``veog_ref``: the channels used (fallbacks).

        Indices are NaN when invalid or when no source exists. ``df.attrs``
        holds the window, blink-detection summary and build metadata.
        z-scoring within child x task is left to the caller.
    """
    c = as_ocular_config(cfg)
    ind = c.indices
    meta = ocular_meta(ocular_raw)
    if window_ms is None:
        if task is None or task not in ind.windows_ms:
            raise ValueError(f"give window_ms or a task in {sorted(ind.windows_ms)}")
        window_ms = ind.windows_ms[task]
    window_ms = (float(window_ms[0]), float(window_ms[1]))
    orig_sfreq, decim = float(meta["orig_sfreq"]), int(meta["decim"])
    sf = float(ocular_raw.info["sfreq"])
    first_oc = int(ocular_raw.first_samp)
    names = ocular_raw.ch_names
    data = ocular_raw.get_data(picks=[names.index(ch) for ch in (*DERIVED, VALID_CH, PIECE_CH)])
    v5, h5, v15, h15 = (data[i] * 1e6 for i in range(4))
    valid = data[4] > 0.5
    piece = np.where(valid, np.round(data[5]), -1)
    bounds = _piece_bounds(piece)
    n = valid.size
    src = meta["sources"]
    has_h, has_v = src["heog"] is not None, bool(src["fp"]) and src["ref"] is not None

    half = int(round(ind.running_median_half_s * sf))
    rm_v, cov = running_median_by_piece(v5, bounds, half)
    rm_h, _ = running_median_by_piece(h5, bounds, half)
    dev_v, dev_h = v5 - rm_v, h5 - rm_h

    blinks, binfo = (detect_blinks(v15, sf, c, bounds) if has_v
                     else (pd.DataFrame(columns=["peak", "start", "stop", "piece"]),
                           {"sigma_uv": np.nan, "threshold_uv": np.nan, "n_candidates": 0, "n_rejected_local": 0,
                            "half_length_s": fir_half_length_s(sf, *c.blinks.band_hz)}))
    g_fast = int(math.ceil(meta["half_length_fast_s"] * sf))
    g_slow = int(math.ceil(meta["half_length_slow_s"] * sf))
    g_blink = g_fast + int(math.ceil(binfo["half_length_s"] * sf))

    onsets = np.asarray(onsets_samples, dtype=np.float64).ravel()
    wlo, whi = _window_idx(onsets, window_ms, orig_sfreq, decim, first_oc)
    clo, chi = _window_idx(onsets, ind.closure_window_ms, orig_sfreq, decim, first_oc)
    on_rel = onsets / decim - first_oc  # fractional ocular index of the onset
    bounds_by_id = {pid: (a, b) for pid, a, b in bounds}
    bpk = blinks["peak"].to_numpy(dtype=float) if len(blinks) else np.array([])
    bst = blinks["start"].to_numpy(dtype=float) if len(blinks) else np.array([])
    bsp = blinks["stop"].to_numpy(dtype=float) if len(blinks) else np.array([])
    bpc = blinks["piece"].to_numpy(dtype=int) if len(blinks) else np.array([], dtype=int)

    def check(lo: int, hi: int, guard: int) -> tuple[str, int | None]:
        if hi <= lo:
            return "empty_window", None
        if lo < 0 or hi > n:
            return "out_of_range", None
        if not valid[lo:hi].all():
            return "invalid_samples", None
        p = piece[lo:hi]
        if p.min() != p.max():
            return "crosses_join", None
        pid = int(p[0])
        a, b = bounds_by_id[pid]
        if lo - a < guard or b - hi < guard:
            return "near_edge", pid
        return "", pid

    log_base = ind.log_base
    detrend_linear = ind.logsd_detrend == "linear"
    rows = []
    for k in range(onsets.size):
        lo, hi, cl, ch = int(wlo[k]), int(whi[k]), int(clo[k]), int(chi[k])
        r: dict[str, Any] = {"onset_sample": int(onsets[k])}
        reason, pid = check(lo, hi, g_fast)
        r["valid_window"] = reason == ""
        r["window_invalid_reason"] = reason
        hsd = vsd = np.nan
        if reason == "":
            t = np.arange(hi - lo, dtype=float)
            for arr, have, key in ((h15, has_h, "h"), (v15, has_v, "v")):
                if not have:
                    continue
                seg = arr[lo:hi]
                if detrend_linear and seg.size >= 3:
                    seg = seg - np.polyval(np.polyfit(t, seg, 1), t)
                else:
                    seg = seg - seg.mean()
                val = _log(float(seg.std(ddof=1)) if seg.size > 1 else np.nan, log_base)
                if key == "h":
                    hsd = val
                else:
                    vsd = val
        r["heog_logsd"], r["vfp_logsd"] = hsd, vsd

        creason, _ = check(cl, ch, g_slow)
        covk = float(np.nanmin(cov[cl:ch])) if creason in ("", "near_edge") else np.nan
        if creason == "" and not covk >= ind.running_median_min_coverage:
            creason = "rm_coverage"
        r["rm_coverage"] = covk
        r["valid_closure"] = creason == ""
        r["closure_invalid_reason"] = creason
        r["closure"] = float(np.mean(dev_v[cl:ch])) if (creason == "" and has_v) else np.nan
        r["gaze_dev"] = float(np.mean(np.abs(dev_h[cl:ch]))) if (creason == "" and has_h) else np.nan

        breason, bpid = check(lo, hi, g_blink)
        r["valid_blink"] = breason == "" and has_v
        if r["valid_blink"]:
            same = bpc == bpid
            ov = same & (bsp >= lo) & (bst < hi)
            r["n_blinks_in_window"] = int(ov.sum())
            r["blink_in_window"] = bool(ov.any())
            prior = same & (bpk < on_rel[k])
            if prior.any():
                r["time_since_last_blink_s"] = float((on_rel[k] - bpk[prior].max()) / sf)
                r["tslb_censored"] = False
            else:
                r["time_since_last_blink_s"] = np.nan
                r["tslb_censored"] = True
        else:
            r["n_blinks_in_window"] = pd.NA
            r["blink_in_window"] = pd.NA
            r["time_since_last_blink_s"] = np.nan
            r["tslb_censored"] = pd.NA
        r["heog_source"] = src["heog_label"]
        r["veog_ref"] = src["ref_label"] if has_v else "none"
        rows.append(r)
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if len(df):
        df["n_blinks_in_window"] = df["n_blinks_in_window"].astype("Int64")
        df["blink_in_window"] = df["blink_in_window"].astype("boolean")
        df["tslb_censored"] = df["tslb_censored"].astype("boolean")
    valid_minutes = valid.sum() / sf / 60.0
    df.attrs.update({
        "window_ms": list(window_ms), "closure_window_ms": list(ind.closure_window_ms), "task": task,
        "blink_threshold_uv": binfo["threshold_uv"], "blink_sigma_uv": binfo["sigma_uv"],
        "n_blinks": int(len(blinks)), "blink_candidates": binfo["n_candidates"],
        "blink_rejected_local": binfo["n_rejected_local"], "blink_rate_per_min": (len(blinks) / valid_minutes) if valid_minutes else np.nan,
        "sources": src,
    })
    return df


# --------------------------------------------------------------------------- saccades
#: default rule when ``configs/ocular.yaml`` has no ``saccades:`` section
SPEC_DEFAULT_SACCADES = SaccadeConfig(rule="step", signal="HEOG_lp15", step_uv=30.0, within_ms=50.0,
                                      refractory_ms=0.0)
SACCADE_COLUMNS = ["piece", "start", "stop", "amp_uv"]


def saccade_config(cfg: OcularConfig | SaccadeConfig | Mapping[str, Any] | None = None) -> SaccadeConfig:
    """The configured ``saccades:`` rule, or :data:`SPEC_DEFAULT_SACCADES` if the section is absent."""
    if isinstance(cfg, SaccadeConfig):
        return cfg
    c = as_ocular_config(cfg)
    return c.saccades if c.saccades is not None else SPEC_DEFAULT_SACCADES


def detect_saccades(heog_uv: np.ndarray, sfreq: float, bounds: Sequence[tuple[int, int, int]],
                    cfg: OcularConfig | SaccadeConfig | Mapping[str, Any] | None = None,
                    edge_guard: int = 0, blinks: pd.DataFrame | None = None) -> pd.DataFrame:
    """HEOG step events inside each piece (``bounds``: ``(piece id, start, stop)``, relative indices).

    ``rule = "step"``: supra-threshold samples t with ``max_{1<=l<=k} |x[t+l] - x[t]| > step_uv``, ``k`` = the
    largest lag within ``within_ms`` (>= 1); event extent ``[first t, last t + k]``.

    ``rule = "step_like"``: supra-threshold samples t with ``|mean(x[t, t+h)) - mean(x[t-h, t))| > step_uv``,
    ``h = round(within_ms / 2)`` in samples (>= 1); event extent ``[first t - h, last t + h - 1]``.

    Samples closer than ``edge_guard`` to a piece edge are not used. Events of the same piece whose gap is shorter
    than ``refractory_ms`` are merged. With ``exclude_blinks``, events overlapping a blink extent of
    ``blinks`` (:func:`detect_blinks` on the same copy; required then) widened by ``blink_pad_ms`` are dropped.
    Columns: ``piece``, ``start``, ``stop`` (inclusive relative indices), ``amp_uv`` (largest supra-threshold
    statistic of the event).
    """
    r = saccade_config(cfg)
    if r.exclude_blinks and blinks is None:
        raise ValueError("this saccade rule excludes blinks: pass blinks=detect_blinks(...)[0]")
    x = np.asarray(heog_uv, dtype=float)
    rows: list[tuple[int, int, int, float]] = []
    for pid, a, b in bounds:
        lo, hi = a + edge_guard, b - edge_guard
        seg = x[lo:hi]
        n = seg.size
        if r.rule == "step":
            k = max(1, int(math.floor(r.within_ms * sfreq / 1000.0 + 1e-9)))
            if n <= k:
                continue
            d = np.zeros(n - k)
            for lag in range(1, k + 1):
                d = np.maximum(d, np.abs(seg[lag: n - k + lag] - seg[: n - k]))
            off_lo, off_hi, base = 0, k, 0
        else:
            h = max(1, int(round(r.within_ms / 2.0 * sfreq / 1000.0)))
            if n < 2 * h + 1:
                continue
            c = np.concatenate([[0.0], np.cumsum(seg)])
            t = np.arange(h, n - h + 1)
            d = np.abs((c[t + h] - c[t]) / h - (c[t] - c[t - h]) / h)
            off_lo, off_hi, base = -h, h - 1, h
        hit = d > r.step_uv
        if not hit.any():
            continue
        e = np.flatnonzero(np.diff(np.concatenate([[0], hit.astype(int), [0]])))
        ev = [(lo + base + int(s0) + off_lo, lo + base + int(t0) - 1 + off_hi, float(d[s0:t0].max()))
              for s0, t0 in zip(e[0::2], e[1::2])]
        gap = int(round(r.refractory_ms * sfreq / 1000.0))
        merged: list[list[float]] = []
        for st, sp, amp in ev:
            if merged and gap > 0 and st - merged[-1][1] < gap:
                merged[-1][1] = max(merged[-1][1], sp)
                merged[-1][2] = max(merged[-1][2], amp)
            else:
                merged.append([st, sp, amp])
        rows.extend((int(pid), int(st), int(sp), float(amp)) for st, sp, amp in merged)
    out = pd.DataFrame(rows, columns=SACCADE_COLUMNS)
    if r.exclude_blinks and len(out) and blinks is not None and len(blinks):
        pad = r.blink_pad_ms * sfreq / 1000.0
        bs = blinks["start"].to_numpy(float) - pad
        be = blinks["stop"].to_numpy(float) + pad
        keep = np.array([not np.any((bs <= sp) & (be >= st)) for st, sp in zip(out["start"], out["stop"])])
        out = out[keep].reset_index(drop=True)
    return out

