"""Single-window spectral estimation of prestimulus alpha.

Building blocks used by :mod:`feature_extraction.prestimulus_alpha`. Parameters come from ``configs/features.yaml``
(:mod:`utils.config_features`); the keyword defaults below are the same values and are used when a function is called
on its own (e.g. by the AVSRT spectrum used for the IAF in :mod:`feature_extraction.iaf`).

Windows
-------
A window is the half-open sample interval ``[onset + round(start_ms x sfreq / 1000), onset + round(stop_ms x
sfreq / 1000))``, the same rounding as the ``edge_valid_<window>`` columns of the trial tables. At 512 Hz the AVSRT
window [-850, -50) ms is ``[onset - 435, onset - 26)`` (409 samples, 798.8 ms); FAST / IC [-550, -50) ms is
``[onset - 282, onset - 26)`` (256 samples). No sample at or after onset - 26 enters a window.

Validity (:func:`window_validity`, :func:`roi_validity`); the first failing reason is reported:

1. ``out_of_range``: the window leaves the recording;
2. ``edge_flag``: the trial-table ``edge_valid_<window>`` flag is not True (no join and no file edge within the
   1-40 Hz FIR half-length, 1.65 s, of either window edge; NA counts as not valid);
3. ``segment``: the window does not lie inside ONE valid span of the preprocessing
   :class:`~preprocessing.segments.SegmentInfo` with the same margin (adds rail / flat / unusable spans, at which the
   filters also restart, to the join rule);
4. ``bad_annotation``: the window overlaps a ``BAD*`` annotation of the cleaned Raw (checked directly on the
   annotations, independently of 3; :func:`assert_no_bad_overlap` re-checks the final valid set);
5. ``n_roi``: fewer than ``min_channels`` good channels of the ROI (5 of 8 for the primary ROI);
6. ``artefact``: peak-to-peak > ``ptp_uv`` (200 uV) on any good ROI channel. A window exactly at the threshold is
   kept.

Estimators (one-sided power spectral density of data in uV, so uV^2/Hz; 0.5-Hz grid from zero-padding to
NFFT = 1024):

- ``dpss`` (primary): MNE's multitaper with ``bandwidth = 2.0``, ``low_bias = True``, ``adaptive = False``,
  ``normalization = "full"``, ``remove_dc = True``. ``psd_array_multitaper`` has no zero-padding argument, so
  :func:`psd_dpss` calls the same MNE internals (``_compute_mt_params``, ``_mt_spectra`` with ``n_fft = 1024``,
  ``_psd_from_mt``, then ``/ sfreq``); at the native frequency bins it equals ``psd_array_multitaper`` to rounding.
  With these settings MNE keeps ONE taper at every window length used here, so the method is a DPSS single-taper
  periodogram. Where the bandwidth gives a normalised half-bandwidth below MNE's minimum of 0.5 (windows shorter
  than 256 samples: the 400-ms window, the split-time halves) it is raised to ``sfreq / n_times`` (still one taper);
  the value used is reported (:func:`dpss_params`).
- ``welch`` (sensitivity analysis, Additional file 1): Hann window, 256-sample segments, 50% overlap, constant
  detrend, density scaling; the whole window is one segment when it is shorter than 256 samples.
- ``mt4`` (sensitivity analysis): the DPSS estimator with ``bandwidth = 4.0`` (1-3 tapers depending on window
  length).

Band power: the mean of the PSD over all grid bins in ``[lo, hi]`` (inclusive, 1e-6 Hz tolerance). ROI value:
``log10`` of the mean over the good ROI channels of the band power (linear power averaged before the log).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Defaults for stand-alone calls (the feature code passes the configs/features.yaml values).
NFFT = 1024
DPSS_BANDWIDTH_HZ = 2.0
MT4_BANDWIDTH_HZ = 4.0
WELCH_NPERSEG = 256
BAND_TOL_HZ = 1e-6
ESTIMATORS = ("dpss", "welch", "mt4")


@dataclass(frozen=True)
class ROISpec:
    """A channel set and the minimum number of good channels a window needs."""

    name: str
    channels: tuple[str, ...]
    min_channels: int


def rois_from_config(acfg) -> dict[str, ROISpec]:
    """:class:`ROISpec` of every ROI in ``AlphaConfig.rois``."""
    return {k: ROISpec(k, tuple(r.channels), int(r.min_channels)) for k, r in acfg.rois.items()}


def all_roi_channels(rois: Iterable[ROISpec]) -> list[str]:
    """Union of the channels of ``rois`` in first-seen order."""
    out: list[str] = []
    for r in rois:
        out.extend(c for c in r.channels if c not in out)
    return out


# --------------------------------------------------------------------------- windows
def ms_to_samples(ms: float, sfreq: float) -> int:
    """``int(round(ms x sfreq / 1000))`` (the trial-table convention; Python rounds half to even)."""
    return int(round(float(ms) * float(sfreq) / 1000.0))


def window_bounds(onsets: Sequence[int] | np.ndarray, start_ms: float, stop_ms: float, sfreq: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Window ``[a, b)`` sample bounds for each onset (see module docstring)."""
    s = np.asarray(onsets, np.int64)
    a = s + ms_to_samples(start_ms, sfreq)
    b = s + ms_to_samples(stop_ms, sfreq)
    if ms_to_samples(stop_ms, sfreq) <= ms_to_samples(start_ms, sfreq):
        raise ValueError(f"empty window [{start_ms}, {stop_ms}) ms")
    return a, b


def window_length(start_ms: float, stop_ms: float, sfreq: float) -> int:
    """Samples in a ``[start_ms, stop_ms)`` window."""
    return ms_to_samples(stop_ms, sfreq) - ms_to_samples(start_ms, sfreq)


def extract_windows(data: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """``(n_windows, n_channels, n)`` copy of ``data[:, a:a+n]`` for each start ``a``.

    Starts outside ``[0, n_times - n]`` are clipped (the caller marks them invalid via ``out_of_range``).
    """
    data = np.asarray(data)
    st = np.clip(np.asarray(starts, np.int64), 0, max(data.shape[1] - n, 0))
    idx = st[:, None] + np.arange(n)[None, :]
    return np.ascontiguousarray(np.transpose(data[:, idx], (1, 0, 2)))


def split_time_slices(n: int) -> tuple[slice, slice]:
    """Two disjoint halves of an ``n``-sample window: the first and the last ``n // 2`` samples.

    For an odd ``n`` (AVSRT 409 samples) the middle sample belongs to neither half.
    """
    h = int(n) // 2
    if h < 1:
        raise ValueError("window too short to split")
    return slice(0, h), slice(int(n) - h, int(n))


def bad_annotation_spans(raw, prefix: str = "BAD") -> np.ndarray:
    """``(k, 2)`` sample spans (data-array indices, half-open) of annotations whose description starts with
    ``prefix`` (case-insensitive), e.g. ``BAD_rail`` / ``BAD_flat`` / ``BAD_unusable`` / any manual ``BAD``."""
    ann = raw.annotations
    if len(ann) == 0:
        return np.zeros((0, 2), np.int64)
    sf = float(raw.info["sfreq"])
    base = raw.first_time if ann.orig_time is not None else 0.0
    out = []
    for on, du, de in zip(ann.onset, ann.duration, ann.description):
        if str(de).lower().startswith(prefix.lower()):
            a = int(np.floor((on - base) * sf + 1e-9))
            b = int(np.ceil((on - base + du) * sf - 1e-9))
            out.append((a, max(b, a + 1)))       # a zero-duration BAD marks at least its own sample
    return np.asarray(out, np.int64).reshape(-1, 2)


def overlaps_spans(a: np.ndarray, b: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Whether each ``[a, b)`` intersects any half-open span of ``spans``."""
    a = np.atleast_1d(np.asarray(a, np.int64))
    b = np.atleast_1d(np.asarray(b, np.int64))
    out = np.zeros(a.shape, bool)
    for s0, s1 in np.asarray(spans, np.int64).reshape(-1, 2):
        out |= (a < s1) & (b > s0)
    return out


def assert_no_bad_overlap(a: np.ndarray, b: np.ndarray, valid: np.ndarray, bad_spans: np.ndarray, what: str) -> None:
    """Invariant: no window marked valid overlaps a ``BAD*`` annotation (raises ``AssertionError``)."""
    v = np.asarray(valid, bool)
    if len(bad_spans) and v.any():
        hit = overlaps_spans(np.asarray(a)[v], np.asarray(b)[v], bad_spans)
        if hit.any():
            raise AssertionError(f"{what}: {int(hit.sum())} valid window(s) overlap a BAD annotation")


def as_bool_na_false(x) -> np.ndarray:
    """Boolean array with missing values (NA / None / NaN) as False."""
    import pandas as pd

    return pd.array(list(x) if not hasattr(x, "dtype") else x, dtype="boolean").fillna(False).to_numpy(bool)


def window_validity(a: np.ndarray, b: np.ndarray, *, n_times: int, seg=None, margin: int = 0,
                    bad_spans: np.ndarray | None = None, edge_flag: Sequence | np.ndarray | None = None
                    ) -> np.ndarray:
    """First failing reason per window among ``out_of_range``, ``edge_flag``, ``segment``, ``bad_annotation``
    (empty string = valid). See the module docstring.

    Parameters
    ----------
    a, b
        Window bounds (samples).
    n_times
        Recording length.
    seg
        :class:`~alpha_state.preprocessing.segments.SegmentInfo` (``None`` skips the ``segment`` check).
    margin
        Samples required between each window edge and its valid span's edges (FIR half-length).
    bad_spans
        Spans of ``BAD*`` annotations (:func:`bad_annotation_spans`).
    edge_flag
        Trial-table ``edge_valid_<window>`` (``None`` skips; NA counts as not valid).
    """
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    reason = np.full(a.shape, "", dtype=object)

    def mark(bad: np.ndarray, why: str) -> None:
        reason[bad & (reason == "")] = why

    mark((a < 0) | (b > int(n_times)), "out_of_range")
    if edge_flag is not None:
        mark(~as_bool_na_false(edge_flag), "edge_flag")
    if seg is not None:
        mark(~seg.window_valid(a, b, margin=int(margin)), "segment")
    if bad_spans is not None and len(bad_spans):
        mark(overlaps_spans(a, b, bad_spans), "bad_annotation")
    return reason


def available_channels(roi: ROISpec, ch_names: Sequence[str], bads: Iterable[str]) -> list[str]:
    """Good channels of ``roi`` present in ``ch_names`` (bad channels are dropped)."""
    bad = set(bads)
    have = set(ch_names)
    return [c for c in roi.channels if c in have and c not in bad]


def peak_to_peak(win: np.ndarray) -> np.ndarray:
    """Max over channels of the peak-to-peak of ``(n_windows, n_channels, n)`` data (same unit as the data)."""
    if win.shape[1] == 0:
        return np.full(win.shape[0], np.nan)
    return np.ptp(win, axis=-1).max(axis=1)


def roi_validity(base_reason: np.ndarray, n_good: int, min_channels: int, ptp: np.ndarray, threshold: float,
                 keep_equal: bool = True) -> np.ndarray:
    """Add the ``n_roi`` and ``artefact`` rules to ``base_reason`` (copy). ``ptp`` and ``threshold`` share a unit."""
    r = np.asarray(base_reason, dtype=object).copy()
    if n_good < min_channels:
        r[r == ""] = "n_roi"
        return r
    p = np.asarray(ptp, float)
    ok = (p <= float(threshold)) if keep_equal else (p < float(threshold))   # NaN ptp is invalid
    r[(r == "") & ~ok] = "artefact"
    return r


# --------------------------------------------------------------------------- estimators
@lru_cache(maxsize=64)
def _dpss_cached(n_times: int, sfreq: float, bandwidth: float, low_bias: bool, raise_min: bool
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    from mne.time_frequency.multitaper import _compute_mt_params

    bw = float(bandwidth)
    if bw * n_times / (2.0 * sfreq) < 0.5:
        if not raise_min:
            raise ValueError(f"bandwidth {bw} Hz is below MNE's minimum for {n_times} samples")
        bw = float(sfreq) / n_times * (1.0 + 1e-12)       # MNE minimum (normalised half-bandwidth 0.5)
    tapers, eig, _ = _compute_mt_params(int(n_times), float(sfreq), bw, bool(low_bias), False, verbose="error")
    tapers = np.atleast_2d(np.asarray(tapers, float))
    tapers.setflags(write=False)
    eig = np.asarray(eig, float)
    eig.setflags(write=False)
    return tapers, eig, bw


def dpss_params(n_times: int, sfreq: float, bandwidth: float = DPSS_BANDWIDTH_HZ, low_bias: bool = True,
                raise_min: bool = True) -> dict:
    """Tapers, eigenvalues and the bandwidth actually used (MNE ``low_bias``; minimum raised if needed)."""
    tapers, eig, bw = _dpss_cached(int(n_times), float(sfreq), float(bandwidth), bool(low_bias), bool(raise_min))
    return {"tapers": tapers, "eigvals": eig, "bandwidth_hz": bw, "n_tapers": int(len(eig)),
            "bandwidth_raised": bool(bw != float(bandwidth))}


def psd_dpss(x: np.ndarray, sfreq: float, bandwidth: float = DPSS_BANDWIDTH_HZ, nfft: int = NFFT, *,
             low_bias: bool = True, remove_dc: bool = True, raise_min: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    """DPSS (multi)taper PSD of ``x (..., n_times)`` zero-padded to ``nfft`` (MNE ``normalization='full'``,
    ``adaptive=False``). Returns ``(psd (..., nfft//2+1), freqs)``."""
    from mne.time_frequency.multitaper import _mt_spectra, _psd_from_mt

    x = np.asarray(x, float)
    n = x.shape[-1]
    if nfft < n:
        raise ValueError(f"nfft {nfft} < window length {n}")
    p = dpss_params(n, sfreq, bandwidth, low_bias, raise_min)
    x2 = x.reshape(-1, n)
    x_mt, freqs = _mt_spectra(x2, p["tapers"], float(sfreq), n_fft=int(nfft), remove_dc=bool(remove_dc))
    w = np.sqrt(p["eigvals"])[np.newaxis, :, np.newaxis]
    psd = _psd_from_mt(x_mt, w) / float(sfreq)
    return psd.reshape(x.shape[:-1] + (len(freqs),)), freqs


def psd_welch(x: np.ndarray, sfreq: float, nperseg: int = WELCH_NPERSEG, nfft: int = NFFT, *,
              window: str = "hann", overlap_frac: float = 0.5, detrend: str = "constant"
              ) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (``min(n, nperseg)``-sample segments, density scaling)."""
    from scipy.signal import welch

    x = np.asarray(x, float)
    n = x.shape[-1]
    nper = min(n, int(nperseg))
    f, p = welch(x, fs=float(sfreq), window=window, nperseg=nper, noverlap=int(nper * overlap_frac),
                 nfft=int(nfft), detrend=detrend, scaling="density", axis=-1)
    return p, f


def psd_estimate(x: np.ndarray, sfreq: float, estimator: str = "dpss", acfg=None) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the estimator named in :data:`ESTIMATORS` with the ``AlphaConfig`` settings (module
    defaults when ``acfg`` is None)."""
    if acfg is None:
        if estimator == "dpss":
            return psd_dpss(x, sfreq, DPSS_BANDWIDTH_HZ, NFFT)
        if estimator == "mt4":
            return psd_dpss(x, sfreq, MT4_BANDWIDTH_HZ, NFFT)
        if estimator == "welch":
            return psd_welch(x, sfreq, WELCH_NPERSEG, NFFT)
        raise ValueError(f"unknown estimator {estimator!r}; one of {ESTIMATORS}")
    d = acfg.dpss
    if estimator in ("dpss", "mt4"):
        bw = d.bandwidth_hz if estimator == "dpss" else acfg.mt4_bandwidth_hz
        return psd_dpss(x, sfreq, bw, acfg.nfft, low_bias=d.low_bias, remove_dc=d.remove_dc,
                        raise_min=d.raise_to_mne_minimum)
    if estimator == "welch":
        w = acfg.welch
        return psd_welch(x, sfreq, w.nperseg, acfg.nfft, window=w.window, overlap_frac=w.overlap_frac,
                         detrend=w.detrend)
    raise ValueError(f"unknown estimator {estimator!r}; one of {ESTIMATORS}")


def estimator_info(n_times: int, sfreq: float, acfg=None) -> dict:
    """Tapers / bandwidth / Welch segments of each estimator at this window length."""
    bw2 = acfg.dpss.bandwidth_hz if acfg is not None else DPSS_BANDWIDTH_HZ
    bw4 = acfg.mt4_bandwidth_hz if acfg is not None else MT4_BANDWIDTH_HZ
    nps = acfg.welch.nperseg if acfg is not None else WELCH_NPERSEG
    ovl = acfg.welch.overlap_frac if acfg is not None else 0.5
    d2 = dpss_params(n_times, sfreq, bw2)
    d4 = dpss_params(n_times, sfreq, bw4)
    nper = min(n_times, nps)
    step = nper - int(nper * ovl)
    n_seg = 1 + (n_times - nper) // step
    return {"n_times": int(n_times), "dpss_n_tapers": d2["n_tapers"], "dpss_bandwidth_hz": d2["bandwidth_hz"],
            "dpss_bandwidth_raised": d2["bandwidth_raised"], "mt4_n_tapers": d4["n_tapers"],
            "mt4_bandwidth_hz": d4["bandwidth_hz"], "welch_nperseg": int(nper), "welch_n_segments": int(n_seg)}


# --------------------------------------------------------------------------- bands
def band_mask(freqs: np.ndarray, lo: float, hi: float, tol: float = BAND_TOL_HZ) -> np.ndarray:
    """Grid bins in ``[lo, hi]`` (inclusive)."""
    f = np.asarray(freqs, float)
    return (f >= lo - tol) & (f <= hi + tol)


def iaf_band(iaf_hz: float, half_width: float = 2.0) -> tuple[float, float]:
    """``(IAF - 2, IAF + 2)`` Hz."""
    if iaf_hz is None or not np.isfinite(iaf_hz):
        raise ValueError("IAF must be finite")
    return float(iaf_hz) - half_width, float(iaf_hz) + half_width


def band_mean_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float, tol: float = BAND_TOL_HZ
                    ) -> np.ndarray:
    """Mean PSD over the band bins (linear), last axis reduced."""
    m = band_mask(freqs, lo, hi, tol)
    if not m.any():
        raise ValueError(f"no frequency bin in [{lo}, {hi}] Hz")
    return psd[..., m].mean(axis=-1)


def roi_log_band_power(psd: np.ndarray, freqs: np.ndarray, ch_names: Sequence[str], channels: Sequence[str],
                       band: tuple[float, float], tol: float = BAND_TOL_HZ) -> np.ndarray:
    """``log10`` of the mean over ``channels`` of the band-mean power.

    ``psd`` is ``(n_windows, len(ch_names), n_freqs)``; ``channels`` must be a subset of ``ch_names`` (the good
    ROI channels). Linear power is averaged over channels before the log.
    """
    if len(channels) == 0:
        return np.full(psd.shape[0], np.nan)
    names = list(ch_names)
    idx = [names.index(c) for c in channels]
    bp = band_mean_power(psd[:, idx, :], freqs, *band, tol=tol)         # (n_windows, n_ch)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(bp.mean(axis=1))


def roi_log_spectrum(psd: np.ndarray, ch_names: Sequence[str], channels: Sequence[str]) -> np.ndarray:
    """``log10`` of the channel-mean (linear) PSD, ``(n_windows, n_freqs)``."""
    names = list(ch_names)
    idx = [names.index(c) for c in channels]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(psd[:, idx, :].mean(axis=1))


# --------------------------------------------------------------------------- context band power (a_ctx10)
def context_band_power(x: np.ndarray, sfreq: float, win_start: np.ndarray, onsets: np.ndarray, seg,
                       lo: float, hi: float, lengths_s: Sequence[float], min_valid_frac: float, *,
                       pad_s: float = 4.0, fir_design: str = "firwin", fir_window: str = "hamming",
                       phase: str = "zero") -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Mean IAF-band Hilbert power over the context interval of each trial, from data **before its onset only**.

    For trial i the piece ``[max(span start, w_i - max(L) - pad), onset_i)`` (same valid span as the window start
    ``w_i``) is band-passed (MNE ``filter_data``, FIR, default transition bands) and Hilbert-transformed on its own, so no sample at or after
    the current onset can reach the value (the zero-phase FIR and the FFT-based Hilbert transform of the whole span
    would let post-onset data reach back by the filter half-length, ~0.83 s at 8-12 Hz, into the FAST / IC context
    interval that ends 0.55 s before onset). Samples within the filter half-length of the span start and of the
    piece end (the onset) are NaN; the piece start (``pad`` before the longest interval) is not an edge of the
    interval. For AVSRT (window start 0.85 s before onset) the onset edge never reaches the interval; for FAST / IC
    it removes its last ~0.28 s.

    Returns ``{L: (mean power, valid fraction of L)}``; the mean is NaN below ``min_valid_frac``.
    """
    import mne
    from scipy.fft import next_fast_len
    from scipy.signal import hilbert

    x = np.asarray(x, float)
    w = np.asarray(win_start, np.int64)
    on = np.asarray(onsets, np.int64)
    Ls = [float(v) for v in lengths_s]
    nL = {L: int(round(L * sfreq)) for L in Ls}
    out = {L: (np.full(len(w), np.nan), np.zeros(len(w))) for L in Ls}
    if x.shape[0] == 0 or len(w) == 0:
        return out
    h = mne.filter.create_filter(None, sfreq, lo, hi, fir_design=fir_design, fir_window=fir_window, phase=phase,
                                 verbose="error")
    half = (len(h) - 1) // 2
    pad = int(round(pad_s * sfreq))
    k = seg.segment_of(w)
    for i in range(len(w)):
        if k[i] < 0:
            continue
        s0, s1 = (int(v) for v in seg.valid_spans[k[i]])
        p0 = max(s0, int(w[i]) - max(nL.values()) - pad)
        p1 = min(int(on[i]), s1)
        if p1 - p0 <= len(h):
            continue
        y = mne.filter.filter_data(x[:, p0:p1], sfreq, lo, hi, method="fir", fir_design=fir_design,
                                   fir_window=fir_window, phase=phase, verbose="error")
        n = y.shape[1]
        z = hilbert(y, N=next_fast_len(n), axis=-1)[:, :n]
        pw = (np.abs(z) ** 2).mean(axis=0)
        if p0 == s0:
            pw[:half] = np.nan
        pw[n - half:] = np.nan                         # truncation at onset: filter edge (FAST/IC: last 0.28 s)
        for L in Ls:
            c0 = max(int(w[i]) - nL[L], s0)
            seg_pw = pw[c0 - p0:int(w[i]) - p0]
            fin = np.isfinite(seg_pw)
            frac = fin.sum() / float(nL[L])
            out[L][1][i] = frac
            if fin.any() and frac >= min_valid_frac - 1e-12:
                out[L][0][i] = seg_pw[fin].mean()
    return out


def context_intervals(win_start: np.ndarray, seg, length_s: float, sfreq: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[max(win_start - L, span start), win_start)`` per window (the same valid span).

    Returns ``(start, stop, in_span)``; windows whose start is not in a valid span get ``start = stop`` and
    ``in_span = False``.
    """
    a = np.asarray(win_start, np.int64)
    k = seg.segment_of(a)
    ok = k >= 0
    span0 = np.where(ok, seg.valid_spans[np.clip(k, 0, None), 0] if len(seg.valid_spans) else a, a)
    nL = int(round(float(length_s) * float(sfreq)))
    st = np.where(ok, np.maximum(a - nL, span0), a)
    return st.astype(np.int64), a, ok


def count_in_intervals(event_samples: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Number of events with ``start <= sample < stop``."""
    ev = np.sort(np.asarray(event_samples, np.int64))
    return (np.searchsorted(ev, np.asarray(stops, np.int64), side="left")
            - np.searchsorted(ev, np.asarray(starts, np.int64), side="left")).astype(np.int64)

