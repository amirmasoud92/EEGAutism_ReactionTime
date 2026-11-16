"""BOSC-type alpha bursts on the continuous cleaned ROI signal and per-trial burst occupancy.

``burst_occ`` = fraction of samples in [-W-50, -250] ms inside a BOSC-type alpha burst (continuous-run 6-cycle Morlet
TFR, IAF +- 2 Hz, background from the run-level aperiodic fit, 95th percentile, >= 2 cycles). Grid: percentile
0.95 / 0.99 x duration 2 / 3 cycles. Parameters: ``configs/bursts.yaml`` (:func:`load_burst_config`).

Licence note: the TFR, robust fit and detection are derived from eBOSC / BOSC (GPL-3.0).

Procedure (one recording)
-------------------------
1. **ROI signal** (:func:`roi_signal`): the mean of the good channels of the posterior ROI (PO7, PO8, PO3, PO4, POz,
   O1, O2, Oz; bad channels dropped, not interpolated) of the cleaned continuous data
   (``load_clean(rec, apply_ica=True, interpolate=False)``). Fewer than ``min_roi_channels`` (5) good ROI channels
   -> every window invalid.
2. **TFR** (:func:`bosc_tf`, :func:`bosc_tf_segmentwise`): the BOSC wavelet of ``BOSC_tf.m`` (complex Morlet,
   ``wavenumber`` = 6 cycles, ``sigma_t = wavenumber / (2 pi f)``, truncated at +-3.6 sigma_t and sampled from exactly
   -3.6 sigma_t as the MATLAB colon does, unit energy), full convolution, centre part kept, power = ``|.|^2``; a port
   checked against the MATLAB code. It runs **separately in each valid span** of
   :class:`~preprocessing.segments.SegmentInfo` (between joins, without rail / flat / unusable spans), so no wavelet
   straddles a join. A convolution near a span edge would need samples beyond the edge; instead of padding, the first
   and last ``ceil(edge_sd * sigma_t * fs)`` samples of each span (default ``edge_sd`` = 3.6, the full half-support of
   the truncated wavelet; 491 ms at 7 Hz, 344 ms at 10 Hz) are set to **NaN** (invalid) at every frequency. Every
   finite TFR value is therefore identical to the TFR of an infinitely long recording.
3. **Background** (:func:`run_spectra`, :func:`fit_background`): the run-level spectrum of the wavelet power of the
   ROI signal on a linear 0.5-Hz grid, 2-30 Hz, averaged over the *background samples*: finite (valid, non-edge) TFR
   samples inside the 1-s chunks in which every good ROI channel is within the 200-uV peak-to-peak limit. Its
   aperiodic component is fitted
   - ``specparam`` (primary; fBOSC-style): specparam 2.0 fixed mode on [2, 30] Hz (peak widths 1-8 Hz, <= 6 peaks,
     minimum height 0.05); ``mp(f) = 10 ** (offset - exponent * log10 f)``. A fit with R^2 < ``min_r2`` or an
     error falls back to the robust fit, and the fallback is recorded;
   - ``robustfit`` (eBOSC ``eBOSC_getThresholds.m`` lines 46-72; cross-check): MATLAB ``robustfit`` in log-log space
     (exact port) after leaving out +- half the wavelet's spectral FWHM around the largest-power frequency in
     ``exclude_peak_hz``.
   The background and the threshold are in the same units (wavelet power of the same signal), so no PSD-to-wavelet
   conversion is needed.
4. **Threshold and its calibration** (:func:`thresholds`): ``pt(f) = chi2.ppf(p, 2) / 2 * mp(f)``. For Gaussian
   background the wavelet power at one time-frequency point is ``mean * chi2(2) / 2`` (exponential), so ``pt`` has
   the nominal exceedance ``1 - p`` **only if ``mp`` estimates the arithmetic mean of the background power**.
   - ``arithmetic`` calibration (primary): the spectrum is the arithmetic mean of linear power. Realised pure-noise
     exceedance about 5 % at p = 0.95 and 1 % at p = 0.99.
   - ``geometric`` calibration (published BOSC / eBOSC / fBOSC): the spectrum is ``10 ** mean(log10 power)``, a
     geometric mean (e^-gamma = 0.561 x the arithmetic mean), used as if arithmetic: about 18.6 % / 7.5 % of
     pure-noise samples exceed the "95th / 99th percentile". Comparison only.
   A *variant* = ``<background>_<calibration>``: ``specparam_arithmetic`` (primary), ``robustfit_arithmetic`` (BOSC
   fit, cross-check), ``robustfit_geometric`` (= published eBOSC), ``specparam_geometric`` (= published fBOSC).
5. **Duration** (:func:`bosc_detect`): per band frequency, samples with power > ``pt``; a run of consecutive
   supra-threshold samples is kept if it lasts ``>= cycles * fs / f`` samples (``N`` cycles *at that frequency*).
   NaN (edge zone, outside spans) is never supra-threshold, so runs never cross an edge zone or a join. Default
   marking = exactly the run. ``matlab_marking=True`` reproduces ``BOSC_detect.m`` (which also marks the first
   sub-threshold sample after a run and counts a run touching the end one sample short), for comparison with MATLAB.
6. **Burst state** of a sample: detected at any band frequency (IAF - 2 ... IAF + 2 Hz, 9 frequencies 0.5 Hz apart,
   both edges included).
7. **burst_occ** of a trial (:func:`trial_burst_features`): the fraction of the samples
   ``[onset + round((-W-50) fs/1000), onset + round(-250 fs/1000))`` (half-open) in a burst state. NaN when the
   window, widened on each side by the *guard* = edge zone at IAF - 2 Hz + the longest duration rule of the grid at
   IAF - 2 Hz, does not lie inside one valid span (``burst_edge_ok``), or when fewer than 5 good ROI channels exist.
   With this guard any supra-threshold run that touches a window sample and reaches an edge zone is already long
   enough to pass every duration rule, so the burst state of every valid window equals that of an infinitely long
   recording. **BAD* annotations**: the same guarded window must not overlap a ``BAD*`` annotation of the cleaned
   data (``BAD_icafit``, ``BAD_short``, any other ``BAD*``), i.e. a ``BAD*`` span is treated like a span edge,
   because the wavelet power and the duration rule reach beyond the window. ``BAD*`` samples are also left out of the
   background spectrum (:func:`exclude_spans`). The predictor-set rules (lookback, edge-validity and artefact rules)
   are applied downstream from the trial table.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

#: Posterior ROI and its minimum number of good channels.
SPEC_ROI: tuple[str, ...] = ("PO7", "PO8", "PO3", "PO4", "POz", "O1", "O2", "Oz")
SPEC_MIN_ROI_CHANNELS = 5
#: Grid (percentile, duration in cycles); the first entry is the primary setting.
CALIBRATION_GRID: tuple[tuple[float, float], ...] = ((0.95, 2.0), (0.95, 3.0), (0.99, 2.0), (0.99, 3.0))
PRIMARY_SETTING: tuple[float, float] = (0.95, 2.0)
#: End of the burst window relative to onset (ms).
BURST_WINDOW_STOP_MS = -250.0
#: Start of the burst window relative to ``-W`` (ms): the window starts at ``-W - 50`` ms.
BURST_WINDOW_START_OFFSET_MS = -50.0
#: Truncation of the BOSC wavelet (+- sigma_t; ``BOSC_tf.m``).
BOSC_TRUNC_SD = 3.6
BACKGROUNDS = ("specparam", "robustfit")
CALIBRATIONS = ("arithmetic", "geometric")
PRIMARY_VARIANT = "specparam_arithmetic"
#: Column tag of each variant (``""`` = primary, no tag).
VARIANT_TAGS = {"specparam_arithmetic": "", "robustfit_arithmetic": "rf", "robustfit_geometric": "bosc",
                "specparam_geometric": "fbosc"}


def split_variant(variant: str) -> tuple[str, str]:
    """``"robustfit_geometric"`` -> ``("robustfit", "geometric")`` (checked against the known variants)."""
    if variant not in VARIANT_TAGS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANT_TAGS)}")
    b, c = variant.split("_")
    return b, c


def setting_suffix(percentile: float, duration_cycles: float) -> str:
    """Column suffix of a grid setting, e.g. ``p95_d2``."""
    return f"p{int(round(percentile * 100))}_d{duration_cycles:g}"


def occ_column(percentile: float, duration_cycles: float, variant: str = PRIMARY_VARIANT) -> str:
    """Column name of burst occupancy.

    Primary variant: ``burst_occ`` for the primary setting, else ``burst_occ_<p>_<d>`` (``burst_occ_p99_d3``).
    Cross-check variants carry their tag: ``burst_occ_rf_p95_d2``, ``burst_occ_bosc_p95_d2``.
    """
    split_variant(variant)
    s = setting_suffix(percentile, duration_cycles)
    tag = VARIANT_TAGS[variant]
    if tag:
        return f"burst_occ_{tag}_{s}"
    if (percentile, duration_cycles) == PRIMARY_SETTING:
        return "burst_occ"
    return f"burst_occ_{s}"


@dataclass(frozen=True)
class BurstConfig:
    """Parameters of the burst detector (defaults = primary setting; ``configs/bursts.yaml``)."""

    roi: tuple[str, ...] = SPEC_ROI
    min_roi_channels: int = SPEC_MIN_ROI_CHANNELS
    wavenumber: float = 6.0
    band_half_width_hz: float = 2.0
    band_step_hz: float = 0.5
    edge_sd: float = BOSC_TRUNC_SD                              # TFR edge zone (x sigma_t) set to NaN in each span
    method: str = "fft"                                         # convolution: "fft" | "direct" (np.convolve)
    bg_freqs: tuple[float, float, float] = (2.0, 30.0, 0.5)     # start, stop (inclusive), step; linear grid
    bg_fit_range: tuple[float, float] = (2.0, 30.0)
    aperiodic_mode: str = "fixed"
    peak_width_limits: tuple[float, float] = (1.0, 8.0)
    max_n_peaks: int = 6
    min_peak_height: float = 0.05
    min_r2: float = 0.9                                         # specparam fit below this -> robustfit fallback
    exclude_peak_hz: tuple[float, float] = (6.0, 14.0)          # robustfit: peak search range (eBOSC excludePeak)
    artefact_ptp_uv: float = 200.0                              # background chunks (ROI peak-to-peak limit)
    artefact_chunk_s: float = 1.0
    variants: tuple[str, ...] = (PRIMARY_VARIANT,)              # first = primary
    grid: tuple[tuple[float, float], ...] = CALIBRATION_GRID   # (percentile, cycles); first = primary
    matlab_marking: bool = False                                # True: BOSC_detect.m marking (MATLAB comparison)
    window_start_offset_ms: float = BURST_WINDOW_START_OFFSET_MS
    window_stop_ms: float = BURST_WINDOW_STOP_MS
    task_window_ms: tuple[tuple[str, float], ...] = (("AVSRT", 800.0), ("FAST", 500.0), ("IC", 500.0))

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError("at least one variant is required")
        for v in self.variants:
            split_variant(v)
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("duplicate variants")
        if self.aperiodic_mode != "fixed":
            raise ValueError("only the fixed aperiodic mode is supported for the threshold")
        if self.method not in ("fft", "direct"):
            raise ValueError(f"method must be 'fft' or 'direct', got {self.method!r}")
        if not self.grid:
            raise ValueError("empty grid")
        for p, d in self.grid:
            if not (0 < p < 1 and d > 0):
                raise ValueError(f"bad grid entry {(p, d)}")
        if self.edge_sd <= 0:
            raise ValueError("edge_sd must be > 0")
        if self.window_stop_ms > 0:
            raise ValueError("the burst window must end before onset")

    @property
    def primary_variant(self) -> str:
        return self.variants[0]

    def window_ms(self, task: str) -> float:
        """W of a task (800 ms AVSRT, 500 ms FAST / IC)."""
        d = dict(self.task_window_ms)
        if task not in d:
            raise KeyError(f"no task window for {task!r}")
        return float(d[task])

    def bg_freq_grid(self) -> np.ndarray:
        a, b, s = self.bg_freqs
        n = int(round((b - a) / s)) + 1
        return a + s * np.arange(n)

    def band_freqs(self, iaf: float) -> np.ndarray:
        """IAF +- half width, both edges included (step ``band_step_hz``)."""
        if not np.isfinite(iaf) or iaf <= self.band_half_width_hz:
            raise ValueError(f"invalid IAF {iaf}")
        n = int(round(2 * self.band_half_width_hz / self.band_step_hz)) + 1
        return np.linspace(iaf - self.band_half_width_hz, iaf + self.band_half_width_hz, n)

    def to_dict(self) -> dict:
        def conv(v):
            if isinstance(v, tuple):
                return [conv(x) for x in v]
            return v

        return {f.name: conv(getattr(self, f.name)) for f in fields(self)}


# =============================================================================== config file
_TUPLE_FIELDS = {"roi", "bg_freqs", "bg_fit_range", "peak_width_limits", "exclude_peak_hz", "variants"}


def burst_config_from_mapping(d: Mapping, where: str = "bursts") -> BurstConfig:
    """Build a :class:`BurstConfig` from the ``bursts:`` mapping (unknown keys are an error)."""
    allowed = {f.name for f in fields(BurstConfig)} | {"version"}
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    kw = {}
    for k, v in d.items():
        if k == "version":
            continue
        if k in _TUPLE_FIELDS:
            v = tuple(v)
        elif k == "grid":
            v = tuple((float(p), float(c)) for p, c in v)
        elif k == "task_window_ms":
            v = tuple((str(t), float(w)) for t, w in dict(v).items())
        kw[k] = v
    cfg = BurstConfig(**kw)
    if tuple(cfg.roi) != SPEC_ROI or cfg.min_roi_channels != SPEC_MIN_ROI_CHANNELS:
        raise ValueError(f"{where}: roi / min_roi_channels must equal the posterior ROI {SPEC_ROI} "
                         f"(>= {SPEC_MIN_ROI_CHANNELS} good channels)")
    return cfg


def load_burst_config(path: str | os.PathLike[str] | None = None, root: str | os.PathLike[str] | None = None
                      ) -> BurstConfig:
    """Read ``configs/bursts.yaml`` (section ``bursts:``)."""
    from utils.config import read_yaml

    if path is None:
        base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        path = base / "configs" / "bursts.yaml"
    raw = read_yaml(path)
    if "bursts" not in raw:
        raise ValueError(f"{path}: missing 'bursts:' section")
    sec = raw["bursts"]
    return burst_config_from_mapping(sec, f"{path}:bursts")


# =============================================================================== TFR (BOSC_tf.m)
def bosc_wavelet(freq: float, fs: float, wavenumber: float = 6.0) -> np.ndarray:
    """The complex BOSC Morlet wavelet at ``freq`` (``BOSC_tf.m`` lines 39-44).

    ``st = 1 / (2 pi f / wavenumber)``; samples ``t = -3.6 st : 1/fs : 3.6 st`` (MATLAB colon, so the first sample is
    exactly at -3.6 st and the wavelet is generally not centred on a sample); ``A = 1 / sqrt(st sqrt(pi))``.
    """
    st = 1.0 / (2 * np.pi * (freq / wavenumber))
    a_ = 1.0 / np.sqrt(st * np.sqrt(np.pi))
    d = 1.0 / fs
    n_pts = int(np.floor((2 * BOSC_TRUNC_SD * st) / d * (1 + 1e-12))) + 1
    t = -BOSC_TRUNC_SD * st + np.arange(n_pts) * d
    return a_ * np.exp(-t ** 2 / (2 * st ** 2)) * np.exp(1j * 2 * np.pi * freq * t)


def wavelet_sd_s(freq: float | np.ndarray, wavenumber: float = 6.0) -> float | np.ndarray:
    """Temporal SD of the Morlet envelope, ``sigma_t = wavenumber / (2 pi f)`` (s)."""
    return wavenumber / (2 * np.pi * np.asarray(freq, float))


def wavelet_halfwidth_s(freq: float | np.ndarray, wavenumber: float = 6.0) -> float | np.ndarray:
    """Half-support (s) of the truncated BOSC wavelet: ``3.6 sigma_t``."""
    return BOSC_TRUNC_SD * wavelet_sd_s(freq, wavenumber)


def wavelet_halfwidth_samples(freq: float | np.ndarray, fs: float, wavenumber: float = 6.0) -> np.ndarray:
    """Half-support of the truncated wavelet in samples (ceil)."""
    return np.ceil(np.asarray(wavelet_halfwidth_s(freq, wavenumber)) * fs).astype(np.int64)


def edge_samples(freq: float | np.ndarray, fs: float, wavenumber: float = 6.0,
                 edge_sd: float = BOSC_TRUNC_SD) -> np.ndarray:
    """TFR edge zone (samples) at ``freq``: ``ceil(edge_sd * sigma_t * fs)``.

    With ``edge_sd = 3.6`` this is >= the one-sided extent ``floor(L / 2)`` of the sampled wavelet, so no finite
    TFR value uses a sample beyond the span.
    """
    return np.ceil(edge_sd * np.asarray(wavelet_sd_s(freq, wavenumber)) * fs - 1e-9).astype(np.int64)


def _conv_power(sig: np.ndarray, m: np.ndarray, method: str) -> np.ndarray:
    if method == "direct":
        y = np.convolve(sig, m)
    else:
        from scipy.signal import fftconvolve

        y = fftconvolve(sig, m, mode="full")
    y = np.abs(y) ** 2
    L = len(m)
    start = int(np.ceil(L / 2)) - 1               # MATLAB ceil(L/2) (1-based)
    stop = len(y) - int(np.floor(L / 2))          # MATLAB end - floor(L/2) (inclusive, 1-based)
    return y[start:stop]


def bosc_tf(sig: np.ndarray, freqs: Sequence[float] | np.ndarray, fs: float, wavenumber: float = 6.0,
            method: str = "fft") -> np.ndarray:
    """Wavelet power ``(n_freqs, n_times)`` of a 1-D signal (port of ``BOSC_tf.m``; zero-padded at the ends).

    ``method="direct"`` uses ``np.convolve`` (line-by-line port); ``"fft"`` uses
    ``scipy.signal.fftconvolve`` (same result to ~1e-12 relative, much faster on long runs).
    """
    sig = np.asarray(sig, dtype=float)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    out = np.empty((len(freqs), len(sig)))
    for i, f in enumerate(freqs):
        out[i] = _conv_power(sig, bosc_wavelet(f, fs, wavenumber), method)
    return out


def bosc_tf_segmentwise(sig: np.ndarray, spans: np.ndarray, freqs: Sequence[float] | np.ndarray, fs: float,
                        wavenumber: float = 6.0, method: str = "fft",
                        edge_sd: float | None = BOSC_TRUNC_SD) -> np.ndarray:
    """:func:`bosc_tf` run separately in each half-open span.

    Samples outside every span are NaN; with ``edge_sd`` (default) the first and last :func:`edge_samples` of each
    span are NaN too (edge zone: the wavelet would reach beyond the span). ``edge_sd=None`` keeps the zero-padded
    values (BOSC behaviour; for comparisons only).
    """
    sig = np.asarray(sig, dtype=float)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    out = np.full((len(freqs), len(sig)), np.nan)
    eh = None if edge_sd is None else edge_samples(freqs, fs, wavenumber, edge_sd)
    for a, b in _spans(spans):
        if b <= a:
            continue
        if eh is not None and b - a <= 2 * int(eh.min()):
            continue                                  # span shorter than two edge zones at every frequency
        p = bosc_tf(sig[a:b], freqs, fs, wavenumber, method)
        if eh is not None:
            for i, h in enumerate(eh):
                h = int(h)
                if b - a <= 2 * h:
                    p[i] = np.nan
                else:
                    p[i, :h] = np.nan
                    p[i, b - a - h:] = np.nan
        out[:, a:b] = p
    return out


def tfr_valid_mask(n_times: int, spans: np.ndarray, freq: float, fs: float, wavenumber: float = 6.0,
                   edge_sd: float = BOSC_TRUNC_SD) -> np.ndarray:
    """True where the TFR at ``freq`` is valid (inside a span and outside its edge zones)."""
    h = int(edge_samples(freq, fs, wavenumber, edge_sd))
    m = np.zeros(int(n_times), bool)
    for a, b in _spans(spans):
        if b - a > 2 * h:
            m[a + h:b - h] = True
    return m


# =============================================================================== robust fit (MATLAB robustfit)
def matlab_robustfit(x: np.ndarray, y: np.ndarray, tune: float = 4.685, iterlim: int = 50) -> np.ndarray:
    """MATLAB ``robustfit`` (bisquare, leverage-adjusted residuals, ``madsigma``); returns ``[intercept, slope]``.

    Port of MATLAB ``robustfit`` (matches MATLAB to 1e-15).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    X = np.column_stack([np.ones_like(x), x])
    n, p = X.shape
    Q, R = np.linalg.qr(X)
    b = np.linalg.solve(R, Q.T @ y)
    h = np.minimum(0.9999, np.sum(Q * Q, axis=1))
    adjfactor = 1.0 / np.sqrt(1 - h)
    D = np.sqrt(np.finfo(float).eps)
    tiny_s = 1e-6 * np.std(y, ddof=1)
    if tiny_s == 0:
        tiny_s = 1.0
    b0 = np.zeros_like(b)
    it = 0
    while (it == 0 or np.any(np.abs(b - b0) > D * np.maximum(np.abs(b), np.abs(b0)))) and it < iterlim:
        it += 1
        r = y - X @ b
        radj = r * adjfactor
        rs = np.sort(np.abs(radj))
        s = np.median(rs[max(1, p) - 1:]) / 0.6745
        s = max(s, tiny_s)
        u = radj / (s * tune)
        w = (np.abs(u) < 1) * (1 - u ** 2) ** 2
        b0 = b
        sw = np.sqrt(w)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return b


def robust_background_fit(freqs: np.ndarray, log10_spec: np.ndarray, wavenumber: float = 6.0,
                          exclude_peak_hz: tuple[float, float] | None = (6.0, 14.0),
                          peak_spectrum: np.ndarray | None = None) -> dict:
    """eBOSC background (``eBOSC_getThresholds.m`` lines 46-72) on a given log10 spectrum.

    The frequency with the largest power inside ``exclude_peak_hz`` is found and +- half the wavelet's spectral FWHM
    around it, ``F +- (2 / wavenumber) F / 2``, is left out; then ``robustfit(log10 F, log10_spec)``. eBOSC searches
    the peak on the *arithmetic* mean power (``max(mean(BG(freqidx,:),2))``) whatever is fitted; pass that spectrum
    as ``peak_spectrum`` (default: ``log10_spec``).
    Returns ``{"offset", "exponent", "freq_keep", "peak_freq"}`` with ``log10 mp = offset - exponent log10 f``.
    """
    freqs = np.asarray(freqs, float)
    y = np.asarray(log10_spec, float)
    ps = y if peak_spectrum is None else np.asarray(peak_spectrum, float)
    keep = np.isfinite(y)
    pk = np.nan
    if exclude_peak_hz is not None:
        lo, hi = exclude_peak_hz
        idx = np.flatnonzero((freqs >= lo) & (freqs <= hi) & np.isfinite(y))
        if idx.size:
            ipos = idx[np.argmax(ps[idx])]
            pk = float(freqs[ipos])
            half = (2.0 / wavenumber) * pk / 2
            keep &= ~((freqs >= pk - half) & (freqs <= pk + half))
    b = matlab_robustfit(np.log10(freqs[keep]), y[keep])
    return {"offset": float(b[0]), "exponent": float(-b[1]), "freq_keep": keep, "peak_freq": pk}


# =============================================================================== background
@dataclass
class Background:
    """Run-level aperiodic background of the ROI signal (one variant)."""

    variant: str                      # requested variant, e.g. "specparam_arithmetic"
    method: str                       # fit actually used ("specparam" or "robustfit" after a fallback)
    calibration: str                  # "arithmetic" | "geometric"
    freqs: np.ndarray                 # background grid (Hz)
    spectrum: np.ndarray              # fitted run-level spectrum (linear power; arithmetic or geometric mean)
    offset: float
    exponent: float
    r2: float = np.nan
    error: float = np.nan
    n_samples: np.ndarray | None = None   # background samples per frequency
    fallback_reason: str = ""
    peaks: np.ndarray | None = None       # specparam peaks [cf, pw, bw]

    def mp(self, f: np.ndarray | float) -> np.ndarray:
        """Background power at ``f`` (linear, wavelet-power units)."""
        return 10.0 ** (self.offset - self.exponent * np.log10(np.asarray(f, float)))

    def summary(self) -> dict:
        return {"variant": self.variant, "method": self.method, "calibration": self.calibration,
                "offset": self.offset, "exponent": self.exponent, "r2": self.r2, "error": self.error,
                "fallback_reason": self.fallback_reason,
                "n_bg_samples_min": int(np.min(self.n_samples)) if self.n_samples is not None else None}


def fit_background(freqs: np.ndarray, spectrum: np.ndarray, cfg: BurstConfig, variant: str,
                   n_samples: np.ndarray | None = None, peak_spectrum: np.ndarray | None = None) -> Background:
    """Aperiodic fit of a run-level spectrum (linear power) for ``variant``.

    ``specparam`` variants fall back to the robust fit when specparam errors or R^2 < ``cfg.min_r2`` (recorded in
    ``fallback_reason``). ``peak_spectrum`` (linear; default ``spectrum``) is where the robust fit searches the peak
    to exclude (eBOSC: the arithmetic mean power).
    """
    bgm, cal = split_variant(variant)
    freqs = np.asarray(freqs, float)
    spectrum = np.asarray(spectrum, float)
    reason = ""
    if bgm == "specparam":
        try:
            from specparam import SpectralModel

            sm = SpectralModel(peak_width_limits=list(cfg.peak_width_limits), max_n_peaks=cfg.max_n_peaks,
                               min_peak_height=cfg.min_peak_height, aperiodic_mode=cfg.aperiodic_mode,
                               verbose=False)
            sm.fit(freqs, spectrum, list(cfg.bg_fit_range))
            ap = np.asarray(sm.get_params("aperiodic"), float).ravel()
            met = sm.results.metrics.results
            r2 = float(met.get("gof_rsquared", np.nan))
            err = float(met.get("error_mae", np.nan))
            if ap.size == 2 and np.all(np.isfinite(ap)) and np.isfinite(r2) and r2 >= cfg.min_r2:
                pk = np.atleast_2d(sm.get_params("periodic"))
                return Background(variant, "specparam", cal, freqs, spectrum, float(ap[0]), float(ap[1]), r2, err,
                                  n_samples, "", pk)
            reason = f"specparam r2 {r2:.3f} < {cfg.min_r2}" if np.isfinite(r2) else "specparam fit not finite"
        except Exception as e:                                          # noqa: BLE001 - fallback is recorded
            reason = f"specparam failed: {type(e).__name__}: {e}"[:200]
    sel = (freqs >= cfg.bg_fit_range[0]) & (freqs <= cfg.bg_fit_range[1]) & (spectrum > 0)
    pks = spectrum if peak_spectrum is None else np.asarray(peak_spectrum, float)
    ly = np.log10(spectrum[sel])
    rf = robust_background_fit(freqs[sel], ly, cfg.wavenumber, cfg.exclude_peak_hz, peak_spectrum=pks[sel])
    resid = ly - (rf["offset"] - rf["exponent"] * np.log10(freqs[sel]))
    k = rf["freq_keep"]
    ss = np.sum((ly[k] - ly[k].mean()) ** 2)
    r2 = float(1 - np.sum(resid[k] ** 2) / ss) if ss > 0 else np.nan
    return Background(variant, "robustfit", cal, freqs, spectrum, rf["offset"], rf["exponent"], r2,
                      float(np.mean(np.abs(resid[k]))), n_samples, reason, None)


def artefact_chunk_mask(roi_data_v: np.ndarray, spans: np.ndarray, fs: float, ptp_uv: float = 200.0,
                        chunk_s: float = 1.0) -> np.ndarray:
    """True on samples of 1-s chunks (inside each span, from its start) where every ROI channel is within ``ptp_uv``.

    A trailing partial chunk is judged on its own samples. Samples outside the spans are False.
    """
    x = np.asarray(roi_data_v, float)
    if x.ndim == 1:
        x = x[None]
    n = x.shape[1]
    ok = np.zeros(n, bool)
    c = max(1, int(round(chunk_s * fs)))
    lim = ptp_uv * 1e-6
    for a, b in _spans(spans):
        for s in range(a, b, c):
            e = min(s + c, b)
            if np.all(np.ptp(x[:, s:e], axis=1) <= lim):
                ok[s:e] = True
    return ok


@dataclass
class RunSpectra:
    """Run-level wavelet spectra of the ROI signal on the background grid."""

    freqs: np.ndarray
    arithmetic: np.ndarray            # mean of linear power over the background samples
    geometric: np.ndarray             # 10 ** mean(log10 power)
    n_samples: np.ndarray


def run_spectra(sig: np.ndarray, spans: np.ndarray, fs: float, cfg: BurstConfig,
                sample_mask: np.ndarray | None = None) -> RunSpectra:
    """Run-level spectra of a 1-D signal: valid (non-edge) TFR samples inside ``sample_mask`` (default: all).

    The TFR is computed one frequency at a time (memory: one row).
    """
    sig = np.asarray(sig, float)
    freqs = cfg.bg_freq_grid()
    base = np.ones(len(sig), bool) if sample_mask is None else np.asarray(sample_mask, bool)
    s_lin = np.zeros(len(freqs))
    s_log = np.zeros(len(freqs))
    cnt = np.zeros(len(freqs), np.int64)
    for i, f in enumerate(freqs):
        row = bosc_tf_segmentwise(sig, spans, [f], fs, cfg.wavenumber, cfg.method, cfg.edge_sd)[0]
        v = row[base & np.isfinite(row)]
        v = v[v > 0]
        cnt[i] = v.size
        if v.size:
            s_lin[i] = v.mean()
            s_log[i] = np.log10(v).mean()
    if np.any(cnt == 0):
        raise ValueError("no background samples at some frequency (run too short or fully artefactual)")
    return RunSpectra(freqs, s_lin, 10.0 ** s_log, cnt)


def run_backgrounds(sig: np.ndarray, spans: np.ndarray, fs: float, cfg: BurstConfig,
                    sample_mask: np.ndarray | None = None, variants: Iterable[str] | None = None,
                    spectra: RunSpectra | None = None) -> dict[str, Background]:
    """Aperiodic background of a 1-D signal for each variant (default ``cfg.variants``)."""
    sp = spectra if spectra is not None else run_spectra(sig, spans, fs, cfg, sample_mask)
    out = {}
    for v in (list(variants) if variants is not None else list(cfg.variants)):
        _, cal = split_variant(v)
        spec = sp.arithmetic if cal == "arithmetic" else sp.geometric
        out[v] = fit_background(sp.freqs, spec, cfg, v, n_samples=sp.n_samples, peak_spectrum=sp.arithmetic)
    return out


def thresholds(band_freqs: np.ndarray, bg: Background, percentile: float, duration_cycles: float, fs: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """``pt = chi2.ppf(p, 2) / 2 * mp(f)`` and ``dt = cycles * fs / f`` (samples) (``eBOSC_getThresholds.m`` 76-78)."""
    from scipy.stats import chi2

    f = np.asarray(band_freqs, float)
    return chi2.ppf(percentile, 2) * bg.mp(f) / 2.0, duration_cycles * fs / f


# =============================================================================== detection (BOSC_detect.m)
def bosc_detect(power: np.ndarray, powthresh: float, durthresh: float, matlab_marking: bool = False) -> np.ndarray:
    """Detected samples (bool) of one frequency row (vectorised ``BOSC_detect.m``).

    A run of supra-threshold samples ``[s, e)`` (``power > powthresh``; NaN is never above) is kept if its duration
    ``e - s >= durthresh`` samples, and ``[s, e)`` is marked (default). With ``matlab_marking`` (identical to the
    MATLAB code) the duration is ``e - s`` for an interior run but
    ``e - s - 1`` for a run that reaches the last sample, and ``[s, min(e + 1, n))`` is marked (the first
    sub-threshold sample after a run is marked too).
    """
    b = np.asarray(power, float)
    n = b.size
    x = b > powthresh
    out = np.zeros(n, bool)
    if not x.any():
        return out
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    if matlab_marking:
        h2 = np.where(e < n, e + 1, n)            # MATLAB H(2) (1-based) = end of the marked slice (0-based)
        keep = (h2 - (s + 1)) >= durthresh
        stops = h2
    else:
        keep = (e - s) >= durthresh
        stops = e
    if keep.any():
        # cumulative-sum fill of the kept runs
        delta = np.zeros(n + 1, np.int64)
        np.add.at(delta, s[keep], 1)
        np.add.at(delta, stops[keep], -1)
        out = np.cumsum(delta[:-1]) > 0
    return out


def detect_segmentwise(power: np.ndarray, spans: np.ndarray, pt: np.ndarray, dt: np.ndarray,
                       matlab_marking: bool = False) -> np.ndarray:
    """:func:`bosc_detect` for every row of ``power`` ``(n_freqs, n_times)``, separately in each span."""
    p = np.atleast_2d(power)
    out = np.zeros(p.shape, bool)
    for a, b in _spans(spans):
        for i in range(p.shape[0]):
            out[i, a:b] = bosc_detect(p[i, a:b], pt[i], dt[i], matlab_marking)
    return out


# =============================================================================== ROI signal
def roi_signal(data_v: np.ndarray, ch_names: Sequence[str], bads: Iterable[str] = (),
               roi: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Mean of the good ROI channels: ``(signal (n_times,), good channel names)``.

    ``data_v`` is ``(n_channels, n_times)`` with rows named ``ch_names``. Bad channels are dropped. With no good
    ROI channel the signal is all NaN.
    """
    roi = tuple(roi) if roi is not None else BurstConfig().roi
    bad = set(bads)
    names = list(ch_names)
    good = [c for c in roi if c in names and c not in bad]
    x = np.atleast_2d(np.asarray(data_v, float))
    if not good:
        return np.full(x.shape[1], np.nan), []
    return x[[names.index(c) for c in good]].mean(axis=0), good


# =============================================================================== one recording
@dataclass
class RunBursts:
    """Burst state of the ROI signal of one recording."""

    channels: list[str]               # good ROI channels averaged into the signal
    fs: float
    iaf: float
    band_freqs: np.ndarray
    spans: np.ndarray
    #: (variant, percentile, duration) -> bool (n_times,): burst at any band frequency
    state: dict[tuple[str, float, float], np.ndarray]
    #: variant -> Background
    backgrounds: dict[str, Background]
    cfg: BurstConfig
    #: (variant, percentile) -> fraction of valid band TF points with power > pt (before the duration rule)
    above_fraction: dict[tuple[str, float], float] = field(default_factory=dict)
    #: variant -> mean over valid samples of band power / mp, per band frequency
    band_snr: dict[str, np.ndarray] = field(default_factory=dict)
    bg_sample_fraction: float = np.nan
    valid_sample_fraction: float = np.nan     # in-span samples with a valid TFR at IAF - 2 Hz

    @property
    def n_roi(self) -> int:
        return len(self.channels)

    def guard_samples(self) -> int:
        """Window guard: edge zone at the lowest band frequency + the longest duration rule there."""
        f_lo = float(np.min(self.band_freqs))
        dmax = max(d for _, d in self.cfg.grid)
        return int(edge_samples(f_lo, self.fs, self.cfg.wavenumber, self.cfg.edge_sd)) + \
            int(np.ceil(dmax * self.fs / f_lo))


def detect_run_bursts(sig: np.ndarray, fs: float, spans: np.ndarray, iaf: float, cfg: BurstConfig | None = None,
                      *, channels: Sequence[str] = ("roi",), bg_mask: np.ndarray | None = None,
                      backgrounds: Mapping[str, Background] | None = None) -> RunBursts:
    """Burst detection on the continuous ROI signal of one recording.

    Parameters
    ----------
    sig
        ``(n_times,)`` ROI signal in volts on the full recording sample axis (:func:`roi_signal`).
    spans
        Valid spans (``SegmentInfo.valid_spans``): TFR, background and detection run separately in each.
    iaf
        Individual alpha frequency (Hz); band = IAF +- ``cfg.band_half_width_hz``.
    channels
        Names of the good ROI channels in ``sig`` (their number is the ROI channel count).
    bg_mask
        Samples allowed in the background (default: all; the runner passes :func:`artefact_chunk_mask` of the
        good ROI channels).
    backgrounds
        Precomputed backgrounds per variant (:func:`run_backgrounds`); computed here otherwise.
    """
    cfg = cfg or BurstConfig()
    x = np.asarray(sig, float)
    if x.ndim != 1:
        raise ValueError("sig must be 1-D (the ROI signal)")
    spans = _spans(spans)
    n = x.size
    in_span = np.zeros(n, bool)
    for a, b in spans:
        in_span[a:b] = True
    if bg_mask is None:
        bg_mask = in_span
    bgs = dict(backgrounds) if backgrounds is not None else run_backgrounds(x, spans, fs, cfg, bg_mask)
    missing = [v for v in cfg.variants if v not in bgs]
    if missing:
        raise ValueError(f"backgrounds missing for {missing}")
    bf = cfg.band_freqs(iaf)
    pw = bosc_tf_segmentwise(x, spans, bf, fs, cfg.wavenumber, cfg.method, cfg.edge_sd)
    fin = np.isfinite(pw)
    state: dict[tuple[str, float, float], np.ndarray] = {}
    above: dict[tuple[str, float], float] = {}
    snr: dict[str, np.ndarray] = {}
    for v in cfg.variants:
        bg = bgs[v]
        mp = bg.mp(bf)
        with np.errstate(invalid="ignore"):
            snr[v] = np.array([np.nanmean(pw[i] / mp[i]) if fin[i].any() else np.nan for i in range(len(bf))])
        for p in sorted({p for p, _ in cfg.grid}):
            pt0, _ = thresholds(bf, bg, p, 1.0, fs)
            with np.errstate(invalid="ignore"):
                above[(v, p)] = float(np.sum(pw > pt0[:, None]) / fin.sum()) if fin.any() else np.nan
        for p, d in cfg.grid:
            pt, dt = thresholds(bf, bg, p, d, fs)
            state[(v, p, d)] = detect_segmentwise(pw, spans, pt, dt, cfg.matlab_marking).any(axis=0)
    frac_bg = float(np.mean(bg_mask[in_span])) if in_span.any() else np.nan
    frac_valid = float(np.mean(fin[0][in_span])) if in_span.any() else np.nan
    return RunBursts(list(channels), fs, float(iaf), bf, spans, state, bgs, cfg, above, snr, frac_bg, frac_valid)


# =============================================================================== trial windows
def burst_window_samples(onsets: np.ndarray, window_ms: float, fs: float, stop_ms: float = BURST_WINDOW_STOP_MS,
                         start_offset_ms: float = BURST_WINDOW_START_OFFSET_MS) -> tuple[np.ndarray, np.ndarray]:
    """Half-open burst windows ``[onset + round((-W + start_offset) fs/1000), onset + round(stop_ms fs/1000))``.

    ``window_ms`` is W (800 for AVSRT, 500 for FAST / IC; 500 / 1000 / 400 for the window-length variants);
    default ``start_offset_ms`` = -50 gives the window [-W-50, -250) ms.
    """
    on = np.asarray(onsets, np.int64)
    a = on + int(round((-window_ms + start_offset_ms) * fs / 1000.0))
    b = on + int(round(stop_ms * fs / 1000.0))
    if np.any(b <= a):
        raise ValueError("empty burst window (W + 50 must exceed 250 ms)")
    return a, b


def window_inside_spans(starts: np.ndarray, stops: np.ndarray, spans: np.ndarray, guard: int = 0) -> np.ndarray:
    """True where ``[start - guard, stop + guard)`` lies inside one span."""
    st = np.asarray(starts, np.int64) - guard
    sp = np.asarray(stops, np.int64) + guard
    ok = np.zeros(st.shape, bool)
    for a, b in _spans(spans):
        ok |= (st >= a) & (sp <= b)
    return ok


def burst_occupancy(state: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Fraction of window samples in a burst.

    ``state`` ``(n_times,)`` -> ``(n_windows,)``; ``(n_rows, n_times)`` -> ``(n_windows, n_rows)``. Windows outside
    ``[0, n_times)`` give NaN.
    """
    s0 = np.asarray(state, bool)
    s = np.atleast_2d(s0)
    n = s.shape[1]
    cs = np.concatenate([np.zeros((s.shape[0], 1), np.int64), np.cumsum(s, axis=1, dtype=np.int64)], axis=1)
    a = np.asarray(starts, np.int64)
    b = np.asarray(stops, np.int64)
    ok = (a >= 0) & (b <= n) & (b > a)
    out = np.full((len(a), s.shape[0]), np.nan)
    if ok.any():
        out[ok] = ((cs[:, b[ok]] - cs[:, a[ok]]) / (b[ok] - a[ok])).T
    return out[:, 0] if s0.ndim == 1 else out


def window_overlaps_spans(starts: np.ndarray, stops: np.ndarray, spans: np.ndarray | None, guard: int = 0
                          ) -> np.ndarray:
    """True where ``[start - guard, stop + guard)`` intersects any half-open span of ``spans`` (e.g. ``BAD*``)."""
    st = np.asarray(starts, np.int64) - guard
    sp = np.asarray(stops, np.int64) + guard
    out = np.zeros(st.shape, bool)
    if spans is None:
        return out
    for a, b in np.asarray(spans, np.int64).reshape(-1, 2):
        out |= (st < b) & (sp > a)
    return out


def exclude_spans(mask: np.ndarray, spans: np.ndarray | None) -> np.ndarray:
    """Copy of the boolean sample ``mask`` with the half-open ``spans`` (e.g. ``BAD*`` annotations) set False."""
    out = np.asarray(mask, bool).copy()
    if spans is not None:
        n = out.size
        for a, b in np.asarray(spans, np.int64).reshape(-1, 2):
            out[max(0, int(a)):min(n, int(b))] = False
    return out


def trial_burst_features(rb: RunBursts, onsets: np.ndarray, windows_ms: Mapping[str, float], *,
                         bad_spans: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-trial burst occupancy columns for each window, variant and grid setting.

    Per window name ``w`` (``"primary"`` gets no suffix, others the suffix ``__<w>``): one column per variant x grid
    setting (:func:`occ_column`), plus ``burst_edge_ok`` (window + guard inside one span) and ``burst_valid``
    (``burst_edge_ok``, >= ``min_roi_channels`` good ROI channels and the window + guard overlaps no span
    of ``bad_spans``: the ``BAD*`` annotations of the cleaned data). ``burst_n_roi`` is the channel count.
    Invalid windows are NaN in every occupancy column.
    """
    guard = rb.guard_samples()
    out: dict[str, np.ndarray] = {}
    enough = rb.n_roi >= rb.cfg.min_roi_channels
    n_on = len(np.asarray(onsets))
    for wname, wms in windows_ms.items():
        sfx = "" if wname == "primary" else f"__{wname}"
        a, b = burst_window_samples(onsets, wms, rb.fs, rb.cfg.window_stop_ms, rb.cfg.window_start_offset_ms)
        edge_ok = window_inside_spans(a, b, rb.spans, guard)
        valid = edge_ok & enough & ~window_overlaps_spans(a, b, bad_spans, guard)
        out[f"burst_edge_ok{sfx}"] = edge_ok
        out[f"burst_valid{sfx}"] = valid
        for (v, p, d), st in rb.state.items():
            occ = burst_occupancy(st, a, b)
            out[f"{occ_column(p, d, v)}{sfx}"] = np.where(valid, occ, np.nan)
    out["burst_n_roi"] = np.full(n_on, rb.n_roi, np.int64)
    return out


# =============================================================================== helpers
def _spans(spans) -> np.ndarray:
    if spans is None:
        return np.zeros((0, 2), np.int64)
    a = np.asarray(spans, np.int64)
    if a.size == 0:
        return np.zeros((0, 2), np.int64)
    return a.reshape(-1, 2)


def with_settings(cfg: BurstConfig, **kw) -> BurstConfig:
    """A copy of ``cfg`` with fields replaced."""
    return replace(cfg, **kw)


__all__ = [
    "BACKGROUNDS", "BURST_WINDOW_START_OFFSET_MS", "BURST_WINDOW_STOP_MS", "CALIBRATIONS", "CALIBRATION_GRID",
    "PRIMARY_SETTING", "PRIMARY_VARIANT", "SPEC_MIN_ROI_CHANNELS", "SPEC_ROI", "VARIANT_TAGS", "Background",
    "BurstConfig", "RunBursts", "RunSpectra", "artefact_chunk_mask", "bosc_detect", "bosc_tf",
    "bosc_tf_segmentwise", "bosc_wavelet", "burst_config_from_mapping", "burst_occupancy",
    "burst_window_samples", "detect_run_bursts", "detect_segmentwise", "edge_samples", "exclude_spans",
    "fit_background", "load_burst_config", "matlab_robustfit", "occ_column",
    "robust_background_fit", "roi_signal", "run_backgrounds", "run_spectra", "setting_suffix", "split_variant",
    "tfr_valid_mask", "thresholds", "trial_burst_features", "wavelet_halfwidth_s", "wavelet_halfwidth_samples",
    "wavelet_sd_s", "window_inside_spans", "window_overlaps_spans", "with_settings",
]
