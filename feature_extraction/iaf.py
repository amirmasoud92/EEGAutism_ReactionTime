# SPDX-License-Identifier: GPL-3.0-or-later
# The restingIAF part of this module is a port of restingIAF v1.0.3 (Corcoran, Alday, Schlesewsky &
# Bornkessel-Schlesewsky, 2018; GPL-3.0; Zenodo 10.5281/zenodo.2575868), so this file is GPL-3.0 licensed.
"""Individual alpha frequency (IAF).

Contents
--------
1. **restingIAF v1.0.3 port** (:func:`resting_iaf`, :func:`restingiaf_from_psd`), checked against the MATLAB code
   (agreement to ~1e-14 on simulated data). MATLAB behaviour is reproduced *including its quirks* (listed in the
   function docstrings). Bins returned as ``*_bins`` are 1-based (MATLAB convention); frequencies are in Hz.
2. **Rest chunks** (:func:`rest_chunk_starts`, :func:`extract_rest_chunks`, :func:`load_rest_run`): clean eyes-open
   rest, 4-s chunks on a grid starting at the first code-50 (``Recording_start``) marker, kept if the chunk lies in
   one valid span (no join, bad span or file edge within 1.65 s, the FIR half-length), overlaps no ``BAD*``
   annotation and the peak-to-peak of every good ROI channel is <= 150 uV. Bad ROI channels (``info['bads']`` after
   ``load_clean(interpolate=False)``) are left out (NaN). Chunks of all runs and all recorded days are pooled; the
   main-day sensitivity analysis pools the main-day runs only.
3. **Spectra.** With ``tover = 0`` and ``tlen`` = the chunk length, restingIAF's Welch estimate of the concatenated
   chunks is exactly the mean of the per-chunk Hamming periodograms. The per-channel mean is taken over the chunks in
   which that channel is good, which generalises MATLAB's channels x samples matrix to channels that are bad in some
   runs only.
4. **specparam estimate** (:func:`specparam_alpha`): fixed aperiodic mode, 2-30 Hz, on the ROI-mean rest spectrum
   (the same Hamming periodograms, linear power averaged over good ROI channels); alpha CF = the highest-PW peak with
   CF in ``alpha_range``.
5. **Fallback chain** (:func:`choose_iaf`): (1) restingIAF peak alpha frequency (PAF); (2) restingIAF centre of
   gravity (CoG); (3) specparam alpha CF with PW >= 0.1 on the ROI rest spectrum; (4) specparam CF of the mean AVSRT
   prestimulus spectrum (:func:`avsrt_prestim_spectrum`); (5) 10 Hz. The source is recorded (:data:`IAF_SOURCES`).

Settings are explicit fields of :class:`IAFSettings`: restingIAF arguments (cmin 3, 1-40 Hz, w [7, 13] Hz, Fw 11,
k 5, 4-s windows, tover 0), specparam peak settings, ``alpha_range`` [7, 13] Hz, ``min_chunks`` = 5 (20 s) and the
AVSRT window rules (the task-window rules with the 200-uV ROI threshold).

The AVSRT fallback reads only the onset samples and the stimulus-defined ``predictor_set`` flags of the trial table
(:func:`avsrt_predictor_onsets`).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Posterior ROI (also ``configs/preprocessing.yaml`` ``artefacts.roi``).
ROI: tuple[str, ...] = ("PO7", "PO8", "PO3", "PO4", "POz", "O1", "O2", "Oz")

#: IAF sources in fallback order.
IAF_SOURCES: tuple[str, ...] = ("restingIAF_PAF", "restingIAF_CoG", "specparam_rest", "specparam_avsrt",
                                "default_10Hz")


class IAFError(RuntimeError):
    """A condition under which MATLAB restingIAF v1.0.3 itself raises an error."""


# =========================================================================== settings
@dataclass(frozen=True)
class RestingIAFParams:
    """restingIAF arguments (cmin 3, 1-40 Hz, w [7, 13], Fw 11, k 5, 4-s windows, tover 0)."""

    cmin: int = 3
    f_range: tuple[int, int] = (1, 40)
    w: tuple[float, float] = (7.0, 13.0)
    Fw: int = 11
    k: int = 5
    mpow: float = 1.0
    mdiff: float = 0.20
    tlen_s: float = 4.0
    tover: int = 0
    nfft: int | None = None
    norm: bool = True


@dataclass(frozen=True)
class SpecparamParams:
    """specparam settings for fallbacks (3) and (4)."""

    freq_range: tuple[float, float] = (2.0, 30.0)
    aperiodic_mode: str = "fixed"
    peak_width_limits: tuple[float, float] = (1.0, 8.0)
    max_n_peaks: int = 6
    min_peak_height: float = 0.05
    peak_threshold: float = 2.0
    alpha_range: tuple[float, float] = (7.0, 13.0)
    #: fallback (3): PW >= this
    min_pw_rest: float = 0.1
    #: fallback (4): no PW threshold; ``None`` = any alpha-range peak
    min_pw_avsrt: float | None = None


@dataclass(frozen=True)
class RestChunkParams:
    """Rest chunk rule."""

    chunk_s: float = 4.0
    edge_margin_s: float = 1.65
    ptp_uv: float = 150.0
    start_code: int = 50
    roi: tuple[str, ...] = ROI
    min_chunks: int = 5
    apply_ica: bool = True
    #: thresholds whose exclusion rates are reported (diagnostics only)
    report_ptp_uv: tuple[float, ...] = (150.0, 200.0)


@dataclass(frozen=True)
class AvsrtPrestimParams:
    """Mean AVSRT prestimulus spectrum for fallback (4) (task-window rules, DPSS estimator)."""

    window_ms: tuple[float, float] = (-850.0, -50.0)
    edge_margin_s: float = 1.65
    ptp_uv: float = 200.0
    min_roi_channels: int = 5
    nfft: int = 1024
    bandwidth: float = 2.0
    subset: str = "predictor_set"
    min_windows: int = 20


@dataclass(frozen=True)
class IAFSettings:
    """All IAF settings."""

    resting: RestingIAFParams = field(default_factory=RestingIAFParams)
    specparam: SpecparamParams = field(default_factory=SpecparamParams)
    chunks: RestChunkParams = field(default_factory=RestChunkParams)
    avsrt: AvsrtPrestimParams = field(default_factory=AvsrtPrestimParams)
    default_hz: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================================== restingIAF port
def _nextpow2(n: int) -> int:
    return int(math.ceil(math.log2(n)))


def _dsearchn(f: np.ndarray, v: float) -> int:
    """0-based index of the grid point nearest ``v`` (first on ties, as MATLAB ``dsearchn``)."""
    return int(np.argmin(np.abs(np.asarray(f) - v)))


def sgolay_matrix(k: int, Fw: int) -> np.ndarray:
    """MATLAB ``[~, g] = sgolay(k, Fw)``: ``G = S * inv(S' * S)``."""
    x = np.arange(-(Fw - 1) / 2, (Fw - 1) / 2 + 1)
    S = x[:, None] ** np.arange(k + 1)[None, :]
    return S @ np.linalg.inv(S.T @ S)


def sgf_diff(x: np.ndarray, Fw: int, k: int, Fs: float, tlen: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sgfDiff.m`` (lines 49-59): S-G smoothed spectrum and its 1st / 2nd derivatives (``conv(..., 'same')``)."""
    G = sgolay_matrix(k, Fw)
    dt = Fs / tlen
    out = []
    for p in range(3):
        kern = math.factorial(p) / (-dt) ** p * G[:, p]
        out.append(np.convolve(x, kern, mode="same"))
    return out[0], out[1], out[2]


def _polyfit_minpow(f: np.ndarray, y: np.ndarray, mpow: float) -> np.ndarray:
    """``restingIAF.m`` lines 147-149: ``polyfit(f, log10(pxx), 1)`` + ``mpow`` x the polyval prediction SE."""
    A = np.column_stack([f, np.ones_like(f)])
    Q, R = np.linalg.qr(A)
    p = np.linalg.solve(R, Q.T @ y)
    yval = A @ p
    normr = np.linalg.norm(y - yval)
    df = len(y) - 2
    E = A @ np.linalg.inv(R)
    e = np.sqrt(1 + np.sum(E * E, axis=1))
    delta = normr / np.sqrt(df) * e
    return yval + mpow * delta


def _less_than1(seg: np.ndarray) -> bool:
    """``lessThan1.m``. QUIRK: ``abs(d1 < 1)`` takes abs of a logical, so the test is ``d1 < 1``, not ``|d1| < 1``."""
    if len(seg) < 2:
        raise IAFError("Length of 1st derivative segment < 2")
    return bool(np.all(seg < 1))


def _sortrows_desc(M: np.ndarray, col: int) -> np.ndarray:
    return M[np.argsort(-M[:, col], kind="stable")]       # MATLAB sortrows(M, -col) is stable


def _sortrows_asc(M: np.ndarray, col: int) -> np.ndarray:
    return M[np.argsort(M[:, col], kind="stable")]


def find_f1(f, d0, d1, negZ, minPow, slen, bin1):
    """``findF1.m``: lower alpha-window bound. ``bin1`` and the returned bin are 1-based.

    QUIRK: ``minPow`` is indexed with ``negZ(1,2)`` (the first row), not ``negZ(z,2)``. If no bound is found MATLAB
    returns bin 0 (which later makes ``chanGravs`` fail); so does this port.
    """
    if negZ.shape[0] > 1:
        negZ = _sortrows_asc(negZ, 2)
        leftPeak = bin1
        for z in range(negZ.shape[0]):
            if np.log10(negZ[z, 3]) > minPow[int(negZ[0, 1]) - 1] or negZ[z, 3] > 0.5 * d0[bin1 - 1]:
                leftPeak = int(negZ[z, 1])
                break
            leftPeak = bin1
    else:
        leftPeak = bin1
    rows = []
    for kb in range(2, leftPeak):                     # MATLAB k = 2:leftPeak-1 (1-based)
        kk = kb - 1
        if np.sign(d1[kk]) < np.sign(d1[kk + 1]):
            mink = int(np.argmin(np.abs([d0[kk - 1], d0[kk], d0[kk + 1]])))
            minim = kb - 1 + mink
            rows.append([len(rows) + 1, minim, f[minim - 1]])
        elif abs(d1[kk]) < 1 and _less_than1(d1[kk + 1:kk + 1 + slen]):
            minim = kb
            rows.append([len(rows) + 1, minim, f[minim - 1]])
    if not rows:
        return 0, 0.0
    P = np.array(rows, float)
    if P.shape[0] == 1:
        return int(P[0, 1]), float(P[0, 2])
    P = _sortrows_desc(P, 2)
    return int(P[0, 1]), float(P[0, 2])


def find_f2(f, d0, d1, negZ, minPow, slen, bin1):
    """``findF2.m``: upper alpha-window bound (bins 1-based; same quirks as :func:`find_f1`)."""
    if negZ.shape[0] > 1:
        negZ = _sortrows_desc(negZ, 2)
        rightPeak = bin1
        for z in range(negZ.shape[0]):
            if np.log10(negZ[z, 3]) > minPow[int(negZ[0, 1]) - 1] or negZ[z, 3] > 0.5 * d0[bin1 - 1]:
                rightPeak = int(negZ[z, 1])
                break
            rightPeak = bin1
    else:
        rightPeak = bin1
    rows = []
    for kb in range(rightPeak + 1, len(d1) - slen + 1):   # k = rightPeak+1 : length(d1)-slen
        kk = kb - 1
        if np.sign(d1[kk]) < np.sign(d1[kk + 1]):
            mink = int(np.argmin(np.abs([d0[kk - 1], d0[kk], d0[kk + 1]])))
            minim = kb - 1 + mink
            rows.append([len(rows) + 1, minim, f[minim - 1]])
        elif abs(d1[kk]) < 1 and _less_than1(d1[kk + 1:kk + 1 + slen]):
            minim = kb
            rows.append([len(rows) + 1, minim, f[minim - 1]])
    if not rows:
        return 0, 0.0
    return int(rows[0][1]), float(rows[0][2])


def peak_bounds(d0, d1, d2, f, w, minPow, minDiff, fres) -> dict:
    """``peakBounds.m``: channel peak (PAF), alpha-window bounds f1/f2 (1-based bins), Q and Qf.

    Candidate peaks are downward zero-crossings of d1 within ``w`` (bins +-1). PAF = the highest candidate whose
    log10 power exceeds ``minPow``, provided it is more than ``minDiff`` higher than the runner-up. If the runner-up
    is too close, there is no PAF, but f1/f2 are still found around the highest candidate (for CoG).

    Raises
    ------
    IAFError
        If no d2 inflection is found on either side of the peak (MATLAB errors with undefined ``min1``/``min2``).
    """
    lower = _dsearchn(f, w[0]) + 1
    upper = _dsearchn(f, w[1]) + 1
    rows = []
    for kb in range(lower - 1, upper + 2):            # k = lower_alpha-1 : upper_alpha+1
        kk = kb - 1
        if np.sign(d1[kk]) > np.sign(d1[kk + 1]):
            maxk = int(np.argmax([d0[kk], d0[kk + 1]]))
            maxim = kb + maxk
            rows.append([len(rows) + 1, maxim, f[maxim - 1], d0[maxim - 1]])
    peakF, subBin, peakBin = np.nan, None, None
    if not rows:
        negZ = np.zeros((1, 4))
    else:
        negZ = np.array(rows, float)
        if negZ.shape[0] == 1:
            if np.log10(negZ[0, 3]) > minPow[int(negZ[0, 1]) - 1]:
                peakBin = int(negZ[0, 1])
                peakF = float(negZ[0, 2])
        else:
            negZ = _sortrows_desc(negZ, 3)
            if np.log10(negZ[0, 3]) > minPow[int(negZ[0, 1]) - 1]:
                if negZ[0, 3] * (1 - minDiff) > negZ[1, 3]:
                    peakBin = int(negZ[0, 1])
                    peakF = float(negZ[0, 2])
                else:
                    subBin = int(negZ[0, 1])
    slen = int(np.floor(1 / fres + 0.5))
    out = dict(peakF=peakF, posZ1=np.nan, posZ2=np.nan, f1=np.nan, f2=np.nan, inf1=np.nan, inf2=np.nan,
               Q=np.nan, Qf=np.nan, n_cross=len(rows))
    if peakBin is None and subBin is None:
        return out
    if peakBin is None:
        f1, p1 = find_f1(f, d0, d1, negZ, minPow, slen, subBin)
        f2, p2 = find_f2(f, d0, d1, negZ, minPow, slen, subBin)
        out.update(f1=f1, posZ1=p1, f2=f2, posZ2=p2)
        return out
    f1, p1 = find_f1(f, d0, d1, negZ, minPow, slen, peakBin)
    f2, p2 = find_f2(f, d0, d1, negZ, minPow, slen, peakBin)
    out.update(f1=f1, posZ1=p1, f2=f2, posZ2=p2)
    inf1_list, min1 = [], None
    for kb in range(1, peakBin):                      # k = 1:peakBin-1
        kk = kb - 1
        if np.sign(d2[kk]) > np.sign(d2[kk + 1]):
            mink = int(np.argmin(np.abs([d2[kk], d2[kk + 1]])))
            min1 = kb + mink
            inf1_list.append(f[min1 - 1])
    min2, inf2 = None, None
    for kb in range(peakBin + 1, len(d2)):            # k = peakBin+1:length(d2)-1
        kk = kb - 1
        if np.sign(d2[kk]) < np.sign(d2[kk + 1]):
            mink = int(np.argmin(np.abs([d2[kk], d2[kk + 1]])))
            min2 = kb + mink
            inf2 = f[min2 - 1]
            break
    if min1 is None or min2 is None:
        raise IAFError("inflection point not found (MATLAB peakBounds errors here)")
    Q = float(np.trapezoid(d0[min1 - 1:min2], f[min1 - 1:min2]))
    out.update(inf1=float(max(inf1_list)), inf2=float(inf2), Q=Q, Qf=Q / (min2 - min1))
    return out


def matlab_welch(x: np.ndarray, Fs: float, tlen: int, tover: int | None = None,
                 nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``pwelch(x, hamming(tlen), tover, nfft, Fs)`` (one-sided PSD, no detrending) along the last axis.

    ``tover=None`` is MATLAB's default (50 % overlap); ``nfft=None`` is ``max(256, 2^nextpow2(tlen))``.
    """
    from scipy.signal import welch
    from scipy.signal.windows import hamming

    if tover is None:
        tover = int(np.fix(0.5 * tlen))
    if nfft is None:
        nfft = max(256, 2 ** _nextpow2(tlen))
    win = hamming(int(tlen), sym=True)               # MATLAB hamming(L) is symmetric
    f, p = welch(x, fs=Fs, window=win, noverlap=int(tover), nfft=int(nfft), detrend=False,
                 return_onesided=True, scaling="density", axis=-1)
    return f, p


@dataclass
class RestingIAFResult:
    """Output of :func:`restingiaf_from_psd` (MATLAB ``pSum`` plus the per-channel structs).

    ``channels`` holds one dict per *analysed* channel (channels without data are skipped, as MATLAB skips NaN
    channels); ``channel_index`` maps them back to the input rows.
    """

    paf: float
    paf_std: float
    cog: float
    cog_std: float
    p_sel: int
    g_sel: int
    iaw_bins: tuple[float, float]
    iaw_hz: tuple[float, float]
    f: np.ndarray
    channels: list[dict]
    channel_index: list[int]

    def as_psum(self) -> dict:
        """MATLAB-style ``pSum`` dict."""
        return dict(paf=self.paf, pafStd=self.paf_std, cog=self.cog, cogStd=self.cog_std, pSel=self.p_sel,
                    gSel=self.g_sel, iaw_bins=self.iaw_bins, iaw_hz=self.iaw_hz)


def _channel_from_psd(f_full: np.ndarray, pxx_full: np.ndarray, Fs: float, tlen: int, p: RestingIAFParams) -> dict:
    """Per-channel restingIAF steps after ``pwelch`` (``restingIAF.m`` lines 135-160)."""
    i1, i2 = _dsearchn(f_full, p.f_range[0]), _dsearchn(f_full, p.f_range[1])
    f = f_full[i1:i2 + 1]
    pxx = pxx_full[i1:i2 + 1]
    pxx = pxx / np.mean(pxx) if p.norm else pxx
    minPow = _polyfit_minpow(f, np.log10(pxx), p.mpow)
    d0, d1, d2 = sgf_diff(pxx, p.Fw, p.k, Fs, tlen)
    fres = Fs / 2 ** _nextpow2(tlen)
    pb = peak_bounds(d0, d1, d2, f, p.w, minPow, p.mdiff, fres)
    pb.update(pxx=pxx, d0=d0, d1=d1, d2=d2, minPow=minPow, f=f)
    return pb


def restingiaf_from_psd(f_full: np.ndarray, pxx: np.ndarray, Fs: float, tlen: int,
                        params: RestingIAFParams | None = None) -> RestingIAFResult:
    """restingIAF from per-channel ``pwelch`` spectra (``chanGravs.m`` + ``chanMeans.m`` across channels).

    Parameters
    ----------
    f_full
        Welch frequency grid (0 .. Fs/2).
    pxx
        ``(n_chan, n_freq)`` PSDs; rows that are all-NaN (or contain NaN) are skipped like MATLAB's NaN channels.
    Fs, tlen
        Sampling rate and Welch window length in samples (enter the S-G derivative scale and ``slen``).
    params
        restingIAF arguments (``cmin``, ``f_range``, ``w``, ``Fw``, ``k``, ``mpow``, ``mdiff``, ``norm``).

    Raises
    ------
    IAFError
        Where MATLAB restingIAF errors: no d2 inflection around a channel peak, or a bound search returning bin 0
        (index error in ``chanGravs``).

    Notes
    -----
    CoG uses the mean f1/f2 across channels as a common individual alpha window and is computed for every
    analysed channel; ``cog = nanmean``. PAF is the Qf-weighted mean of channel peaks (``wts = Qf / max(Qf)``).
    PAF (CoG) is NaN if fewer than ``cmin`` channels give a peak (an f1). With no f1 anywhere MATLAB returns
    ``iaw = [1 1]`` (``dsearchn`` of NaN); this port returns NaN (the only known output difference).
    """
    p = params or RestingIAFParams()
    pxx = np.atleast_2d(np.asarray(pxx, float))
    chans, idx = [], []
    for i, row in enumerate(pxx):
        if np.any(np.isnan(row)):
            continue
        chans.append(_channel_from_psd(np.asarray(f_full, float), row, Fs, tlen, p))
        idx.append(i)
    if not chans:
        return RestingIAFResult(np.nan, np.nan, np.nan, np.nan, 0, 0, (np.nan, np.nan), (np.nan, np.nan),
                                np.array([]), [], [])
    f = chans[0]["f"]
    # chanGravs.m
    f1 = np.array([c["f1"] for c in chans], float)
    f2 = np.array([c["f2"] for c in chans], float)
    t1, t2 = f1[~np.isnan(f1)].astype(int), f2[~np.isnan(f2)].astype(int)
    if np.any(t1 == 0) or np.any(t2 == 0):
        raise IAFError("findF1/findF2 returned bin 0 (MATLAB index error in chanGravs)")
    D0 = np.column_stack([c["d0"] for c in chans])
    if len(t1) == 0 or len(t2) == 0:
        cogs = np.full(len(chans), np.nan)
        iaw: tuple[float, float] = (np.nan, np.nan)
    else:
        m1 = _dsearchn(f, float(np.mean(f[t1 - 1])))
        m2 = _dsearchn(f, float(np.mean(f[t2 - 1])))
        iaw = (m1 + 1, m2 + 1)
        seg = slice(m1, m2 + 1)
        cogs = np.nansum(D0[seg] * f[seg, None], axis=0) / np.sum(D0[seg], axis=0)
    selG = ~np.isnan(f1)
    # chanMeans.m
    peaks = np.array([c["peakF"] for c in chans], float)
    qf = np.array([c["Qf"] for c in chans], float)
    selP = ~np.isnan(peaks)
    with np.errstate(invalid="ignore", divide="ignore"):
        wts = qf / np.nanmax(qf) if np.any(~np.isnan(qf)) else qf
    if selP.sum() < p.cmin:
        paf, paf_std = np.nan, np.nan
    else:
        paf = float(np.nansum(peaks * wts) / np.nansum(wts))
        paf_std = float(np.nanstd(peaks, ddof=1)) if selP.sum() > 1 else 0.0
    if selG.sum() < p.cmin:
        cog, cog_std = np.nan, np.nan
    else:
        cog = float(np.nanmean(cogs))
        cog_std = float(np.nanstd(cogs, ddof=1)) if len(cogs) > 1 else 0.0
    for c, g in zip(chans, cogs):
        c["gravs"] = float(g)
    iaw_hz = (float(f[int(iaw[0]) - 1]), float(f[int(iaw[1]) - 1])) if not np.isnan(iaw[0]) else (np.nan, np.nan)
    return RestingIAFResult(paf, paf_std, cog, cog_std, int(selP.sum()), int(selG.sum()), iaw, iaw_hz, f, chans, idx)


def resting_iaf(data: np.ndarray, cmin: int, fRange: Sequence[int], Fs: float, w: Sequence[float], Fw: int, k: int,
                mpow: float = 1.0, mdiff: float = 0.20, tlen: int | None = None, tover: int | None = None,
                nfft: int | None = None, norm: bool = True) -> tuple[dict, list[dict], np.ndarray]:
    """MATLAB-signature ``restingIAF(data, nchan, cmin, fRange, Fs, w, Fw, k, mpow, mdiff, 'hamming', tlen, tover,
    nfft, norm)`` on continuous ``(n_chan, n_samples)`` data (channels with NaN are skipped).

    Defaults are MATLAB's (``tlen = 4 Fs``, ``tover`` = 50 %); the IAF here uses ``tover = 0``.
    Returns ``(pSum, channel dicts, f)``.
    """
    data = np.atleast_2d(np.asarray(data, float))
    tlen = int(Fs * 4) if tlen is None else int(tlen)
    params = RestingIAFParams(cmin=cmin, f_range=(int(fRange[0]), int(fRange[1])), w=(float(w[0]), float(w[1])),
                              Fw=int(Fw), k=int(k), mpow=mpow, mdiff=mdiff, tlen_s=tlen / Fs,
                              tover=-1 if tover is None else int(tover), nfft=nfft, norm=norm)
    n_f = (max(256, 2 ** _nextpow2(tlen)) if nfft is None else int(nfft)) // 2 + 1
    P = np.full((data.shape[0], n_f), np.nan)
    f_full = None
    for i, x in enumerate(data):
        if np.any(np.isnan(x)):
            continue
        f_full, P[i] = matlab_welch(x, Fs, tlen, tover, nfft)
    if f_full is None:
        f_full = np.fft.rfftfreq((n_f - 1) * 2, 1 / Fs)
    res = restingiaf_from_psd(f_full, P, Fs, tlen, params)
    return res.as_psum(), res.channels, res.f


# =========================================================================== rest chunks
def rest_start_sample(events, code: int = 50) -> int | None:
    """Sample of the first ``code`` event (``Recording_start``) in an events table, or None."""
    if events is None or len(events) == 0:
        return None
    s = np.asarray(events["sample"], np.int64)[np.asarray(events["value"], np.int64) == int(code)]
    return int(s.min()) if s.size else None


def rest_chunk_starts(start_sample: int, n_times: int, sfreq: float, seg=None,
                      params: RestChunkParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Grid of 4-s chunk starts from ``start_sample`` and the edge-validity mask.

    The grid is ``start_sample + j * L`` (``L = round(chunk_s * sfreq)``) for every chunk that ends inside the
    recording. A chunk is edge-valid if it lies in ONE valid span of ``seg`` (:class:`SegmentInfo`) with
    ``edge_margin_s`` to spare on both sides (no join, bad span or file edge within 1.65 s). Without
    ``seg``, only the file edges are checked.
    """
    p = params or RestChunkParams()
    L = int(round(p.chunk_s * sfreq))
    m = int(round(p.edge_margin_s * sfreq))
    st = np.arange(int(start_sample), int(n_times) - L + 1, L, dtype=np.int64)
    if seg is not None:
        ok = seg.window_valid(st, st + L, margin=m) if st.size else np.zeros(0, bool)
    else:
        ok = (st - m >= 0) & (st + L + m <= n_times)
    return st, np.asarray(ok, bool)


def hamming_periodograms(chunks: np.ndarray, sfreq: float, nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-chunk MATLAB-``pwelch`` periodograms (Hamming window = whole chunk, one segment) along the last axis."""
    L = chunks.shape[-1]
    return matlab_welch(chunks, sfreq, L, 0, nfft)


@dataclass
class RestRunChunks:
    """Clean rest chunks of one run, stored as per-chunk periodograms.

    Attributes
    ----------
    psd
        ``(n_kept, n_roi, n_freq)`` Hamming periodograms (uV^2/Hz) of the kept chunks; NaN rows = bad channel.
    f
        Frequency grid of ``psd``.
    roi
        Channel names (rows of ``psd``).
    good_roi
        Good ROI channels of this run.
    n_grid, n_edge_valid, n_kept
        Chunks on the grid, edge-valid chunks, chunks kept (edge-valid, no ``BAD*`` overlap, ptp <= threshold).
    ptp_uv
        Max ROI peak-to-peak (uV) of every edge-valid chunk that overlaps no ``BAD*`` annotation (threshold
        diagnostics; ``n_edge_valid - n_bad_annotation`` values).
    meta
        Free-form (run, is_main_day, status, ...).
    n_bad_annotation
        Edge-valid chunks dropped because they overlap a ``BAD*`` annotation of the cleaned Raw (``BAD_icafit``,
        ``BAD_short``, rail / flat / unusable, manual ``BAD``), as for the task windows.
    """

    psd: np.ndarray
    f: np.ndarray
    roi: tuple[str, ...]
    good_roi: tuple[str, ...]
    n_grid: int
    n_edge_valid: int
    n_kept: int
    ptp_uv: np.ndarray
    meta: dict = field(default_factory=dict)
    n_bad_annotation: int = 0


def _overlaps(a: np.ndarray, b: np.ndarray, spans: np.ndarray | None) -> np.ndarray:
    """Whether each half-open ``[a, b)`` intersects a half-open span of ``spans`` (``(k, 2)`` sample indices)."""
    a = np.atleast_1d(np.asarray(a, np.int64))
    b = np.atleast_1d(np.asarray(b, np.int64))
    out = np.zeros(a.shape, bool)
    if spans is None:
        return out
    for s0, s1 in np.asarray(spans, np.int64).reshape(-1, 2):
        out |= (a < s1) & (b > s0)
    return out


def bad_annotation_spans(raw, prefix: str = "BAD") -> np.ndarray:
    """``BAD*`` annotation spans of a cleaned Raw (:func:`feature_extraction.spectral.bad_annotation_spans`: the
    rule of the task windows)."""
    from feature_extraction.spectral import bad_annotation_spans as _bas

    return _bas(raw, prefix)


def extract_rest_chunks(roi_data_v: np.ndarray, sfreq: float, start_sample: int, seg=None,
                        good: Sequence[bool] | None = None, params: RestChunkParams | None = None,
                        nfft: int | None = None, meta: Mapping[str, Any] | None = None,
                        bad_spans: np.ndarray | None = None) -> RestRunChunks:
    """Cut, screen and transform the rest chunks of one run.

    Parameters
    ----------
    roi_data_v
        ``(n_roi, n_times)`` cleaned ROI data in volts (rows in ``params.roi`` order).
    sfreq
        Sampling rate.
    start_sample
        First code-50 sample.
    seg
        :class:`~preprocessing.segments.SegmentInfo` of the recording (joins and bad spans).
    good
        Per-ROI-channel goodness (False = bad channel, left out of ptp and set to NaN).
    bad_spans
        ``(k, 2)`` half-open sample spans of the cleaned Raw's ``BAD*`` annotations (:func:`bad_annotation_spans`);
        an edge-valid chunk overlapping one is dropped before the peak-to-peak rule (the task-window rule, so
        ``BAD_icafit`` / ``BAD_short`` spans never enter the IAF).
    """
    p = params or RestChunkParams()
    X = np.asarray(roi_data_v, float) * 1e6
    n_roi, n_times = X.shape
    good_m = np.ones(n_roi, bool) if good is None else np.asarray(good, bool)
    st, ok = rest_chunk_starts(start_sample, n_times, sfreq, seg, p)
    L = int(round(p.chunk_s * sfreq))
    bad = ok & _overlaps(st, st + L, bad_spans)
    sv = st[ok & ~bad]
    if sv.size and good_m.any():
        C = np.stack([X[:, a:a + L] for a in sv])                     # (n, roi, L)
        ptp = np.ptp(C[:, good_m, :], axis=-1).max(axis=1)
        keep = ptp <= p.ptp_uv
        C = C[keep]
    else:
        ptp = np.zeros(0)
        C = np.zeros((0, n_roi, L))
    nf = (max(256, 2 ** _nextpow2(L)) if nfft is None else int(nfft)) // 2 + 1
    if len(C):
        f, P = hamming_periodograms(C, sfreq, nfft)
        P[:, ~good_m, :] = np.nan
    else:
        f = np.fft.rfftfreq((nf - 1) * 2, 1 / sfreq)
        P = np.zeros((0, n_roi, nf))
    return RestRunChunks(psd=P, f=f, roi=tuple(p.roi), good_roi=tuple(c for c, g in zip(p.roi, good_m) if g),
                         n_grid=int(st.size), n_edge_valid=int(ok.sum()), n_kept=int(len(C)), ptp_uv=ptp,
                         meta=dict(meta or {}), n_bad_annotation=int(bad.sum()))


def load_rest_run(rec, params: RestChunkParams | None = None, *, root=None, events=None,
                  meta: Mapping[str, Any] | None = None, load_clean_fn: Callable | None = None) -> RestRunChunks:
    """Clean rest chunks of one Restingstate recording.

    ``load_clean(rec, apply_ica=params.apply_ica, interpolate=False)``: bad channels stay in ``info['bads']``
    (the final reference is over the originally good channels either way, so good channels are identical to the
    interpolated output) and are left out. Joins and bad spans come from the stored ``SegmentInfo``; chunks
    overlapping a ``BAD*`` annotation of the cleaned Raw (incl. ``BAD_icafit``) are dropped.

    Parameters
    ----------
    rec
        :class:`~utils.bids_index.Recording` (events from ``rec.events_path`` unless ``events`` given).
    root
        Directory of the preprocessing outputs.
    """
    from preprocessing import io as pio
    from preprocessing.clean import load_clean
    from trials.io import read_events_tsv

    p = params or RestChunkParams()
    md = dict(meta or {})
    md.setdefault("run", int(rec.run))
    if events is None:
        events = read_events_tsv(rec.events_path) if rec.events_path is not None else None
    s0 = rest_start_sample(events, p.start_code)
    if s0 is None:
        md["status"] = "no_start_marker"
        nf = max(256, 2 ** _nextpow2(int(round(p.chunk_s * 512)))) // 2 + 1
        return RestRunChunks(np.zeros((0, len(p.roi), nf)), np.zeros(nf), tuple(p.roi), (), 0, 0, 0, np.zeros(0), md)
    if root is None:
        raise ValueError("root (directory of the preprocessing outputs) is required")
    paths = pio.recording_outputs(root, rec)
    raw = (load_clean_fn or load_clean)(rec, apply_ica=p.apply_ica, interpolate=False, root=root)
    seg = pio.read_segments(paths["segments"])
    sf = float(raw.info["sfreq"])
    bads = set(raw.info["bads"])
    missing = [c for c in p.roi if c not in raw.ch_names]
    if missing:
        raise ValueError(f"ROI channels missing from the recording: {missing}")
    X = raw.get_data(picks=list(p.roi))
    good = [c not in bads for c in p.roi]
    md.update(status="ok", start_sample=int(s0), n_times=int(raw.n_times), n_bad_roi=int(len(p.roi) - sum(good)),
              ica_applied=bool(raw.info["temp"].get("ica_applied", False)) if raw.info.get("temp") else None)
    bsp = bad_annotation_spans(raw)
    md["n_bad_annotation_spans"] = int(len(bsp))
    return extract_rest_chunks(X, sf, s0, seg, good, p, meta=md, bad_spans=bsp)


def pooled_channel_psd(runs: Iterable[RestRunChunks], min_chunks: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel mean periodogram over all kept chunks of ``runs`` in which the channel is good.

    Returns ``(f, psd (n_roi, n_f), n_chunks per channel)``; a channel with fewer than ``min_chunks`` chunks is NaN.
    """
    runs = [r for r in runs if r.n_kept]
    if not runs:
        return np.zeros(0), np.zeros((0, 0)), np.zeros(0, int)
    f = runs[0].f
    P = np.concatenate([r.psd for r in runs], axis=0)                # (n, roi, nf)
    okc = ~np.isnan(P[:, :, 0])
    n = okc.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(P, axis=0)
    mean[n < max(1, int(min_chunks))] = np.nan
    return f, mean, n.astype(int)


# =========================================================================== specparam
def specparam_alpha(f: np.ndarray, psd: np.ndarray, params: SpecparamParams | None = None) -> dict:
    """specparam fit of one spectrum and its alpha peak.

    Returns a dict with ``cf``, ``pw`` (specparam's converted peak power, used for the PW >= 0.1 rule),
    ``bw``, ``height`` (fitted gaussian height above the aperiodic fit), ``exponent``, ``offset``, ``r2``,
    ``n_peaks``, ``n_alpha_peaks`` and ``error`` (None on success). With no peak in ``alpha_range``, ``cf`` is NaN
    and ``pw`` is 0.
    """
    from specparam import SpectralModel

    p = params or SpecparamParams()
    out = dict(cf=np.nan, pw=np.nan, bw=np.nan, height=np.nan, exponent=np.nan, offset=np.nan, r2=np.nan,
               n_peaks=0, n_alpha_peaks=0, error=None)
    f = np.asarray(f, float)
    psd = np.asarray(psd, float)
    if f.size == 0 or psd.size != f.size or not np.all(np.isfinite(psd)):
        out["error"] = "no_spectrum"
        return out
    fm = SpectralModel(peak_width_limits=list(p.peak_width_limits), max_n_peaks=p.max_n_peaks,
                       min_peak_height=p.min_peak_height, peak_threshold=p.peak_threshold,
                       aperiodic_mode=p.aperiodic_mode, verbose=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fm.fit(f, psd, list(p.freq_range))
        ap = np.atleast_1d(fm.get_params("aperiodic"))
        out.update(offset=float(ap[0]), exponent=float(ap[-1]), r2=float(fm.get_metrics("gof", "rsquared")))
        pk = np.atleast_2d(fm.get_params("periodic"))
        raw_pk = np.atleast_2d(fm.results.get_params("periodic", version="fit"))
    except Exception as e:  # noqa: BLE001 - recorded, the chain falls through
        out["error"] = f"specparam: {e!r}"[:200]
        return out
    if pk.size == 0 or np.all(np.isnan(pk)):
        out.update(pw=0.0)
        return out
    out["n_peaks"] = int(len(pk))
    sel = (pk[:, 0] >= p.alpha_range[0]) & (pk[:, 0] <= p.alpha_range[1])
    out["n_alpha_peaks"] = int(sel.sum())
    if not sel.any():
        out.update(pw=0.0)
        return out
    i = np.flatnonzero(sel)[int(np.argmax(pk[sel, 1]))]
    out.update(cf=float(pk[i, 0]), pw=float(pk[i, 1]), bw=float(pk[i, 2]),
               height=float(raw_pk[i, 1]) if raw_pk.shape == pk.shape else np.nan)
    return out


# =========================================================================== fallback chain
@dataclass
class IAFChoice:
    """IAF used downstream and where it came from (one of :data:`IAF_SOURCES`)."""

    iaf: float
    source: str
    reasons: list[str] = field(default_factory=list)


def choose_iaf(paf: float, cog: float, sp_rest: Mapping[str, Any] | None,
               sp_avsrt: Mapping[str, Any] | Callable[[], Mapping[str, Any] | None] | None = None,
               settings: IAFSettings | None = None) -> IAFChoice:
    """IAF fallback chain (module docstring, item 5).

    Parameters
    ----------
    paf, cog
        restingIAF PAF and CoG (NaN if absent).
    sp_rest
        :func:`specparam_alpha` result on the ROI rest spectrum (or None).
    sp_avsrt
        :func:`specparam_alpha` result on the mean AVSRT prestimulus spectrum, or a zero-argument callable that
        computes it (called only if steps 1-3 fail), or None.
    """
    s = settings or IAFSettings()
    reasons: list[str] = []
    if np.isfinite(paf):
        return IAFChoice(float(paf), IAF_SOURCES[0], reasons)
    reasons.append("no_paf")
    if np.isfinite(cog):
        return IAFChoice(float(cog), IAF_SOURCES[1], reasons)
    reasons.append("no_cog")
    if sp_rest is not None and np.isfinite(sp_rest.get("cf", np.nan)) and \
            float(sp_rest.get("pw", 0.0)) >= s.specparam.min_pw_rest:
        return IAFChoice(float(sp_rest["cf"]), IAF_SOURCES[2], reasons)
    reasons.append("no_specparam_rest_peak" if sp_rest is None or not np.isfinite(sp_rest.get("cf", np.nan))
                   else "specparam_rest_pw_below_threshold")
    sa = sp_avsrt() if callable(sp_avsrt) else sp_avsrt
    if sa is not None and np.isfinite(sa.get("cf", np.nan)) and \
            (s.specparam.min_pw_avsrt is None or float(sa.get("pw", 0.0)) >= s.specparam.min_pw_avsrt):
        return IAFChoice(float(sa["cf"]), IAF_SOURCES[3], reasons)
    reasons.append("no_specparam_avsrt_peak" if sa is None or not np.isfinite(sa.get("cf", np.nan))
                   else "specparam_avsrt_pw_below_threshold")
    return IAFChoice(float(s.default_hz), IAF_SOURCES[4], reasons)


# =========================================================================== child level
def estimate_from_runs(runs: Sequence[RestRunChunks], settings: IAFSettings | None = None, *,
                       sfreq: float = 512.0,
                       sp_avsrt: Mapping[str, Any] | Callable[[], Mapping[str, Any] | None] | None = None) -> dict:
    """IAF of one child from the clean chunks of the given runs (all days, or the main day only).

    Returns a flat dict: chunk counts, restingIAF outputs (``paf``, ``cog``, ``p_sel``, ``g_sel``, ``iaw_lo_hz``,
    ``iaw_hi_hz``, ``restingiaf_error``), specparam rest outputs (``sp_*``), the chosen ``iaf`` and ``iaf_source``,
    and the AVSRT fallback outputs (``avsrt_*``) when the chain reached step 4.
    """
    s = settings or IAFSettings()
    rp = s.resting
    tlen = int(round(rp.tlen_s * sfreq))
    if int(round(s.chunks.chunk_s * sfreq)) != tlen or rp.tover != 0:
        raise ValueError("pooling chunk periodograms equals restingIAF only with tlen = chunk length and tover = 0")
    used = [r for r in runs if r.n_kept]
    n_kept = int(sum(r.n_kept for r in runs))
    out: dict[str, Any] = dict(
        n_runs=len(runs), n_runs_with_chunks=len(used), n_chunks_grid=int(sum(r.n_grid for r in runs)),
        n_chunks_edge_valid=int(sum(r.n_edge_valid for r in runs)),
        n_chunks_bad_annotation=int(sum(getattr(r, "n_bad_annotation", 0) for r in runs)), n_chunks_clean=n_kept,
        rest_s_clean=n_kept * s.chunks.chunk_s)
    for thr in s.chunks.report_ptp_uv:
        ptp = np.concatenate([r.ptp_uv for r in runs]) if runs else np.zeros(0)
        out[f"n_chunks_ptp_gt_{int(thr)}"] = int(np.sum(ptp > thr))
    out.update(paf=np.nan, paf_std=np.nan, cog=np.nan, cog_std=np.nan, p_sel=0, g_sel=0, n_channels=0,
               iaw_lo_hz=np.nan, iaw_hi_hz=np.nan, restingiaf_error=None)
    sp_rest = None
    if n_kept >= s.chunks.min_chunks:
        f, P, nch = pooled_channel_psd(used, min_chunks=s.chunks.min_chunks)
        out["n_channels"] = int(np.sum(np.all(np.isfinite(P), axis=1)))
        out["min_chunks_per_channel"] = int(nch[nch > 0].min()) if np.any(nch > 0) else 0
        try:
            r = restingiaf_from_psd(f, P, sfreq, tlen, rp)
            out.update(paf=r.paf, paf_std=r.paf_std, cog=r.cog, cog_std=r.cog_std, p_sel=r.p_sel, g_sel=r.g_sel,
                       iaw_lo_hz=r.iaw_hz[0], iaw_hi_hz=r.iaw_hz[1])
        except IAFError as e:
            out["restingiaf_error"] = str(e)
        good = np.all(np.isfinite(P), axis=1)
        if good.any():
            sp_rest = specparam_alpha(f, P[good].mean(axis=0), s.specparam)
    else:
        out["restingiaf_error"] = f"too_few_chunks (< {s.chunks.min_chunks})"
    for k in ("cf", "pw", "bw", "height", "exponent", "r2", "n_alpha_peaks", "error"):
        out[f"sp_{k}"] = (sp_rest or {}).get(k, np.nan if k != "error" else "not_run")
    holder: dict[str, Any] = {}

    def _avsrt():
        if sp_avsrt is None:
            holder["avsrt"] = None
            return None
        holder["avsrt"] = sp_avsrt() if callable(sp_avsrt) else sp_avsrt
        return holder["avsrt"]

    ch = choose_iaf(out["paf"], out["cog"], sp_rest, _avsrt, s)
    out.update(iaf=ch.iaf, iaf_source=ch.source, iaf_fallback_reasons=";".join(ch.reasons))
    if "avsrt" in holder:
        a = holder["avsrt"] or {}
        for k in ("cf", "pw", "r2", "n_windows", "error"):
            out[f"avsrt_{k}"] = a.get(k, np.nan if k != "error" else "unavailable")
    return out


# =========================================================================== AVSRT fallback (4)
def dpss_single_taper_psd(x: np.ndarray, sfreq: float, nfft: int = 1024, bandwidth: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """DPSS estimator of the alpha windows: MNE ``psd_array_multitaper`` (``bandwidth``, ``low_bias=True``, ``adaptive=False``,
    ``normalization='full'``, DC removed) zero-padded to ``nfft`` points (MNE itself does not zero-pad).

    Equals ``psd_array_multitaper`` exactly when ``nfft == x.shape[-1]``.
    """
    from mne.time_frequency.multitaper import _compute_mt_params, _mt_spectra, _psd_from_mt

    x = np.asarray(x, float)
    n = x.shape[-1]
    dpss, eig, _ = _compute_mt_params(n, sfreq, bandwidth, True, False, verbose="error")
    w = np.sqrt(eig)[np.newaxis, :, np.newaxis]
    shp = x.shape[:-1]
    xm, f = _mt_spectra(x.reshape(-1, n), dpss, sfreq, n_fft=int(nfft), remove_dc=True)
    psd = _psd_from_mt(xm, w) / sfreq
    return f, psd.reshape(*shp, len(f))


def avsrt_prestim_spectrum(roi_data_v: np.ndarray, sfreq: float, onsets: np.ndarray, seg=None,
                           good: Sequence[bool] | None = None,
                           params: AvsrtPrestimParams | None = None,
                           bad_spans: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Sum of the ROI-mean DPSS prestimulus spectra of one AVSRT recording, for fallback (4).

    Windows ``onset + window_ms`` that are edge-valid (``seg``, margin 1.65 s), overlap no ``BAD*`` annotation
    (``bad_spans``) and whose good-ROI peak-to-peak is <= ``ptp_uv`` (200 uV) are kept;
    the recording needs ``min_roi_channels`` good ROI channels.
    Returns ``(f, sum of window spectra (uV^2/Hz), n_windows)`` so that several runs can be pooled exactly.
    """
    p = params or AvsrtPrestimParams()
    X = np.asarray(roi_data_v, float) * 1e6
    g = np.ones(X.shape[0], bool) if good is None else np.asarray(good, bool)
    nf = p.nfft // 2 + 1
    f0 = np.fft.rfftfreq(p.nfft, 1 / sfreq)
    if g.sum() < p.min_roi_channels or len(onsets) == 0:
        return f0, np.zeros(nf), 0
    on = np.asarray(onsets, np.int64)
    a = on + int(round(p.window_ms[0] * sfreq / 1000))
    b = on + int(round(p.window_ms[1] * sfreq / 1000))
    m = int(round(p.edge_margin_s * sfreq))
    if seg is not None:
        ok = seg.window_valid(a, b, margin=m)
    else:
        ok = (a - m >= 0) & (b + m <= X.shape[1])
    ok = ok & ~_overlaps(a, b, bad_spans)
    a, b = a[ok], b[ok]
    if not len(a):
        return f0, np.zeros(nf), 0
    W = np.stack([X[g][:, s:e] for s, e in zip(a, b)])            # (n, g, L)
    keep = np.ptp(W, axis=-1).max(axis=1) <= p.ptp_uv
    W = W[keep]
    if not len(W):
        return f0, np.zeros(nf), 0
    f, P = dpss_single_taper_psd(W, sfreq, p.nfft, p.bandwidth)    # (n, g, nf)
    return f, P.mean(axis=1).sum(axis=0), int(len(W))


def avsrt_predictor_onsets(trials_parquet: str | Path, subjects: Iterable[str], task_cfg) -> dict[tuple[str, int], np.ndarray]:
    """Onset samples of the AVSRT predictor set per (subject, run), reading only the key and flag columns.

    Only ``subject``, ``run``, ``sample`` and the boolean flags of the stimulus-defined subset
    ``task_cfg.analysis_subsets["predictor_set"]`` are read.
    """
    import pandas as pd

    from trials.schema import apply_subset

    sub = task_cfg.analysis_subsets["predictor_set"]
    flags = sorted(set(sub.get("require_true", [])) | set(sub.get("require_false", [])))
    df = pd.read_parquet(trials_parquet, columns=["subject", "run", "sample", *flags],
                         filters=[("subject", "in", sorted(set(subjects)))])
    df = df[apply_subset(df, sub)]
    return {(str(s), int(r)): np.sort(g["sample"].to_numpy(np.int64)) for (s, r), g in df.groupby(["subject", "run"])}


def avsrt_specparam_for_child(recs: Sequence, onsets: Mapping[tuple[str, int], np.ndarray],
                              settings: IAFSettings | None = None, *, root=None,
                              load_clean_fn: Callable | None = None) -> dict:
    """Fallback (4) for one child: specparam on the mean ROI DPSS spectrum of its AVSRT predictor-set windows."""
    from preprocessing import io as pio
    from preprocessing.clean import load_clean

    s = settings or IAFSettings()
    tot, n, f = None, 0, None
    for rec in recs:
        on = onsets.get((rec.subject, int(rec.run)))
        if on is None or not len(on):
            continue
        paths = pio.recording_outputs(root, rec) if root is not None else None
        raw = (load_clean_fn or load_clean)(rec, apply_ica=s.chunks.apply_ica, interpolate=False, root=root)
        seg = pio.read_segments(paths["segments"]) if paths is not None else None
        bads = set(raw.info["bads"])
        X = raw.get_data(picks=list(s.chunks.roi))
        f, S, k = avsrt_prestim_spectrum(X, float(raw.info["sfreq"]), on, seg, [c not in bads for c in s.chunks.roi],
                                         s.avsrt, bad_spans=bad_annotation_spans(raw))
        tot = S if tot is None else tot + S
        n += k
        del raw
    if n < s.avsrt.min_windows or tot is None:
        return dict(cf=np.nan, pw=np.nan, r2=np.nan, n_windows=n, error=f"too_few_windows ({n})")
    r = specparam_alpha(f, tot / n, s.specparam)
    r["n_windows"] = n
    return r
