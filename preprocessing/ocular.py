"""Ocular copy: the preprocessing step that builds the stored ocular output.

1. :func:`build_ocular_copy` makes a small, separate, **non-high-passed**
   copy of the ocular channels from the raw (DC-coupled, CMS-referenced) BDF
   data. It is independent of the 1-40 Hz analysis copy.
2. The trial-level ocular *indices* (:func:`compute_ocular_indices`,
   :func:`detect_blinks`, :func:`detect_saccades`, ...) live in
   :mod:`preprocessing.ocular_indices`: they read the stored copy. The names
   are re-exported here (lazily).

When EXG1 or EXG2 fails, the copy's HEOG is F7 - F8 and the index ``gaze_dev``
is computed from it; ``gaze_dev_z`` is then set to 0 with an indicator by the
feature step (:mod:`feature_extraction.ocular_trial`, ``gaze_dev_fallback``).

Signals
-------
``VEOG`` = mean(Fp1, Fp2) - mean(EXG3, EXG4) (no infra-orbital electrode
exists); ``HEOG`` = EXG1 - EXG2 (right minus left outer canthus). Fallbacks, decided from
:func:`~preprocessing.exg_qc.exg_qc`: HEOG := F7 - F8 if EXG1 or
EXG2 fails (or is missing, or passed as bad); the VEOG reference := mean(T7, T8)
if EXG3 or EXG4 fails. Note that F7 - F8 has the opposite polarity to
EXG1 - EXG2; every HEOG index here (log SD, absolute deviation) is
sign-invariant.

Filtering and time base
-----------------------
Each valid piece (segments between Status bit-16 joins minus rail spans, see
:func:`~preprocessing.exg_qc.valid_pieces`) is low-passed on its
own with a zero-phase FIR (firwin, Hamming, MNE 'auto' length), at 5 Hz
(``VEOG``, ``HEOG``: closure and gaze deviation) and at 15 Hz (``VEOG_lp15``,
``HEOG_lp15`` and the stored constituents: blinks and window log-SD). There is
no high-pass. The low-passed signal is then decimated by ``decim`` (default
8: 512 -> 64 Hz) by keeping the samples whose absolute BDF index is a
multiple of ``decim``; the low-pass is the anti-alias filter, and the build
refuses a configuration whose 15-Hz stop-band edge reaches the new Nyquist.

**Time base:** ocular sample ``j`` (counted *with* the copy's ``first_samp``)
is BDF sample ``j * decim``; for an uncropped BDF (``first_samp`` 0) the copy
has ``ceil(n_times / decim)`` samples and ``first_samp`` 0, and
``ocular.times[j] == j * decim / 512``. Samples outside the valid pieces are
0 in every signal channel and 0 in the ``OC_VALID`` mask channel;
``OC_PIECE`` holds the piece index (-1 outside). The build parameters are
stored as JSON in ``info['description']`` (read with :func:`ocular_meta`).
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.config_ocular import OcularConfig, as_ocular_config
from preprocessing.exg_qc import exg_qc, valid_pieces

__all__ = [
    "OCULAR_FORMAT", "OCULAR_FORMAT_VERSION", "OcularSources", "choose_ocular_sources", "build_ocular_copy",
    "ocular_meta", "save_ocular_copy", "fir_half_length_s", "detect_blinks", "running_median_by_piece",
    "compute_ocular_indices", "INDEX_COLUMNS", "detect_saccades",
]

#: Names served from :mod:`preprocessing.ocular_indices` (module ``__getattr__``).
_INDEX_NAMES = frozenset({"detect_blinks", "running_median_by_piece", "compute_ocular_indices", "INDEX_COLUMNS",
                          "detect_saccades", "_piece_bounds", "_window_idx", "_log"})


def __getattr__(name: str):
    if name in _INDEX_NAMES:
        from preprocessing import ocular_indices

        return getattr(ocular_indices, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

OCULAR_FORMAT = "alpha_state.ocular_copy"
OCULAR_FORMAT_VERSION = 1
VALID_CH = "OC_VALID"
PIECE_CH = "OC_PIECE"
DERIVED = ("VEOG", "HEOG", "VEOG_lp15", "HEOG_lp15")



# --------------------------------------------------------------------------- sources (fallbacks)
@dataclass(frozen=True)
class OcularSources:
    """Channels used for the derived signals, and why.

    ``heog``: (plus, minus) pair or ``None``; ``ref``: reference channels or
    ``None``; ``fp``: Fp channels used (may be empty).
    """

    heog: tuple[str, str] | None
    heog_label: str
    ref: tuple[str, ...] | None
    ref_label: str
    fp: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"heog": list(self.heog) if self.heog else None, "heog_label": self.heog_label,
                "ref": list(self.ref) if self.ref else None, "ref_label": self.ref_label, "fp": list(self.fp),
                "notes": list(self.notes)}


def choose_ocular_sources(qc: Mapping[str, Mapping[str, Any]], ch_names: Sequence[str],
                          cfg: OcularConfig | Mapping[str, Any] | None = None,
                          bad_channels: Sequence[str] = ()) -> OcularSources:
    """Apply the ocular-channel fallbacks.

    A channel is usable if it is in ``ch_names``, not in ``bad_channels``,
    and (for EXG channels checked by QC) passed :func:`exg_qc`. Known-issue
    channel statuses should be passed in ``bad_channels`` by the caller.
    """
    c = as_ocular_config(cfg).channels
    names, bads = set(ch_names), set(bad_channels)
    notes: list[str] = []

    def ok(ch: str) -> bool:
        if ch not in names or ch in bads:
            return False
        r = qc.get(ch)
        return True if r is None else bool(r.get("pass", False))

    def why(ch: str) -> str:
        if ch not in names:
            return f"{ch} missing"
        if ch in bads:
            return f"{ch} marked bad"
        return f"{ch} failed EXG QC ({';'.join(qc[ch].get('fail_reasons', []))})"

    h1, h2 = c.heog_pair
    if ok(h1) and ok(h2):
        heog, hl = (h1, h2), f"{h1}-{h2}"
    else:
        notes += [why(ch) for ch in (h1, h2) if not ok(ch)]
        f1, f2 = c.heog_fallback_pair
        if ok(f1) and ok(f2):
            heog, hl = (f1, f2), f"{f1}-{f2}"
            notes.append(f"HEOG fallback {f1}-{f2}")
        else:
            heog, hl = None, "none"
            notes.append("no usable HEOG pair")
    m1, m2 = c.mastoids
    if ok(m1) and ok(m2):
        ref, rl = (m1, m2), f"mean({m1},{m2})"
    else:
        notes += [why(ch) for ch in (m1, m2) if not ok(ch)]
        t1, t2 = c.ref_fallback
        if ok(t1) and ok(t2):
            ref, rl = (t1, t2), f"mean({t1},{t2})"
            notes.append(f"VEOG reference fallback mean({t1},{t2})")
        else:
            ref, rl = None, "none"
            notes.append("no usable VEOG reference")
    fp = tuple(ch for ch in c.fp if ok(ch))
    if len(fp) < len(c.fp):
        notes += [why(ch) for ch in c.fp if not ok(ch)]
        if not c.allow_single_fp or not fp:
            fp = ()
            notes.append("no usable Fp channels for VEOG")
    return OcularSources(heog=heog, heog_label=hl, ref=ref, ref_label=rl if fp else "none", fp=fp,
                         notes=tuple(notes))


# --------------------------------------------------------------------------- filters
def fir_half_length_s(sfreq: float, l_freq: float | None, h_freq: float | None,
                      fir_window: str = "hamming") -> float:
    """Half-length (s) of the zero-phase FIR MNE builds for these settings ('auto' length)."""
    import mne

    h = mne.filter.create_filter(None, sfreq, l_freq, h_freq, method="fir", fir_design="firwin",
                                 fir_window=fir_window, phase="zero", verbose=False)
    return (len(h) // 2) / sfreq


def _auto_h_trans(sfreq: float, h_freq: float) -> float:
    # MNE 'auto' low-pass transition bandwidth
    return min(max(0.25 * h_freq, 2.0), sfreq / 2.0 - h_freq)


def _filter_piece(x: np.ndarray, sfreq: float, l_freq: float | None, h_freq: float | None,
                  fir_window: str) -> np.ndarray:
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.filter.filter_data(np.ascontiguousarray(x, dtype=np.float64), sfreq, l_freq, h_freq,
                                      method="fir", fir_design="firwin", fir_window=fir_window, phase="zero",
                                      pad="reflect_limited", verbose=False)


# --------------------------------------------------------------------------- ocular copy
def build_ocular_copy(raw: Any, segment_info: Any | None, cfg: OcularConfig | Mapping[str, Any] | None = None,
                      *, exg_qc_result: Mapping[str, Mapping[str, Any]] | None = None,
                      bad_channels: Sequence[str] = ()):
    """Build the non-high-passed, low-passed, decimated ocular copy of one recording.

    Parameters
    ----------
    raw
        ``mne.io.Raw`` straight from ``read_raw_bdf`` (unfiltered, DC). It
        must not have been high-passed or re-referenced.
    segment_info
        Object with ``joins``, ``rail_spans`` (and optionally ``valid_spans``)
        in samples of ``raw``; see
        :func:`~preprocessing.exg_qc.valid_pieces`.
    cfg
        Ocular config (default: project ``configs/ocular.yaml``).
    exg_qc_result
        Result of :func:`~preprocessing.exg_qc.exg_qc` for this
        raw; computed here if ``None``.
    bad_channels
        Channels not to use (bad-channel rule, known-issue statuses).

    Returns
    -------
    mne.io.RawArray
        Channels ``VEOG``, ``HEOG`` (5 Hz), ``VEOG_lp15``, ``HEOG_lp15``
        (15 Hz), optionally the 15-Hz constituents, and the ``OC_VALID`` /
        ``OC_PIECE`` mask channels; volts; see the module docstring for the
        time base.
    """
    import mne

    c = as_ocular_config(cfg)
    oc = c.ocular_copy
    sfreq = float(raw.info["sfreq"])
    decim = int(oc.decim)
    new_sfreq = sfreq / decim
    stop_edge = oc.lowpass_fast_hz + _auto_h_trans(sfreq, oc.lowpass_fast_hz)
    if stop_edge > new_sfreq / 2.0:
        raise ValueError(f"decim={decim} aliases: the {oc.lowpass_fast_hz}-Hz low-pass stop band starts at "
                         f"{stop_edge:.2f} Hz, above the new Nyquist {new_sfreq / 2:.2f} Hz")
    n = int(raw.n_times)
    pieces = valid_pieces(n, segment_info)
    qc = exg_qc(raw, segment_info, c) if exg_qc_result is None else exg_qc_result
    src = choose_ocular_sources(qc, raw.ch_names, c, bad_channels)

    hl_slow = fir_half_length_s(sfreq, None, oc.lowpass_slow_hz, oc.fir_window)
    hl_fast = fir_half_length_s(sfreq, None, oc.lowpass_fast_hz, oc.fir_window)
    min_len = int(math.ceil(2 * hl_slow * sfreq)) + 1
    used_pieces = [(a, b) for a, b in pieces if b - a >= min_len]

    inputs = [ch for ch in c.channels.all_inputs if ch in raw.ch_names]
    X = raw.get_data(picks=inputs) if inputs else np.zeros((0, n))
    slow = np.zeros_like(X)
    fast = np.zeros_like(X)
    for a, b in used_pieces:
        if len(inputs):
            slow[:, a:b] = _filter_piece(X[:, a:b], sfreq, None, oc.lowpass_slow_hz, oc.fir_window)
            fast[:, a:b] = _filter_piece(X[:, a:b], sfreq, None, oc.lowpass_fast_hz, oc.fir_window)
    del X

    def sig(Y: np.ndarray, chans: Sequence[str]) -> np.ndarray:
        return np.mean([Y[inputs.index(ch)] for ch in chans], axis=0)

    def derive(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        veog = sig(Y, src.fp) - sig(Y, src.ref) if (src.fp and src.ref) else np.zeros(n)
        heog = Y[inputs.index(src.heog[0])] - Y[inputs.index(src.heog[1])] if src.heog else np.zeros(n)
        return veog, heog

    v5, h5 = derive(slow)
    v15, h15 = derive(fast)
    del slow

    # decimation phase: keep raw indices i with (first_samp + i) % decim == 0
    first = int(raw.first_samp)
    start0 = (-first) % decim
    idx = np.arange(start0, n, decim)
    valid = np.zeros(n, dtype=bool)
    piece_id = np.full(n, -1.0)
    for k, (a, b) in enumerate(used_pieces):
        valid[a:b] = True
        piece_id[a:b] = k

    names = list(DERIVED)
    types = ["eog"] * 4
    rows = [v5, h5, v15, h15]
    if oc.store_lp15_constituents:
        for ch in inputs:
            names.append(ch)
            types.append("eog" if ch in c.channels.heog_pair else "misc" if ch in c.channels.mastoids else "eeg")
            rows.append(fast[inputs.index(ch)])
    del fast
    data = np.vstack([np.asarray(r)[idx] for r in rows] + [valid[idx].astype(float)[None], piece_id[idx][None]])
    data[: len(rows), ~valid[idx]] = 0.0
    names += [VALID_CH, PIECE_CH]
    types += ["misc", "misc"]

    meta = {
        "format": OCULAR_FORMAT, "version": OCULAR_FORMAT_VERSION,
        "orig_sfreq": sfreq, "decim": decim, "sfreq": new_sfreq, "orig_first_samp": first,
        "orig_n_times": n, "decim_start_index": int(start0),
        "time_base": "ocular sample j (absolute, with first_samp) = BDF sample j*decim",
        "lowpass_slow_hz": oc.lowpass_slow_hz, "lowpass_fast_hz": oc.lowpass_fast_hz,
        "fir": {"design": oc.fir_design, "window": oc.fir_window, "phase": "zero", "length": "auto"},
        "half_length_slow_s": hl_slow, "half_length_fast_s": hl_fast, "highpass": None,
        "channel_lowpass_hz": {**{"VEOG": oc.lowpass_slow_hz, "HEOG": oc.lowpass_slow_hz,
                                  "VEOG_lp15": oc.lowpass_fast_hz, "HEOG_lp15": oc.lowpass_fast_hz},
                               **({ch: oc.lowpass_fast_hz for ch in inputs} if oc.store_lp15_constituents
                                  else {})},
        "pieces": [[int(a), int(b)] for a, b in used_pieces],
        "dropped_short_pieces": [[int(a), int(b)] for a, b in pieces if b - a < min_len],
        "sources": src.to_dict(),
        "exg_qc_pass": {ch: bool(r.get("pass", False)) for ch, r in qc.items()},
    }
    info = mne.create_info(names, new_sfreq, types, verbose=False)
    with info._unlock():
        info["lowpass"] = float(oc.lowpass_fast_hz)
        info["highpass"] = 0.0
        info["description"] = json.dumps(meta, sort_keys=True)
        if raw.info.get("meas_date") is not None:
            info["meas_date"] = raw.info["meas_date"]
    first_oc = (first + start0) // decim
    out = mne.io.RawArray(data, info, first_samp=first_oc, verbose=False)
    # human-readable BAD annotations for the invalid stretches (onsets relative to the copy's first sample)
    v = data[names.index(VALID_CH)] > 0.5
    if (~v).any():
        edges = np.flatnonzero(np.diff(np.concatenate([[1], v.astype(int), [1]])))
        starts, stops = edges[0::2], edges[1::2]
        ann = mne.Annotations(onset=starts / new_sfreq, duration=(stops - starts) / new_sfreq,
                              description=["BAD_ocular_invalid"] * len(starts))
        out.set_annotations(ann, verbose=False)
    return out


def ocular_meta(ocular_raw: Any) -> dict:
    """Build parameters stored by :func:`build_ocular_copy` (raises if the Raw is not an ocular copy)."""
    desc = ocular_raw.info.get("description")
    try:
        meta = json.loads(desc) if desc else None
    except (TypeError, ValueError):
        meta = None
    if not isinstance(meta, dict) or meta.get("format") != OCULAR_FORMAT:
        raise ValueError("not an alpha_state ocular copy (info['description'] lacks the ocular metadata)")
    return meta


def save_ocular_copy(ocular_raw: Any, fname: Any, overwrite: bool = False) -> None:
    """Save as float32 FIF (``fname`` should end in ``_raw.fif`` / ``-raw.fif`` or ``_eog.fif``)."""
    ocular_raw.save(fname, fmt="single", overwrite=overwrite, verbose=False)
