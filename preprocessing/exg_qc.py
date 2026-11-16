"""Per-file QC of the external electrodes EXG1-EXG4.

Processing rules:

- the 1-40 Hz FIR runs separately inside each *valid piece* (segments between
  Status bit-16 joins, minus whole-cap rail spans), after per-piece DC removal;
- 2-s chunks never straddle a piece boundary, and rail spans (cap
  disconnected, all EEG at the rail) are excluded from every denominator, so a
  disconnected cap is not blamed on an EXG channel.

Metrics per channel (data in the raw, DC-coupled, CMS-referenced form):

``frac_sat``
    fraction of valid samples with ``|x| > sat_threshold_mv`` (250 mV).
``median_chunk_sd_uv``
    median over non-overlapping 2-s chunks of the SD of the 1-40 Hz signal.
``frac_bad_chunks``
    fraction of chunks with filtered SD > 200 uV, or ``|raw chunk mean|`` >
    250 mV, or a raw sample-to-sample step larger than
    ``bad_chunk_step_mv``. The step criterion targets intermittent
    rail-to-rail swings (a channel can be several per cent saturated while its median
    chunk SD is normal), which a median-only metric misses.

Rule: a channel fails if ``frac_sat > max_frac_sat`` or
``frac_bad_chunks > max_frac_bad_chunks`` (resting on at least ``min_bad_chunks``
chunks) or the median chunk SD is outside ``[min, max]``. A missing channel fails with reason
``missing``. A failed channel triggers the ocular-channel fallback in
:mod:`preprocessing.ocular`; it never excludes the file.

Sample indices in ``segment_info`` are indices into ``raw``'s data array
(equal to BDF samples for an uncropped ``read_raw_bdf`` raw, whose
``first_samp`` is 0).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from utils.config_ocular import OcularConfig, as_ocular_config

__all__ = ["valid_pieces", "exg_qc", "QC_METRIC_KEYS"]

QC_METRIC_KEYS = ("present", "pass", "fail_reasons", "n_valid_samples", "n_chunks", "dc_mean_mv",
                  "frac_sat", "frac_sat_pos", "frac_sat_neg", "n_sat_onsets", "median_chunk_sd_uv",
                  "p95_chunk_sd_uv", "frac_bad_chunks", "frac_chunks_high_sd", "frac_chunks_rail_mean",
                  "frac_chunks_step", "max_step_mv", "n_bad_chunks")


# --------------------------------------------------------------------------- spans
def _as_spans(v: Any) -> list[tuple[int, int]]:
    if v is None:
        return []
    arr = np.asarray(v, dtype=np.int64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 2)
    return [(int(a), int(b)) for a, b in arr]


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _subtract(pieces: list[tuple[int, int]], cut: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for a, b in pieces:
        cur = a
        for c, d in cut:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def _intersect(pieces: list[tuple[int, int]], keep: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for a, b in pieces:
        for c, d in keep:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                out.append((lo, hi))
    return sorted(out)


def valid_pieces(n_times: int, segment_info: Any | None) -> list[tuple[int, int]]:
    """Half-open sample spans ``[start, stop)`` inside which filters may run.

    The pieces are the segments between joins (``segment_info.joins``, the
    first sample of each new segment) and the file edges, minus
    ``segment_info.rail_spans`` (half-open ``(start, stop)``). If the object
    has a ``valid_spans`` attribute that is not ``None`` (as
    :class:`preprocessing.segments.SegmentInfo` always has), the
    pieces are intersected with it, so samples outside ``valid_spans`` (e.g.
    flat or known-issue unusable spans) are never used; an empty
    ``valid_spans`` means no usable data.
    ``segment_info=None`` means one piece covering the whole file.

    Any object with these attributes works (duck typing); the attributes may be
    missing.
    """
    n = int(n_times)
    if n <= 0:
        return []
    if segment_info is None:
        return [(0, n)]
    j = getattr(segment_info, "joins", None)
    joins = np.asarray([] if j is None else j, dtype=np.int64).ravel()
    joins = np.unique(joins[(joins > 0) & (joins < n)])
    bounds = [0, *joins.tolist(), n]
    pieces = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    rails = _merge((max(0, a), min(n, b)) for a, b in _as_spans(getattr(segment_info, "rail_spans", None)))
    pieces = _subtract(pieces, rails)
    vs_attr = getattr(segment_info, "valid_spans", None)
    if vs_attr is not None:  # authoritative when present (SegmentInfo also removes flat / unusable spans), even if empty
        vs = _as_spans(vs_attr)
        pieces = _intersect(pieces, _merge((max(0, a), min(n, b)) for a, b in vs))
    return pieces


# --------------------------------------------------------------------------- filtering
def _bandpass_piece(x: np.ndarray, sfreq: float, l_freq: float | None, h_freq: float | None,
                    fir_window: str = "hamming") -> np.ndarray:
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short pieces: MNE warns that the filter is longer than the signal
        return mne.filter.filter_data(np.ascontiguousarray(x, dtype=np.float64), sfreq, l_freq, h_freq,
                                      method="fir", fir_design="firwin", fir_window=fir_window, phase="zero",
                                      pad="reflect_limited", verbose=False)


# --------------------------------------------------------------------------- QC
def _channel_qc(x_uv: np.ndarray, pieces: Sequence[tuple[int, int]], sfreq: float, q) -> dict:
    sat_uv = q.sat_threshold_mv * 1e3
    chunk = int(round(q.chunk_s * sfreq))
    n_valid = int(sum(b - a for a, b in pieces))
    res: dict[str, Any] = {"present": True, "n_valid_samples": n_valid}
    if n_valid == 0:
        res.update(n_chunks=0, dc_mean_mv=np.nan, frac_sat=np.nan, frac_sat_pos=np.nan, frac_sat_neg=np.nan,
                   n_sat_onsets=0, median_chunk_sd_uv=np.nan, p95_chunk_sd_uv=np.nan, frac_bad_chunks=np.nan,
                   frac_chunks_high_sd=np.nan, frac_chunks_rail_mean=np.nan, frac_chunks_step=np.nan,
                   max_step_mv=np.nan)
        return res
    n_pos = n_neg = n_onsets = 0
    total = 0.0
    sds, means, steps = [], [], []
    max_step = 0.0
    for a, b in pieces:
        seg = x_uv[a:b]
        total += float(seg.sum())
        pos, neg = seg > sat_uv, seg < -sat_uv
        n_pos += int(pos.sum())
        n_neg += int(neg.sum())
        sat = pos | neg
        n_onsets += int(sat[0]) + int(np.sum(sat[1:] & ~sat[:-1]))
        if b - a > 1:
            d = np.abs(np.diff(seg))
            max_step = max(max_step, float(d.max()))
        k = (b - a) // chunk
        if k == 0:
            continue
        filt = _bandpass_piece(seg - seg.mean(), sfreq, q.band_hz[0], q.band_hz[1])
        f = filt[: k * chunk].reshape(k, chunk)
        r = seg[: k * chunk].reshape(k, chunk)
        sds.append(f.std(axis=1))
        means.append(r.mean(axis=1))
        if b - a > 1:
            dd = np.abs(np.diff(r, axis=1)).max(axis=1)
        else:
            dd = np.zeros(k)
        steps.append(dd)
    res["dc_mean_mv"] = total / n_valid / 1e3
    res["frac_sat_pos"] = n_pos / n_valid
    res["frac_sat_neg"] = n_neg / n_valid
    res["frac_sat"] = (n_pos + n_neg) / n_valid
    res["n_sat_onsets"] = n_onsets
    res["max_step_mv"] = max_step / 1e3
    if not sds:
        res.update(n_chunks=0, median_chunk_sd_uv=np.nan, p95_chunk_sd_uv=np.nan, frac_bad_chunks=np.nan,
                   frac_chunks_high_sd=np.nan, frac_chunks_rail_mean=np.nan, frac_chunks_step=np.nan)
        return res
    sd = np.concatenate(sds)
    mean = np.concatenate(means)
    step = np.concatenate(steps)
    high = sd > q.bad_chunk_sd_uv
    rail = np.abs(mean) > q.bad_chunk_absmean_mv * 1e3
    stp = step > q.bad_chunk_step_mv * 1e3 if q.bad_chunk_step_mv is not None else np.zeros_like(high)
    bad = high | rail | stp
    res.update(n_chunks=int(sd.size), median_chunk_sd_uv=float(np.median(sd)),
               p95_chunk_sd_uv=float(np.percentile(sd, 95)), frac_bad_chunks=float(bad.mean()),
               frac_chunks_high_sd=float(high.mean()), frac_chunks_rail_mean=float(rail.mean()),
               frac_chunks_step=float(stp.mean()), n_bad_chunks=int(bad.sum()))
    return res


def _decide(res: dict, q) -> dict:
    reasons: list[str] = []
    if not res.get("present", False):
        reasons.append("missing")
    elif res.get("n_chunks", 0) == 0:
        reasons.append("no_valid_data")
    else:
        if res["frac_sat"] > q.max_frac_sat:
            reasons.append("frac_sat")
        n_bad = res.get("n_bad_chunks", int(round(res["frac_bad_chunks"] * res["n_chunks"])))
        if res["frac_bad_chunks"] > q.max_frac_bad_chunks and n_bad >= getattr(q, "min_bad_chunks", 1):
            reasons.append("frac_bad_chunks")
        if res["median_chunk_sd_uv"] < q.min_median_chunk_sd_uv:
            reasons.append("median_chunk_sd_low")
        if res["median_chunk_sd_uv"] > q.max_median_chunk_sd_uv:
            reasons.append("median_chunk_sd_high")
    res["fail_reasons"] = reasons
    res["pass"] = not reasons
    return res


def exg_qc(raw: Any, segment_info: Any | None, cfg: OcularConfig | Mapping[str, Any] | None = None,
           channels: Sequence[str] | None = None) -> dict[str, dict]:
    """QC metrics and pass/fail for each EXG channel of one recording.

    Parameters
    ----------
    raw
        ``mne.io.Raw`` of the raw BDF (unfiltered, DC; preloaded or not). EXG
        channel types do not matter.
    segment_info
        Object with ``joins``, ``rail_spans`` and optionally ``valid_spans``
        (see :func:`valid_pieces`), or ``None``.
    cfg
        :class:`~utils.config_ocular.OcularConfig`, its mapping, or
        ``None`` (project ``configs/ocular.yaml``).
    channels
        Channels to check (default ``cfg.exg_qc.channels``: EXG1-EXG4).

    Returns
    -------
    dict
        ``{channel: {"present", "pass", "fail_reasons", metrics...}}`` (keys
        in :data:`QC_METRIC_KEYS`). Thresholds are not repeated per channel;
        they are in the config.
    """
    c = as_ocular_config(cfg)
    q = c.exg_qc
    chans = list(channels) if channels is not None else list(q.channels)
    sfreq = float(raw.info["sfreq"])
    pieces = valid_pieces(raw.n_times, segment_info)
    present = [ch for ch in chans if ch in raw.ch_names]
    out: dict[str, dict] = {}
    data = raw.get_data(picks=present) * 1e6 if present else np.empty((0, raw.n_times))
    for ch in chans:
        if ch in present:
            res = _channel_qc(data[present.index(ch)], pieces, sfreq, q)
        else:
            res = {k: np.nan for k in QC_METRIC_KEYS if k not in ("present", "pass", "fail_reasons")}
            res.update(present=False, n_valid_samples=0, n_chunks=0, n_sat_onsets=0)
        out[ch] = _decide(res, q)
    return out
