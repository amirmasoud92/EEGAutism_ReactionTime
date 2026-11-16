"""Pure, vectorised event-timing primitives used by trial builders.

All inputs are integer sample indices on one recording's (concatenated)
sample clock; outputs are numpy arrays aligned with the trial samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


def as_sorted_samples(x: np.ndarray | list[int]) -> np.ndarray:
    """int64 copy of ``x``, which must be sorted ascending."""
    a = np.asarray(x, dtype=np.int64).reshape(-1)
    if a.size > 1 and np.any(np.diff(a) < 0):
        raise ValueError("sample arrays must be sorted ascending")
    return a


def next_after(samples: np.ndarray, events: np.ndarray) -> np.ndarray:
    """For each sample, the first event strictly after it (``-1`` if none)."""
    ev = as_sorted_samples(events)
    i = np.searchsorted(ev, samples, side="right")
    return np.where(i < ev.size, ev[np.minimum(i, max(ev.size - 1, 0))] if ev.size else -1, -1).astype(np.int64)


def last_before(samples: np.ndarray, events: np.ndarray) -> np.ndarray:
    """For each sample, the last event strictly before it (``-1`` if none)."""
    ev = as_sorted_samples(events)
    i = np.searchsorted(ev, samples, side="left") - 1
    return np.where(i >= 0, ev[np.maximum(i, 0)] if ev.size else -1, -1).astype(np.int64)


def count_in(events: np.ndarray, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Number of events with ``start <= e < stop`` (vectorised)."""
    ev = as_sorted_samples(events)
    return (np.searchsorted(ev, stop, side="left") - np.searchsorted(ev, start, side="left")).astype(np.int64)


def samples_to_ms(n: np.ndarray | float, sfreq: float) -> np.ndarray:
    return np.asarray(n, dtype=float) * 1000.0 / float(sfreq)


@dataclass(frozen=True)
class MatchResult:
    """Per-trial response matching output (arrays aligned with the trials).

    ``rt_ms[name]`` / ``response_sample[name]`` hold, per window, the latency /
    sample of the first press in ``[onset, bound)`` whose latency is in the
    closed window (NaN / -1 if none).
    """

    bound_sample: np.ndarray
    n_press_until_bound: np.ndarray
    first_press_lat_ms: np.ndarray
    raw_first_press_lat_ms: np.ndarray
    raw_first_press_after_bound: np.ndarray
    n_anticipations: np.ndarray
    n_late_presses: np.ndarray
    n_extra_presses: np.ndarray
    rt_ms: dict[str, np.ndarray]
    response_sample: dict[str, np.ndarray]


def match_responses(
    trial_samples: np.ndarray,
    bound_samples: np.ndarray,
    press_samples: np.ndarray,
    sfreq: float,
    windows: Mapping[str, tuple[float, float]],
    primary: str,
) -> MatchResult:
    """Match button presses to trials with the primary response rule.

    Rule (``configs/events.yaml``): the RT of a trial is the latency of the
    first press whose latency lies in the closed window ``[lo, hi]`` ms, whose
    sample is ``>= onset`` and ``< bound`` (onset of the next response-bound
    event). Presses with latency ``< lo`` are skipped (anticipations).

    Parameters
    ----------
    trial_samples
        Trial onsets (sorted).
    bound_samples
        Onsets that close a trial's response window (sorted). A trial's bound
        is the first of these strictly after its onset; at the end of a run
        the window is open-ended.
    press_samples
        Press onsets (sorted).
    sfreq
        Sampling rate (Hz).
    windows
        ``name -> (lo_ms, hi_ms)``.
    primary
        Name of the primary window (anticipation / late / extra counts use it).

    Returns
    -------
    MatchResult
    """
    s = as_sorted_samples(trial_samples)
    b = next_after(s, bound_samples)
    p = as_sorted_samples(press_samples)
    n = s.size
    end = np.where(b >= 0, b, np.iinfo(np.int64).max)
    i0 = np.searchsorted(p, s, side="left")
    i1 = np.searchsorted(p, end, side="left")
    n_until = (i1 - i0).astype(np.int64)
    lo_p, hi_p = windows[primary]
    first = np.full(n, np.nan)
    raw_first = np.full(n, np.nan)
    raw_after = np.zeros(n, dtype=object)
    n_ant = np.zeros(n, np.int64)
    n_late = np.zeros(n, np.int64)
    n_extra = np.zeros(n, np.int64)
    rts = {w: np.full(n, np.nan) for w in windows}
    rs = {w: np.full(n, -1, np.int64) for w in windows}
    for k in range(n):
        if i0[k] < p.size:
            raw_first[k] = (p[i0[k]] - s[k]) * 1000.0 / sfreq
            raw_after[k] = bool(b[k] >= 0 and p[i0[k]] >= b[k])
        else:
            raw_after[k] = None
        if n_until[k] == 0:
            continue
        lat = (p[i0[k]: i1[k]] - s[k]) * 1000.0 / sfreq
        first[k] = lat[0]
        n_ant[k] = int(np.sum(lat < lo_p - _EPS))
        n_late[k] = int(np.sum(lat > hi_p + _EPS))
        for w, (lo, hi) in windows.items():
            ok = np.flatnonzero((lat >= lo - _EPS) & (lat <= hi + _EPS))
            if ok.size:
                rts[w][k] = lat[ok[0]]
                rs[w][k] = p[i0[k] + ok[0]]
                if w == primary:
                    n_extra[k] = int(lat.size - ok[0] - 1)
    return MatchResult(bound_sample=b, n_press_until_bound=n_until, first_press_lat_ms=first,
                       raw_first_press_lat_ms=raw_first, raw_first_press_after_bound=raw_after,
                       n_anticipations=n_ant, n_late_presses=n_late, n_extra_presses=n_extra,
                       rt_ms=rts, response_sample=rs)
