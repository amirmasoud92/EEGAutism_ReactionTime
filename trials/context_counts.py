"""Stimulus and press counts in the alpha context interval (``n_stim10`` / ``n_press10``).

``a_ctx10`` (:mod:`feature_extraction.decomposition`) integrates stimulus- and press-evoked activity, so it is
always fitted with the counts of stimuli and of presses in the same 10 s (20 s in the sensitivity variant). This
module reads event onsets only (events.tsv ``sample`` and ``value``; no EEG, no RT / hit / miss column, no alpha
value).

Interval (identical to the ``actx<L>_log`` interval of :func:`feature_extraction.prestimulus_alpha.
compute_alpha_features`, via :func:`feature_extraction.spectral.context_intervals`): ``[max(w - L, span start),
w)``, where ``w`` = onset + round(primary window start ms x sfreq / 1000) is the primary window start and the span is
the preprocessing valid span (between joins and bad spans) that contains ``w``. Events at ``w`` are not counted.
Rows whose window start is in no valid span get NA counts.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from feature_extraction import spectral as sp

KEYS: tuple[str, ...] = ("subject", "task", "run", "sample")


def context_event_counts(onsets: np.ndarray, window_start_ms: float, sfreq: float, seg, stim_samples: np.ndarray,
                         press_samples: np.ndarray, lengths_s: Sequence[float] = (10.0, 20.0)) -> pd.DataFrame:
    """``n_stim<L>``, ``n_press<L>`` and ``ctx<L>_span_s`` (interval length after truncation at the span start) for
    each onset. Inputs are event onsets (samples) only."""
    on = np.asarray(onsets, np.int64)
    w = on + sp.ms_to_samples(window_start_ms, sfreq)
    cols: dict[str, object] = {}
    for L in lengths_s:
        tag = f"{float(L):g}"
        c0, c1, ok = sp.context_intervals(w, seg, float(L), sfreq)
        ns = sp.count_in_intervals(stim_samples, c0, c1)
        npr = sp.count_in_intervals(press_samples, c0, c1)
        cols[f"n_stim{tag}"] = pd.Series(ns, dtype="Int64").mask(~ok).array
        cols[f"n_press{tag}"] = pd.Series(npr, dtype="Int64").mask(~ok).array
        cols[f"ctx{tag}_span_s"] = np.where(ok, (c1 - c0) / float(sfreq), np.nan)
    return pd.DataFrame(cols)


def recording_event_onsets(events_path: str | os.PathLike[str], task_cfg) -> tuple[np.ndarray, np.ndarray]:
    """``(stimulus onsets, press onsets)`` (samples) of the task's ``stimulus_codes`` / ``response_codes``."""
    from trials.io import read_events_tsv

    ev = read_events_tsv(events_path)
    stim = np.sort(ev.loc[ev["value"].isin(list(task_cfg.stimulus_codes)), "sample"].to_numpy(np.int64))
    press = np.sort(ev.loc[ev["value"].isin(list(task_cfg.response_codes)), "sample"].to_numpy(np.int64))
    return stim, press


def recording_context_counts(keys: pd.DataFrame, events_path: str | os.PathLike[str], task_cfg, seg, sfreq: float,
                             lengths_s: Sequence[float] = (10.0, 20.0)) -> pd.DataFrame:
    """Counts for the rows of ``keys`` (``subject``, ``task``, ``run``, ``sample``; one recording).

    ``keys`` may carry other columns; only the key columns are used and returned.
    """
    k = keys[list(KEYS)].copy().reset_index(drop=True)
    if k[["subject", "task", "run"]].drop_duplicates().shape[0] > 1:
        raise ValueError("recording_context_counts takes the rows of one recording")
    stim, press = recording_event_onsets(events_path, task_cfg)
    w0 = float(task_cfg.prestim_windows["primary"][0])
    c = context_event_counts(k["sample"].to_numpy(np.int64), w0, sfreq, seg, stim, press, lengths_s)
    return pd.concat([k, c], axis=1)
