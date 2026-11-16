"""Zero-phase FIR filtering, run separately inside every valid span.

:func:`filter_raw_per_segment` checks that the annotations of the Raw object reproduce
:attr:`SegmentInfo.valid_spans` (:func:`~preprocessing.segments.check_annotations_match`) and then
calls MNE's ``raw.filter`` with ``skip_by_annotation=SKIP_BY_ANNOTATION``, which
filters each contiguous span on its own (edge padding ``reflect_limited`` at both
span ends), so a step at a join does not leak into the neighbouring segment.
Samples inside BAD spans are not filtered.
"""

from __future__ import annotations

import numpy as np

from preprocessing.config import FilterSpec
from preprocessing.segments import SKIP_BY_ANNOTATION, SegmentInfo, check_annotations_match


def filter_raw_per_segment(raw, seg: SegmentInfo, spec: FilterSpec, picks, *, check: bool = True, n_jobs: int = 1):
    """Band-pass ``raw`` in place, separately in each valid span; returns ``raw``.

    Parameters
    ----------
    raw
        Preloaded Raw annotated with :func:`~preprocessing.segments.annotate_raw`.
    seg
        Segment info of the recording.
    spec
        Filter parameters.
    picks
        Channels to filter (names or indices).
    check
        Verify the annotation-based segmentation first (raises ``ValueError``).
    """
    if check:
        check_annotations_match(raw, seg)
    raw.filter(picks=picks, skip_by_annotation=list(SKIP_BY_ANNOTATION), n_jobs=n_jobs, verbose="error",
               **spec.mne_kwargs())
    return raw


def notch_raw_per_segment(raw, seg: SegmentInfo, freqs, picks, *, check: bool = True, n_jobs: int = 1):
    """FIR notch (zero phase, firwin, Hamming) at ``freqs`` in place, separately in each valid span."""
    freqs = [f for f in freqs if f < raw.info["sfreq"] / 2]
    if not freqs:
        return raw
    if check:
        check_annotations_match(raw, seg)
    raw.notch_filter(freqs, picks=picks, method="fir", phase="zero", fir_window="hamming", fir_design="firwin",
                     skip_by_annotation=list(SKIP_BY_ANNOTATION), n_jobs=n_jobs, verbose="error")
    return raw


def fir_kernel(spec: FilterSpec, sfreq: float) -> np.ndarray:
    """The FIR kernel MNE designs for ``spec`` at ``sfreq``."""
    from mne.filter import create_filter

    kw = spec.mne_kwargs()
    kw.pop("pad")
    kw.pop("method")
    return create_filter(None, sfreq, verbose="error", **kw)


def filter_half_length_s(spec: FilterSpec, sfreq: float) -> float:
    """Half-length of the FIR kernel in seconds (the window edge margin, 1.65 s for 1-40 Hz at 512 Hz)."""
    return (len(fir_kernel(spec, sfreq)) // 2) / sfreq
