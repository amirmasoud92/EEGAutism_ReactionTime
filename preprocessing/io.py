"""Storage of the per-recording preprocessing outputs (writing, reading, cleaning from the stored parts).

Layout (``<derivatives>/preproc/<task>/<subject>/``; names keep the BIDS stem):

=================  =============================================  ==========================================
key                file                                           content
=================  =============================================  ==========================================
``preica``         ``<stem>_desc-preica_raw.fif``                  analysis copy: 1-40 Hz per-segment FIR, CMS
                                                                  reference (not re-referenced), joins and BAD
                                                                  spans annotated, bad channels in
                                                                  ``info['bads']``, float32, Status kept
``ica``            ``<stem>_ica.fif``                              MNE ICA (``exclude`` = ICLabel rule)
``iclabel``        ``<stem>_iclabel.tsv``                          per component: 7 ICLabel probabilities, label,
                                                                  excluded, explained variance
``segments``       ``<stem>_segments.json``                        :class:`SegmentInfo`
``ocular``         ``<stem>_desc-ocular_raw.fif``                  non-high-passed ocular copy (ocular module)
``icasel``         ``<stem>_icasel.json``                          ICA re-selection: exclusions applied by
                                                                  load_clean (posterior-alpha veto) and the
                                                                  ``BAD_icafit`` spans (ica_selection module)
``qc``             ``<stem>_qc.json``                              QC record (written last)
=================  =============================================  ==========================================

:func:`~preprocessing.clean.load_clean` (re-exported here) rebuilds the
cleaned data from the stored parts (:func:`clean_from_parts`):

1. robust average reference (mean of the good EEG channels, subtracted from every
   EEG channel) - the reference the ICA was fitted on;
2. ``apply_ica``: remove the excluded components (good channels only);
3. ``interpolate``: spherical-spline interpolation of the bad EEG channels;
4. robust average reference again, over the originally good channels (a no-op on
   them; it puts interpolated channels on the same reference).

The sample axis is never changed: ``n_times`` and ``first_samp`` equal the
BDF's, so trial-table event samples index the returned data directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from preprocessing.segments import SegmentInfo
from utils.bids_index import Recording

OUTPUT_FILES: dict[str, str] = {
    "preica": "_desc-preica_raw.fif",
    "ica": "_ica.fif",
    "iclabel": "_iclabel.tsv",
    "segments": "_segments.json",
    "ocular": "_desc-ocular_raw.fif",
    "icasel": "_icasel.json",
    "qc": "_qc.json",
}

#: Names served from :mod:`preprocessing.clean` (module ``__getattr__``).
_CLEAN_NAMES = frozenset({"load_clean", "_resolve"})


def __getattr__(name: str):
    if name in _CLEAN_NAMES:
        from preprocessing import clean

        return getattr(clean, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class PreprocessingOutputError(RuntimeError):
    """Raised when stored outputs are missing or inconsistent."""


# --------------------------------------------------------------------------- paths
def preproc_root(derivatives: str | os.PathLike[str], subdir: str = "preproc") -> Path:
    """``<derivatives>/<subdir>``."""
    return Path(derivatives) / subdir


def recording_outputs(root: str | os.PathLike[str], rec: Recording | tuple[str, str, str]) -> dict[str, Path]:
    """Output paths of one recording (``rec`` may be a Recording or ``(task, subject, stem)``)."""
    task, subject, stem = (rec.task, rec.subject, rec.stem) if isinstance(rec, Recording) else rec
    d = Path(root) / task / subject
    return {k: d / f"{stem}{suf}" for k, suf in OUTPUT_FILES.items()}


# --------------------------------------------------------------------------- writing
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dumps_json(obj) -> str:
    """JSON text (indented; numpy types converted; NaN written as null)."""
    def clean(x):
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, np.floating) and not np.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    return json.dumps(clean(obj), indent=1, default=_json_default, allow_nan=False)


def save_raw_fif(raw, path: Path, fmt: str = "single") -> None:
    """``raw.save`` (float32 by default) without splitting and without MNE's name warning."""
    raw.save(path, fmt=fmt, overwrite=True, split_size="2GB", verbose="error")


def save_ica(ica, path: Path) -> None:
    """``ica.save`` (overwrite, quiet)."""
    ica.save(path, overwrite=True, verbose="error")


def save_json(obj, path: Path) -> None:
    """Write :func:`dumps_json` text."""
    Path(path).write_text(dumps_json(obj), encoding="utf-8")


# --------------------------------------------------------------------------- reading
def read_qc(path: str | os.PathLike[str]) -> dict:
    """Read a QC (or any) JSON record."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_raw_fif(path: str | os.PathLike[str]):
    import mne

    return mne.io.read_raw_fif(path, preload=True, verbose="error")


def read_ica(path: str | os.PathLike[str]):
    import mne

    return mne.preprocessing.read_ica(path, verbose="error")


def read_segments(path: str | os.PathLike[str]) -> SegmentInfo:
    return SegmentInfo.from_json(path)


# --------------------------------------------------------------------------- cleaning
def clean_from_parts(raw, ica=None, *, apply_ica: bool = True, interpolate: bool = True,
                     exclude: list[int] | None = None, origin: str = "auto"):
    """Clean a pre-ICA analysis copy in place (steps in the module docstring); returns ``raw``.

    ``raw.info['temp']`` receives ``{"interpolated": [...], "ica_excluded": [...],
    "ica_applied": bool, "reference_channels": [...]}``.
    """
    from preprocessing.reference import eeg_picks, good_eeg_picks, robust_average_reference

    eeg = [raw.ch_names[i] for i in eeg_picks(raw.info)]
    bad_eeg = [c for c in raw.info["bads"] if c in eeg]
    other_bads = [c for c in raw.info["bads"] if c not in eeg]
    good = [raw.ch_names[i] for i in good_eeg_picks(raw.info)]
    robust_average_reference(raw)
    excluded: list[int] = []
    if apply_ica and ica is not None:
        miss = sorted(set(ica.ch_names) - set(good))
        if miss:
            raise PreprocessingOutputError(f"ICA channels not among the good EEG channels: {miss}")
        excluded = sorted(int(i) for i in (ica.exclude if exclude is None else exclude))
        if excluded:
            # MNE's apply(exclude=...) removes the UNION of `exclude` and ica.exclude, so a smaller (vetoed) set must be
            # put into ica.exclude itself for the call
            saved = list(ica.exclude)
            ica.exclude = list(excluded)
            try:
                ica.apply(raw, verbose="error")
            finally:
                ica.exclude = saved
    interpolated: list[str] = []
    if interpolate and bad_eeg:
        raw.interpolate_bads(reset_bads=True, mode="accurate", origin=origin, verbose="error")
        interpolated = bad_eeg
        with raw.info._unlock():
            raw.info["bads"] = list(other_bads)      # non-EEG bads (e.g. a faulty EXG) stay marked
    robust_average_reference(raw, exclude=interpolated)
    with raw.info._unlock():
        raw.info["temp"] = {"interpolated": interpolated, "ica_excluded": excluded,
                            "ica_applied": bool(apply_ica and ica is not None), "reference_channels": good}
    return raw
