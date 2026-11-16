""":func:`load_clean`: the cleaned continuous data of one recording, rebuilt from the stored preprocessing outputs.

``io.load_clean`` re-exports it.

Steps (:func:`~preprocessing.io.clean_from_parts`): robust average reference -> ICA (the exclusions of
the re-selection output ``<stem>_icasel.json``: ICLabel eye components, sparing those with posterior alpha) ->
spherical-spline interpolation -> reference again. Then the ICA fit-rejected chunks are added as ``BAD_icafit``
annotations, in the ICA *and* the no-ICA variant.

Checks:

- ``require_complete`` (default): the QC record must say ``complete`` and carry no incomplete-making flag, and a
  recording with an ICA must have its re-selection output; otherwise :class:`PreprocessingOutputError`.
- ``apply_ica=True`` always requires an ICA with status ``ok`` and its file (no silent fallback to no-ICA data).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from preprocessing.io import (
    OUTPUT_FILES,
    PreprocessingOutputError,
    clean_from_parts,
    preproc_root,
    read_ica,
    read_qc,
    read_raw_fif,
    read_segments,
    recording_outputs,
)
from utils.bids_index import Recording

__all__ = ["load_clean"]


def _resolve(recording, root: Path | None):
    """Output paths of ``recording`` (Recording, (subject, task, run), or a QC/FIF path)."""
    if isinstance(recording, (str, os.PathLike)) and Path(recording).suffix in (".json", ".fif"):
        p = Path(recording)
        stem = p.name
        for suf in OUTPUT_FILES.values():
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        return {k: p.parent / f"{stem}{suf}" for k, suf in OUTPUT_FILES.items()}
    if root is None:
        from preprocessing.config import load_preprocessing_config
        from utils.config import load_project_config

        pc = load_project_config()
        root = preproc_root(pc.paths.derivatives, load_preprocessing_config(root=pc.root).storage_subdir)
    if isinstance(recording, Recording):
        return recording_outputs(root, recording)
    subject, task, run = recording
    d = Path(root) / task / subject
    cands = sorted(d.glob(f"{subject}_task-{task}*_qc.json"))
    for c in cands:
        stem = c.name[: -len(OUTPUT_FILES["qc"])]
        q = read_qc(c)
        if int(q.get("recording", {}).get("run", -1)) == int(run):
            return recording_outputs(root, (task, subject, stem))
    raise PreprocessingOutputError(f"no preprocessing outputs for {subject} {task} run {run} under {d}")


def _add_icafit_annotations(raw, spans, description: str) -> int:
    import mne

    from preprocessing.segments import _onset_s

    sp = np.asarray(spans, np.int64).reshape(-1, 2)
    if not len(sp):
        return 0
    sf = float(raw.info["sfreq"])
    new = mne.Annotations(onset=_onset_s(raw, sp[:, 0]), duration=(sp[:, 1] - sp[:, 0]) / sf,
                          description=[description] * len(sp), orig_time=raw.annotations.orig_time)
    raw.set_annotations(raw.annotations + new)
    return int(len(sp))


def load_clean(recording, apply_ica: bool = True, interpolate: bool = True, *,
               root: str | os.PathLike[str] | None = None, verify: bool = True,
               require_complete: bool = True, annotate_icafit: bool = True,
               ica_selection: str = "icasel"):
    """Cleaned continuous data of one recording (see the module docstring).

    Parameters
    ----------
    recording
        A :class:`~utils.bids_index.Recording`, a ``(subject, task, run)``
        tuple, or the path of any stored output of the recording.
    apply_ica
        Remove the excluded ICA components (``False`` = the no-ICA variant). Requires an ICA with status ``ok``.
    interpolate
        Interpolate bad EEG channels (spherical spline). Without it, bad channels
        stay in ``info['bads']`` (re-referenced, but not used in the reference).
        Note: bad ROI channels are dropped from the alpha ROI; the feature steps pass ``interpolate=False``.
    root
        Preprocessing output root (default: ``<derivatives>/preproc``).
    verify
        Check ``n_times`` / ``first_samp`` / sfreq against the QC record (BDF header)
        and the segment file.
    require_complete
        Refuse an incomplete recording, or one without its ICA re-selection output (default ``True``).
    annotate_icafit
        Add the ``BAD_icafit`` annotations of the re-selection output (default ``True``; both variants).
    ica_selection
        ``"icasel"`` (default): exclusions and ``BAD_icafit`` spans of the re-selection output.
        ``"ica_file"``: the exclusions stored in the ICA file (plain ICLabel rule, before the posterior-alpha
        exception) and no ``BAD_icafit`` annotations; the re-selection output is then not required.

    Returns
    -------
    mne.io.Raw
        Preloaded float64 Raw with the original sample axis and annotations. ``info['temp']`` holds
        ``interpolated``, ``ica_excluded``, ``ica_applied``, ``reference_channels``, ``ica_vetoed``,
        ``ica_selection_source`` (``"icasel"`` or ``"ica_file"``), ``n_icafit_spans`` and ``qc_flags``.
    """
    from preprocessing.recording import INCOMPLETE_FLAGS, qc_flags

    paths = _resolve(recording, Path(root) if root is not None else None)
    for k in ("preica", "qc", "segments"):
        if not paths[k].is_file():
            raise PreprocessingOutputError(f"missing {k} output: {paths[k].name}")
    qc = read_qc(paths["qc"])
    flags = qc_flags(qc)
    ica_status = (qc.get("ica") or {}).get("status")
    if require_complete:
        if not qc.get("complete", False):
            raise PreprocessingOutputError(f"{paths['qc'].name}: recording is not complete "
                                           f"(status {qc.get('status')!r}, flags {flags})")
        bad = [f for f in flags if f in INCOMPLETE_FLAGS]
        if bad:
            raise PreprocessingOutputError(f"{paths['qc'].name}: QC flags {bad}")
    if ica_selection not in ("icasel", "ica_file"):
        raise ValueError(f"ica_selection must be 'icasel' or 'ica_file', got {ica_selection!r}")
    sel = read_qc(paths["icasel"]) if (ica_selection == "icasel" and paths["icasel"].is_file()) else None
    if require_complete and ica_selection == "icasel" and ica_status == "ok" and sel is None:
        raise PreprocessingOutputError(f"missing ICA re-selection output {paths['icasel'].name}")
    raw = read_raw_fif(paths["preica"])
    raw.load_data()
    raw._data = raw._data.astype(np.float64, copy=False)
    if verify:
        seg = read_segments(paths["segments"])
        want_n = int(qc["sample_axis"]["n_times"])
        if raw.n_times != want_n or raw.first_samp != int(qc["sample_axis"]["first_samp"]) or \
                seg.n_times != want_n or abs(raw.info["sfreq"] - float(qc["sample_axis"]["sfreq"])) > 1e-9:
            raise PreprocessingOutputError(
                f"{paths['preica'].name}: sample axis ({raw.n_times}, first_samp {raw.first_samp}) differs from the "
                f"QC record ({want_n}, {qc['sample_axis']['first_samp']})")
    ica = None
    exclude = None
    if apply_ica:
        if ica_status != "ok":
            raise PreprocessingOutputError(f"{paths['qc'].name}: apply_ica=True but the ICA status is "
                                           f"{ica_status!r} (use apply_ica=False for the no-ICA variant)")
        if not paths["ica"].is_file():
            raise PreprocessingOutputError(f"missing ICA file {paths['ica'].name}")
        ica = read_ica(paths["ica"])
        if sel is not None:
            exclude = list(sel["excluded"])
    raw = clean_from_parts(raw, ica, apply_ica=apply_ica, interpolate=interpolate, exclude=exclude,
                           origin=qc.get("interpolation_origin", "auto"))
    n_icafit = 0
    if sel is not None and annotate_icafit and (sel.get("icafit") or {}).get("mark"):
        n_icafit = _add_icafit_annotations(raw, sel["icafit"]["spans"], sel["icafit"]["description"])
    with raw.info._unlock():
        t = dict(raw.info["temp"])
        t.update({"ica_vetoed": list(sel["vetoed"]) if sel is not None else [],
                  "ica_selection_source": ("icasel" if sel is not None else "ica_file") if apply_ica else None,
                  "n_icafit_spans": n_icafit, "qc_flags": flags})
        raw.info["temp"] = t
    return raw
