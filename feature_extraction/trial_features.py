"""Integration of the trial features: IAF -> per-recording table -> per-task decomposition.

1. **IAF** (:func:`compute_child_iaf`). One row per child: restingIAF on the clean rest chunks of all recorded days
   (used for the primary alpha) and of the main day only (columns ``mainday_*``, sensitivity analysis), with the
   fallback chain of :mod:`feature_extraction.iaf` (restingIAF peak, restingIAF centre of gravity, specparam alpha
   peak of the rest spectrum, specparam alpha peak of the mean AVSRT prestimulus spectrum, 10 Hz). Children without
   rest data still get an IAF (fallbacks 4/5). The main-day flags of the rest runs come from the session table
   (:mod:`utils.session`, ``is_main_day``); the AVSRT onsets from :func:`feature_extraction.iaf.avsrt_predictor_onsets`.
2. **Recording** (:func:`compute_recording_features`). For one recording: the child's IAF (:func:`iaf_lookup`), the
   alpha table with every sensitivity variant (:func:`feature_extraction.prestimulus_alpha.compute_alpha_features`),
   the raw trial ocular indices for every prestimulus window
   (:func:`feature_extraction.ocular_trial.recording_ocular_indices`), burst occupancy
   (:mod:`feature_extraction.bursts`) on the same rows, and the stimulus / press counts of the context interval
   (:mod:`trials.context_counts`). The trial rows are read with
   :func:`feature_extraction.prestimulus_alpha.load_predictor_rows`.
3. **Task** (:func:`decompose_task_table`). The per-recording tables of a task (cohorts separately; dev: unaffected
   siblings, used to develop the processing; conf: autistic and typically developing children) are concatenated and
   decomposed within child x task (:func:`feature_extraction.decomposition.build_decomposed`: ``sd_i``, merged
   blocks, ``a_wb``, ``a_slow``, ``a_wb_raw``, running-mean variant, ``a_hist5``, ``a_ctx10/20`` for the primary and
   every sensitivity variant), and the ocular indices are z-scored within child x task over the predictor set with the
   EXG fallbacks and the missing-covariate rule (:func:`ocular_z`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TASKS: tuple[str, ...] = ("AVSRT", "FAST", "IC")
KEYS: tuple[str, ...] = ("subject", "task", "run", "sample")
#: Ocular columns of the primary window kept in the per-recording table (raw indices; z-scored per task).
OCULAR_PRIMARY_COLUMNS: tuple[str, ...] = (
    "heog_logsd", "vfp_logsd", "closure", "gaze_dev", "blink_in_window", "n_blinks_in_window",
    "time_since_last_blink_s", "tslb_censored", "valid_blink", "valid_window", "window_invalid_reason",
    "valid_closure", "closure_invalid_reason", "heog_source", "veog_ref", "gaze_dev_fallback")
#: Window-dependent ocular columns kept for the window-length variants (suffix ``__<window>``).
OCULAR_WINDOW_COLUMNS: tuple[str, ...] = ("heog_logsd", "vfp_logsd", "blink_in_window", "n_blinks_in_window",
                                          "valid_blink", "valid_window", "window_invalid_reason")
#: Alpha window end (ms): every prestimulus window ends 50 ms before onset; the burst window W is its length.
ALPHA_WINDOW_END_MS = -50.0


class FeaturePipelineError(RuntimeError):
    """A recording or task cannot be processed (missing inputs, inconsistent configs)."""


# =========================================================================== settings
@dataclass
class FeatureSettings:
    """The configs the feature code reads."""

    events: Any
    features: Any
    preprocessing: Any
    ocular: Any
    bursts: Any
    iaf: Any
    preproc_root: Path
    sfreq: float = 512.0
    #: ``load_clean`` with ``require_complete=True``, ``ica_selection="icasel"`` and ``annotate_icafit=True``, i.e.
    #: the ICA exclusions (ocular components with the posterior-alpha veto) and the ``BAD_icafit`` spans of the
    #: ICA selection output, in the ICA-corrected and the no-ICA variant.
    ica_selection: str = "icasel"


#: The ICA selection the feature code uses.
ICA_SELECTION = "icasel"


def load_settings(root: str | os.PathLike[str] | None = None, *,
                  derivatives: str | os.PathLike[str] = "derivatives", sfreq: float = 512.0) -> FeatureSettings:
    """Load and cross-check the configs (``features.yaml`` vs events / preprocessing, ``bursts.yaml`` vs
    preprocessing, the ocular window vs events).

    ``root``: folder holding ``configs/`` (default: this repository); ``derivatives``: folder of the derivatives (the
    preprocessing outputs are under ``<derivatives>/<storage.subdir of preprocessing.yaml>``); ``sfreq``: sampling
    rate of the recordings (512 Hz).
    """
    from feature_extraction import bursts as B
    from feature_extraction import iaf as I
    from preprocessing import io as pio
    from preprocessing.config import load_preprocessing_config
    from utils.config import load_events_config
    from utils.config_features import check_consistency, load_features_config
    from utils.config_ocular import load_ocular_config

    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    ev = load_events_config(root=root)
    fc = load_features_config(root=root)
    pcfg = load_preprocessing_config(root=root)
    check_consistency(fc, ev, pcfg)
    oc = load_ocular_config(root=root)
    bc = B.load_burst_config(root=root)
    if tuple(pcfg.artefacts.roi) != tuple(bc.roi) or float(pcfg.artefacts.roi_ptp_uv) != float(bc.artefact_ptp_uv):
        raise FeaturePipelineError("configs/bursts.yaml roi / artefact_ptp_uv differ from configs/preprocessing.yaml")
    if tuple(pcfg.artefacts.roi) != I.ROI:
        raise FeaturePipelineError("configs/preprocessing.yaml artefacts.roi differs from the posterior ROI")
    for task in TASKS:
        tc = ev.task(task)
        w = tuple(float(x) for x in tc.prestim_windows["primary"])
        ow = tuple(float(x) for x in oc.indices.windows_ms.get(task, ()))
        if ow != w:
            raise FeaturePipelineError(f"{task}: events.yaml primary window {w} != ocular.yaml window {ow}")
        if abs((w[1] - w[0]) - bc.window_ms(task)) > 1e-9:
            raise FeaturePipelineError(f"{task}: events.yaml primary W {w[1] - w[0]} != bursts.yaml "
                                       f"{bc.window_ms(task)}")
    return FeatureSettings(events=ev, features=fc, preprocessing=pcfg, ocular=oc, bursts=bc, iaf=I.IAFSettings(),
                           preproc_root=pio.preproc_root(derivatives, pcfg.storage_subdir), sfreq=float(sfreq),
                           ica_selection=ICA_SELECTION)


# =========================================================================== cleaned data
#: ``load_clean`` keyword arguments of every feature read: complete outputs only, the ICA selection output
#: (exclusions with the posterior-alpha veto) and its ``BAD_icafit`` spans added as annotations in BOTH variants, so
#: every window rule that honours ``BAD*`` annotations (task windows, rest chunks, AVSRT fallback windows) leaves
#: them out.
LOAD_CLEAN_KW: dict[str, Any] = {"interpolate": False, "require_complete": True, "ica_selection": ICA_SELECTION,
                                 "annotate_icafit": True}


def load_clean_checked(rec, *, apply_ica: bool, root: Path):
    """``load_clean(**LOAD_CLEAN_KW)``, checked: with ``apply_ica`` the exclusions must come from the re-selection
    output, and in both variants ``load_clean`` must report the ``BAD_icafit`` annotation count."""
    from preprocessing.clean import load_clean

    raw = load_clean(rec, apply_ica=apply_ica, root=root, **LOAD_CLEAN_KW)
    tmp = raw.info.get("temp")
    tmp = tmp if isinstance(tmp, dict) else {}
    src = tmp.get("ica_selection_source")
    if apply_ica and src != ICA_SELECTION:
        raise FeaturePipelineError(f"{rec.stem}: ICA selection source {src!r}, expected the re-selection output")
    if "n_icafit_spans" not in tmp:
        raise FeaturePipelineError(f"{rec.stem}: load_clean did not report the BAD_icafit annotations")
    return raw


# =========================================================================== IAF


def compute_child_iaf(st: FeatureSettings, subject: str, rest: Sequence, main_day: Mapping[int, bool],
                      avsrt: Sequence, onsets: Mapping[tuple[str, int], np.ndarray]) -> tuple[dict, list[dict]]:
    """IAF row (all days, and ``mainday_*`` for the main-day variant) and per-run chunk accounting of one child.

    ``rest``: the child's Restingstate recordings; ``main_day``: run -> recorded on the main session day;
    ``avsrt``: the child's AVSRT recordings and ``onsets`` their predictor-set onsets (fallback 4).
    """
    from feature_extraction import iaf as I

    s = st.iaf
    runs, run_rows = [], []
    for rec in rest:
        md = {"run": int(rec.run), "is_main_day": bool(main_day.get(int(rec.run), False))}
        try:
            r = _load_rest_run(st, rec, md)
        except Exception as e:  # noqa: BLE001 - a run that cannot be loaded must not silently trigger a fallback
            raise FeaturePipelineError(f"rest run {int(rec.run)} could not be loaded: {e!r}"[:400]) from e
        runs.append(r)
        run_rows.append({"subject": subject, "run": int(rec.run), "status": r.meta.get("status"), **md,
                         "n_bad_roi": r.meta.get("n_bad_roi"), "n_chunks_grid": r.n_grid,
                         "n_chunks_edge_valid": r.n_edge_valid, "n_chunks_bad_annotation": r.n_bad_annotation,
                         "n_chunks_clean": r.n_kept})
    cache: dict = {}

    def sp_avsrt():
        if "r" not in cache:
            if not avsrt:
                cache["r"] = {"cf": np.nan, "pw": np.nan, "n_windows": 0, "error": "no_avsrt_recording"}
            else:
                # a load failure raises (no silent step-5 fallback); too few windows is a legitimate fallback
                cache["r"] = _avsrt_specparam(st, avsrt, onsets)
        return cache["r"]

    sf = float(st.sfreq)
    allv = I.estimate_from_runs(runs, s, sfreq=sf, sp_avsrt=sp_avsrt)
    mainv = I.estimate_from_runs([r for r in runs if r.meta.get("is_main_day")], s, sfreq=sf, sp_avsrt=sp_avsrt)
    row: dict[str, Any] = {"subject": subject, "n_rest_recordings": len(rest), "n_avsrt_recordings": len(avsrt)}
    row.update(allv)
    row.update({f"mainday_{k}": v for k, v in mainv.items()})
    return row, run_rows


def _checked_loader(st: FeatureSettings):
    def lc(recording, apply_ica=True, interpolate=True, root=None, **kw):
        if interpolate:
            raise FeaturePipelineError("the feature code never interpolates bad channels")
        return load_clean_checked(recording, apply_ica=apply_ica, root=Path(root or st.preproc_root))

    return lc


def _load_rest_run(st: FeatureSettings, rec, md: Mapping):
    """:func:`feature_extraction.iaf.load_rest_run` with the checked ``load_clean`` (ICA selection mode)."""
    from feature_extraction import iaf as I

    return I.load_rest_run(rec, st.iaf.chunks, root=st.preproc_root, meta=md, load_clean_fn=_checked_loader(st))


def _avsrt_specparam(st: FeatureSettings, avsrt: Sequence, onsets: Mapping) -> dict:
    from feature_extraction import iaf as I

    return I.avsrt_specparam_for_child(avsrt, onsets, st.iaf, root=st.preproc_root,
                                       load_clean_fn=_checked_loader(st))


def iaf_lookup(iaf_tab: pd.DataFrame, subject: str) -> dict:
    """``{"iaf", "iaf_source", "iaf_s_iaf", "iaf_source_s_iaf"}`` of a child (raises if the child has no row)."""
    r = iaf_tab.loc[iaf_tab["subject"] == subject]
    if len(r) != 1:
        raise FeaturePipelineError(f"no IAF row for {subject}")
    r = r.iloc[0]
    iaf = float(r["iaf"])
    if not np.isfinite(iaf):
        raise FeaturePipelineError(f"IAF of {subject} is not finite")
    md = float(r["mainday_iaf"]) if "mainday_iaf" in r and pd.notna(r["mainday_iaf"]) else np.nan
    return {"iaf": iaf, "iaf_source": str(r["iaf_source"]),
            "iaf_s_iaf": md if np.isfinite(md) else None,
            "iaf_source_s_iaf": str(r.get("mainday_iaf_source")) if np.isfinite(md) else None}


# =========================================================================== one recording
def burst_windows(st: FeatureSettings, task: str) -> dict[str, float]:
    """Burst window length W of every prestimulus window (``[-W-50, -250)`` ms)."""
    w = {}
    for name, (a, b) in st.events.task(task).prestim_windows.items():
        if abs(float(b) - ALPHA_WINDOW_END_MS) > 1e-9:
            raise FeaturePipelineError(f"{task} window {name} does not end at {ALPHA_WINDOW_END_MS} ms")
        w[name] = float(b) - float(a)
    return w


def burst_column_names(st: FeatureSettings, windows: Mapping[str, float]) -> list[str]:
    from feature_extraction import bursts as B

    cols = []
    for wn in windows:
        sfx = "" if wn == "primary" else f"__{wn}"
        cols += [f"burst_edge_ok{sfx}", f"burst_valid{sfx}"]
        cols += [f"{B.occ_column(p, d, v)}{sfx}" for v in st.bursts.variants for p, d in st.bursts.grid]
    return cols + ["burst_n_roi"]


def burst_features(st: FeatureSettings, raw, seg, iaf: float, onsets: np.ndarray, task: str
                   ) -> tuple[dict[str, np.ndarray], dict]:
    """Burst columns for ``onsets`` and run-level diagnostics; NaN columns if detection is impossible.

    The ``BAD*`` annotations of the cleaned ``raw`` (``BAD_icafit``, ``BAD_short``, ...) are left out of the
    background spectrum and make every burst window whose guarded extent overlaps them invalid
    (:func:`feature_extraction.bursts.trial_burst_features`)."""
    from feature_extraction import bursts as B
    from feature_extraction import spectral as sp

    cfg = st.bursts
    windows = burst_windows(st, task)
    fs = float(raw.info["sfreq"])
    bads = set(raw.info["bads"])
    good_all = [c for c in cfg.roi if c in raw.ch_names and c not in bads]
    n = len(onsets)
    meta: dict[str, Any] = {"n_roi_good": len(good_all), "windows_W_ms": windows}
    if len(good_all) < cfg.min_roi_channels:
        cols = {c: (np.zeros(n, bool) if c.startswith(("burst_edge_ok", "burst_valid")) else np.full(n, np.nan))
                for c in burst_column_names(st, windows)}
        cols["burst_n_roi"] = np.full(n, len(good_all), np.int64)
        meta["status"] = "too_few_roi_channels"
        return cols, meta
    x = raw.get_data(picks=good_all)
    sig, good = B.roi_signal(x, good_all, (), cfg.roi)
    spans = np.asarray(seg.valid_spans, np.int64).reshape(-1, 2)
    bad = sp.bad_annotation_spans(raw, st.features.alpha.bad_annotation_prefix)
    bg_mask = B.exclude_spans(B.artefact_chunk_mask(x, spans, fs, cfg.artefact_ptp_uv, cfg.artefact_chunk_s), bad)
    spectra = B.run_spectra(sig, spans, fs, cfg, bg_mask)
    bgs = B.run_backgrounds(sig, spans, fs, cfg, spectra=spectra)
    rb = B.detect_run_bursts(sig, fs, spans, iaf, cfg, channels=good, bg_mask=bg_mask, backgrounds=bgs)
    on64 = np.asarray(onsets, np.int64)
    cols = B.trial_burst_features(rb, on64, windows, bad_spans=bad)
    guard = rb.guard_samples()
    bad_counts = {}
    for wn, wms in windows.items():
        a, b = B.burst_window_samples(on64, wms, fs, cfg.window_stop_ms, cfg.window_start_offset_ms)
        edge_ok = B.window_inside_spans(a, b, spans, guard)
        bad_counts[wn] = {"n_bad_guarded": int((edge_ok & B.window_overlaps_spans(a, b, bad, guard)).sum()),
                          "n_bad_window_only": int((edge_ok & B.window_overlaps_spans(a, b, bad, 0)).sum())}
    meta.update(n_bad_annotation_spans=int(len(bad)), bad_annotation_windows=bad_counts)
    meta.update(status="ok", bg_sample_fraction=rb.bg_sample_fraction,
                valid_sample_fraction=rb.valid_sample_fraction,
                backgrounds={v: rb.backgrounds[v].summary() for v in cfg.variants},
                above_fraction={f"{v}:p{int(round(p * 100))}": val for (v, p), val in rb.above_fraction.items()},
                band_snr_median={v: float(np.nanmedian(s)) for v, s in rb.band_snr.items()})
    return cols, meta


def ocular_features(st: FeatureSettings, paths: Mapping[str, Path], qc: Mapping, keys: pd.DataFrame, task: str,
                    *, bad_spans: np.ndarray | None = None, sfreq: float = 512.0) -> tuple[pd.DataFrame, dict]:
    """Raw trial ocular indices of the primary window (all columns) and the window-length variants
    (window-dependent columns, suffix ``__<window>``).

    ``bad_spans`` (the ``BAD*`` annotations of the cleaned data, BDF sample clock at ``sfreq``) invalidate every index
    whose own window overlaps them (:func:`feature_extraction.ocular_trial.apply_bad_annotations`)."""
    import mne

    from feature_extraction import ocular_trial as ot

    if not Path(paths["ocular"]).is_file():
        raise FeaturePipelineError(f"missing ocular copy {Path(paths['ocular']).name}")
    oc = mne.io.read_raw_fif(paths["ocular"], preload=True, verbose="error")
    tc = st.events.task(task)
    out = pd.DataFrame(index=range(len(keys)))
    meta: dict[str, Any] = {}
    for wn, (w0, w1) in tc.prestim_windows.items():
        o = ot.recording_ocular_indices(oc, keys.reset_index(drop=True), (float(w0), float(w1)), qc=qc, cfg=st.ocular)
        o, nbad = ot.apply_bad_annotations(o, keys["sample"].to_numpy(np.int64), (float(w0), float(w1)),
                                           st.ocular.indices.closure_window_ms, bad_spans, sfreq)
        meta.setdefault("bad_annotation_windows", {})[wn] = nbad
        if wn == "primary":
            for c in OCULAR_PRIMARY_COLUMNS:
                out[c] = o[c].to_numpy() if c in o.columns else np.nan
            meta["attrs"] = {k: v for k, v in o.attrs.get("ocular", {}).items()}
            meta["sources"] = o.attrs.get("sources")
        else:
            for c in OCULAR_WINDOW_COLUMNS:
                out[f"{c}__{wn}"] = o[c].to_numpy() if c in o.columns else np.nan
    for c in out.columns:
        if c.startswith(("window_invalid_reason", "closure_invalid_reason", "heog_source", "veog_ref")):
            out[c] = pd.array([None if pd.isna(v) else str(v) for v in out[c]], dtype="string")
    return out, meta


def compute_recording_features(st: FeatureSettings, rec, iaf: Mapping, rows: pd.DataFrame, paths: Mapping[str, Path],
                               qc: Mapping) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """(feature table, mean spectra, metadata, context counts) of one recording.

    ``rows``: :func:`feature_extraction.prestimulus_alpha.load_predictor_rows`; ``paths``: the recording's
    preprocessing outputs (:func:`preprocessing.io.recording_outputs`); ``qc``: its stored QC record.
    """
    from feature_extraction import prestimulus_alpha as af
    from feature_extraction import spectral as sp_mod
    from preprocessing import io as pio
    from trials import context_counts as cc

    tc = st.events.task(rec.task)
    seg = pio.read_segments(paths["segments"])
    keep = st.features.alpha.all_channels()
    raw = load_clean_checked(rec, apply_ica=True, root=st.preproc_root)
    tmp = raw.info.get("temp")
    ica_applied = bool(tmp.get("ica_applied", False)) if isinstance(tmp, dict) else None
    raw.pick([c for c in raw.ch_names if c in keep])
    raw_n = load_clean_checked(rec, apply_ica=False, root=st.preproc_root)
    raw_n.pick([c for c in raw_n.ch_names if c in keep])
    stim = af.recording_stimulus_samples(rec.events_path, tc) if rec.events_path is not None else None
    res = af.compute_alpha_features(raw, rows, tc, iaf["iaf"], seg=seg, fcfg=st.features, raw_noica=raw_n,
                                    stim_samples=stim, iaf_hz_s_iaf=iaf["iaf_s_iaf"])
    del raw_n
    table = res.table
    table.insert(table.columns.get_loc("iaf_hz") + 1, "iaf_source", iaf["iaf_source"])
    onsets = table["sample"].to_numpy(np.int64)
    try:
        bcols, bmeta = burst_features(st, raw, seg, iaf["iaf"], onsets, rec.task)
    except Exception as e:  # noqa: BLE001 - burst detection failure leaves NaN burst columns (recorded)
        windows = burst_windows(st, rec.task)
        bcols = {c: (np.zeros(len(onsets), bool) if c.startswith(("burst_edge_ok", "burst_valid"))
                     else np.full(len(onsets), np.nan)) for c in burst_column_names(st, windows)}
        bmeta = {"status": f"error: {e!r}"[:300]}
    sfreq = float(raw.info["sfreq"])
    bad_sp = sp_mod.bad_annotation_spans(raw, st.features.alpha.bad_annotation_prefix)
    del raw
    ocu, ometa = ocular_features(st, paths, qc, table[list(KEYS)], rec.task, bad_spans=bad_sp, sfreq=sfreq)
    extra = pd.concat([pd.DataFrame(bcols, index=table.index), ocu.set_axis(table.index)], axis=1)
    clash = sorted(set(extra.columns) & set(table.columns))
    if clash:
        raise FeaturePipelineError(f"burst / ocular columns clash with alpha columns: {clash}")
    table = pd.concat([table, extra], axis=1).copy()
    counts = cc.recording_context_counts(table[list(KEYS)], rec.events_path, tc, seg, sfreq,
                                         st.features.alpha.context_lengths_s)
    meta = dict(res.meta)
    sel_src = tmp.get("ica_selection_source") if isinstance(tmp, dict) else None
    meta.update({"ica_applied_primary": ica_applied, "ica_selection": st.ica_selection,
                 "ica_selection_source": sel_src, "iaf": dict(iaf), "bursts": bmeta, "ocular": ometa})
    return table, res.spectra, meta, counts


# =========================================================================== one task
def ocular_z(tab: pd.DataFrame, task: str) -> pd.DataFrame:
    """z-scored ocular covariates for the primary window and every window-length variant (suffix
    ``__<window>``), within child x task (x source when the source differs between runs) over the predictor set of
    the window AND its alpha validity (window rules that need the EEG). The missing-covariate rule
    (:func:`feature_extraction.ocular_trial.apply_missing_rule`) then sets a missing ``heog_z`` / ``vfp_z`` to 0 on
    those rows and adds ``heog_missing`` / ``vfp_missing`` (same suffix)."""
    from feature_extraction import ocular_trial as ot

    out = pd.DataFrame(index=tab.index)
    wins = ["primary"] + sorted({c.split("__", 1)[1] for c in tab.columns if c.startswith("heog_logsd__")})
    for wn in wins:
        sfx = "" if wn == "primary" else f"__{wn}"
        ps = "in_predictor_set" if wn == "primary" else f"in_{wn}_predictor_set"
        valid = "valid" if wn == "primary" else f"valid_{wn}"
        if ps not in tab.columns or valid not in tab.columns:
            continue
        rows = pd.DataFrame({"subject": tab["subject"].to_numpy(), "task": tab["task"].to_numpy(),
                             "run": tab["run"].to_numpy(), "sample": tab["sample"].to_numpy(),
                             "in_predictor_set": tab[ps].astype("boolean").fillna(False).to_numpy(bool),
                             "heog_logsd": tab[f"heog_logsd{sfx}"].to_numpy(float),
                             "vfp_logsd": tab[f"vfp_logsd{sfx}"].to_numpy(float),
                             "closure": tab["closure"].to_numpy(float), "gaze_dev": tab["gaze_dev"].to_numpy(float),
                             "heog_source": tab["heog_source"].astype(str).to_numpy(),
                             "veog_ref": tab["veog_ref"].astype(str).to_numpy(),
                             "gaze_dev_fallback": tab["gaze_dev_fallback"].astype("boolean").to_numpy()})
        z = ot.assemble_ocular_features(rows, extra_valid=tab[valid].astype("boolean").fillna(False).to_numpy(bool))
        # missing-covariate rule for the model covariates heog_z / vfp_z: 0 + indicator on model rows
        z = ot.apply_missing_rule(z, z["in_predictor_set"].to_numpy(bool))
        for c in ot.Z_COLUMNS:
            out[f"{c}{sfx}"] = z[c].to_numpy()
        for ind in ot.MISSING_RULE.values():
            out[f"{ind}{sfx}"] = z[ind].to_numpy(bool)
        if wn == "primary":
            out["gaze_dev_fallback_child"] = z["gaze_dev_fallback_child"].to_numpy(bool)
            out["ocular_z_stratum"] = pd.array(z["z_stratum"].astype(str).to_numpy(), dtype="string")
    return out


def decompose_task_table(st: FeatureSettings, tab: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    """Decomposition (all variants) + ocular z-scores for the concatenated per-recording tables of one task."""
    from feature_extraction import decomposition as dc

    tab = tab.sort_values(["subject", "run", "sample"], kind="stable").reset_index(drop=True)
    dec, dlog = dc.build_decomposed(tab, task, dcfg=st.features.decomposition)
    z = ocular_z(dec, task)
    for c in z.columns:
        dec[c] = z[c].to_numpy()
    dlog["merge_mode"] = st.features.decomposition.merge_mode(task)
    return dec, dlog
