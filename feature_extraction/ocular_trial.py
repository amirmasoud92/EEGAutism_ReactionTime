"""Trial-level ocular covariates: ``heog_z``, ``vfp_z`` (model covariates), ``closure_z``, ``gaze_dev_z``, blinks.

Per recording, :func:`recording_ocular_indices` calls :func:`preprocessing.ocular.compute_ocular_indices` on the
stored non-high-passed ocular copy with the trial-table onsets and the task's prestimulus window (or a window-length
variant); :func:`assemble_ocular_features` then z-scores within child x task over the **predictor set** and applies
the EXG fallbacks. The trial table is read with an explicit column list (:func:`load_predictor_trials`): keys,
stimulus timing and the flags named by the stimulus-defined subset of ``configs/events.yaml``; no response column is
read.

Definitions
-----------
``*_raw`` columns are the indices before z-scoring (``heog_logsd``, ``vfp_logsd``: log10 SD of the 15-Hz horizontal /
vertical EOG proxy in the alpha window after the window's linear trend; ``closure``: mean of the 5-Hz VEOG minus its
+-30-s running median over [-1.5, -0.05] s; ``gaze_dev``: mean |5-Hz HEOG - running median| over the same span).

z-scoring: ``(x - mean) / SD`` (ddof 1) with mean and SD taken over the rows of the child x task that are in the
predictor set and have a finite value; the same affine map is applied to every row with a finite value (rows outside
the predictor set keep a z on the predictor-set scale, flagged by ``in_predictor_set``). With fewer than
:data:`MIN_Z_ROWS` such rows, or SD 0, the z column is NaN for the child x task.

EXG fallbacks, driven by the stored EXG QC record (``<stem>_qc.json``: ``exg_qc[ch].pass``, ``exg_bads``,
known-issue bads):

- EXG1 or EXG2 fails: ``heog_z`` uses F7 - F8 (built into the ocular copy during preprocessing), and
  ``gaze_dev_z`` = 0 with ``gaze_dev_fallback`` = True; ``gaze_dev_fallback_child`` (child x task) is True if any
  recording of the child x task needed it. If F7/F8 are unusable as well, ``heog_z`` is NaN.
- EXG3 or EXG4 fails: ``vfp_z`` and ``closure_z`` use mean(T7, T8) as the reference (built into the copy).

The copy's recorded sources are checked against the QC record (:func:`check_sources_against_qc`); a mismatch raises.
When the sources differ between the runs of one child x task (one run with the fallback), the z-scoring is done
within child x task x source (``z_stratum``), because log SDs of F7 - F8 and EXG1 - EXG2 are on different scales;
with a single source this is exactly child x task.

Blink columns (``blink_in_window``, ``n_blinks_in_window``, ``time_since_last_blink_s``, ``tslb_censored``,
``valid_blink``) are passed through from preprocessing (blink extent overlapping the alpha window; time from the last
blink peak in the same piece to the onset).

Missing values: an index is NaN where its own window is invalid (``window_invalid_reason`` /
``closure_invalid_reason``, e.g. the 1.5-s closure window reaching a join that the alpha edge rule does not see).
The predictor set is **not** narrowed by a missing ocular index. :func:`missingness_summary` reports the pooled
fractions.

**Missing-covariate rule for the model covariates**: for the ocular covariates of the models (:data:`MISSING_RULE`:
``heog_z`` -> ``heog_missing``, ``vfp_z`` -> ``vfp_missing``), :func:`apply_missing_rule` sets the z value to 0 on
every row of the window's predictor set (with a valid alpha window) where it is missing, and marks that row in the
indicator column (True; False elsewhere, including rows outside the predictor set, whose z stays as it was). The rule
is applied per window (suffix ``__<window>`` for the window-length variants) by
:func:`feature_extraction.trial_features.ocular_z`. The model code includes an indicator only when it is True on
some model row; the raw ``heog_logsd`` / ``vfp_logsd`` stay NaN. ``closure_z`` / ``gaze_dev_z`` (descriptive
variant only; see :mod:`feature_extraction.ocular_index_validation`) are not covered by this rule.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trials.schema import apply_subset
from utils.config import EventsConfig, TaskEventsConfig
from utils.config_ocular import OcularConfig, as_ocular_config

__all__ = [
    "KEY_COLUMNS", "RAW_TO_Z", "Z_COLUMNS", "BLINK_COLUMNS", "FEATURE_COLUMNS", "MIN_Z_ROWS", "SAFE_TRIAL_COLUMNS",
    "OcularSourceMismatch", "ExgStatus", "window_spec", "load_predictor_trials", "exg_status_from_qc",
    "check_sources_against_qc", "find_recording_outputs", "recording_ocular_indices", "zscore_within",
    "assemble_ocular_features", "iter_recordings", "missingness_summary", "VALID_COLUMN_CANDIDATES",
    "apply_bad_annotations",
    "BAD_ANNOTATION_REASON", "MISSING_RULE", "apply_missing_rule",
]

KEY_COLUMNS: tuple[str, ...] = ("subject", "task", "run", "sample")
#: Trial-table columns read besides keys and subset flags: stimulus timing only (no response).
SAFE_TRIAL_COLUMNS: tuple[str, ...] = ("trial_key", "onset_s", "trial_index_in_run")
RAW_TO_Z: dict[str, str] = {"heog_logsd": "heog_z", "vfp_logsd": "vfp_z", "closure": "closure_z",
                            "gaze_dev": "gaze_dev_z"}
Z_COLUMNS: tuple[str, ...] = tuple(RAW_TO_Z.values())
BLINK_COLUMNS: tuple[str, ...] = ("blink_in_window", "n_blinks_in_window", "time_since_last_blink_s",
                                  "tslb_censored", "valid_blink")
MIN_Z_ROWS = 3
FEATURE_COLUMNS: tuple[str, ...] = (
    *KEY_COLUMNS, "trial_key", "window", "in_predictor_set", *Z_COLUMNS, "gaze_dev_fallback",
    "gaze_dev_fallback_child", *BLINK_COLUMNS, "heog_logsd", "vfp_logsd", "closure", "gaze_dev",
    "valid_window", "window_invalid_reason", "valid_closure", "closure_invalid_reason", "heog_source", "veog_ref",
    "z_stratum",
)
#: Missing-covariate rule: model covariate -> missingness indicator column.
MISSING_RULE: dict[str, str] = {"heog_z": "heog_missing", "vfp_z": "vfp_missing"}
HEOG_EXG = ("EXG1", "EXG2")
#: Validity columns accepted in ``extra_valid`` (in this order): ours, then the alpha table's.
VALID_COLUMN_CANDIDATES: tuple[str, ...] = ("window_valid", "valid")
REF_EXG = ("EXG3", "EXG4")


class OcularSourceMismatch(RuntimeError):
    """The stored ocular copy's sources disagree with the stored EXG QC record (stale derivative)."""


# --------------------------------------------------------------------------- trials (predictor only)
def window_spec(task_cfg: TaskEventsConfig, window: str = "primary", ocular_cfg: OcularConfig | None = None
                ) -> tuple[tuple[float, float], str]:
    """(window [start, stop) ms, predictor-set subset name) for ``window`` (``primary``, ``s_w500``, ...).

    The primary window must equal the ocular config's window for the task.
    """
    if window not in task_cfg.prestim_windows:
        raise KeyError(f"window {window!r} not in {task_cfg.name} prestim_windows_ms {sorted(task_cfg.prestim_windows)}")
    w = tuple(float(x) for x in task_cfg.prestim_windows[window])
    subset = "predictor_set" if window == "primary" else f"{window}_predictor_set"
    if subset not in task_cfg.analysis_subsets:
        raise KeyError(f"subset {subset!r} not configured for {task_cfg.name}")
    if window == "primary" and ocular_cfg is not None:
        ow = tuple(float(x) for x in ocular_cfg.indices.windows_ms.get(task_cfg.name, ()))
        if ow != w:
            raise ValueError(f"{task_cfg.name}: events.yaml primary window {w} != ocular.yaml window {ow}")
    return (w[0], w[1]), subset


def load_predictor_trials(task: str, events_cfg: EventsConfig, trials_path: str | os.PathLike[str], *,
                          subjects: Iterable[str] | None = None, window: str = "primary") -> pd.DataFrame:
    """Read the predictor-relevant columns of a task's trial table and flag the predictor set.

    Only :data:`KEY_COLUMNS`, :data:`SAFE_TRIAL_COLUMNS` and the flags of the window's stimulus-defined
    predictor-set subset are read.

    Returns
    -------
    pandas.DataFrame
        Keys, ``trial_key``, ``onset_s``, ``trial_index_in_run``, ``window`` and ``in_predictor_set`` (bool), sorted
        by subject, run, sample.
    """
    tc = events_cfg.task(task)
    _, subset = window_spec(tc, window)
    sub = tc.analysis_subsets[subset]
    flags = list(dict.fromkeys([*(sub.get("require_true") or []), *(sub.get("require_false") or [])]))
    cols = list(dict.fromkeys([*KEY_COLUMNS, *SAFE_TRIAL_COLUMNS, *flags]))
    filters = [("subject", "in", sorted(set(subjects)))] if subjects is not None else None
    df = pd.read_parquet(trials_path, columns=cols, filters=filters)
    df["in_predictor_set"] = apply_subset(df, sub).to_numpy(dtype=bool)
    df["window"] = window
    out = df[[*KEY_COLUMNS, *SAFE_TRIAL_COLUMNS, "window", "in_predictor_set"]]
    return out.sort_values(["subject", "run", "sample"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- EXG QC -> fallbacks
@dataclass(frozen=True)
class ExgStatus:
    """Which EXG fallbacks the stored EXG QC implies for one recording."""

    heog_exg_ok: bool          # EXG1 and EXG2 usable -> HEOG = EXG1 - EXG2, gaze_dev_z defined
    ref_exg_ok: bool           # EXG3 and EXG4 usable -> VEOG reference = mean(EXG3, EXG4)
    failed: tuple[str, ...]

    @property
    def gaze_dev_fallback(self) -> bool:
        return not self.heog_exg_ok


def exg_status_from_qc(qc: Mapping[str, Any]) -> ExgStatus:
    """Fallback decision from a stored QC record (``exg_qc``, ``exg_bads``, ``non_eeg_bads``, known bads).

    A channel is unusable if its EXG QC failed, it is absent from the QC (no EXG), or it is listed as bad.
    """
    ex = qc.get("exg_qc") or {}
    bads = set(qc.get("exg_bads") or []) | set(qc.get("non_eeg_bads") or []) | \
        set((qc.get("loading") or {}).get("known_bads") or [])

    def ok(ch: str) -> bool:
        r = ex.get(ch)
        return bool(r) and bool(r.get("present", True)) and bool(r.get("pass", False)) and ch not in bads

    failed = tuple(ch for ch in (*HEOG_EXG, *REF_EXG) if not ok(ch))
    return ExgStatus(all(ok(c) for c in HEOG_EXG), all(ok(c) for c in REF_EXG), failed)


def check_sources_against_qc(sources: Mapping[str, Any], status: ExgStatus) -> None:
    """Raise :class:`OcularSourceMismatch` if the ocular copy's sources contradict the QC-implied fallbacks."""
    heog = tuple(sources.get("heog") or ())
    ref = tuple(sources.get("ref") or ())
    if status.heog_exg_ok != (heog == HEOG_EXG):
        raise OcularSourceMismatch(f"HEOG source {heog or None} but EXG QC says EXG1/EXG2 usable={status.heog_exg_ok}")
    if ref and status.ref_exg_ok != (ref == REF_EXG):
        raise OcularSourceMismatch(f"VEOG reference {ref} but EXG QC says EXG3/EXG4 usable={status.ref_exg_ok}")


# --------------------------------------------------------------------------- stored outputs
def find_recording_outputs(preproc_root: str | os.PathLike[str], subject: str, task: str, run: int
                           ) -> dict[str, Path]:
    """Paths of the stored ocular copy and QC record of one recording (by the QC record's run number)."""
    from preprocessing.io import OUTPUT_FILES, read_qc

    d = Path(preproc_root) / task / subject
    for q in sorted(d.glob(f"{subject}_task-{task}*{OUTPUT_FILES['qc']}")):
        rec = read_qc(q).get("recording", {})
        if int(rec.get("run", -1)) == int(run):
            stem = q.name[: -len(OUTPUT_FILES["qc"])]
            return {"qc": q, "ocular": d / f"{stem}{OUTPUT_FILES['ocular']}"}
    raise FileNotFoundError(f"no preprocessing QC record for {subject} {task} run {run} under {d}")


# --------------------------------------------------------------------------- per recording
def recording_ocular_indices(ocular_raw: Any, trials: pd.DataFrame, window_ms: tuple[float, float], *,
                             qc: Mapping[str, Any] | None = None, cfg: OcularConfig | Mapping | None = None
                             ) -> pd.DataFrame:
    """Ocular indices for the trials of **one** recording, keyed like the trial table.

    Parameters
    ----------
    ocular_raw
        Stored ocular copy (``<stem>_desc-ocular_raw.fif``, read with MNE).
    trials
        Rows of one recording with :data:`KEY_COLUMNS` (``sample`` = BDF onset sample).
    window_ms
        Alpha window [start, stop) ms.
    qc
        Stored QC record; if given, the copy's sources are checked against its EXG QC and ``gaze_dev_fallback`` is
        taken from it, else from the copy's own sources.
    """
    from preprocessing.ocular import compute_ocular_indices, ocular_meta

    if trials[["subject", "task", "run"]].drop_duplicates().shape[0] > 1:
        raise ValueError("recording_ocular_indices takes the trials of one recording")
    c = as_ocular_config(cfg)
    meta = ocular_meta(ocular_raw)
    src = meta["sources"]
    if qc is not None:
        status = exg_status_from_qc(qc)
        check_sources_against_qc(src, status)
        fallback = status.gaze_dev_fallback
    else:
        fallback = tuple(src.get("heog") or ()) != HEOG_EXG
    ind = compute_ocular_indices(ocular_raw, trials["sample"].to_numpy(), c, window_ms=window_ms)
    if len(ind) != len(trials) or not np.array_equal(ind["onset_sample"].to_numpy(),
                                                     trials["sample"].to_numpy(dtype=np.int64)):
        raise RuntimeError("compute_ocular_indices returned rows out of order")
    out = trials.reset_index(drop=True).copy()
    for col in ind.columns:
        if col != "onset_sample":
            out[col] = ind[col].to_numpy()
    out["gaze_dev_fallback"] = bool(fallback)
    out.attrs["ocular"] = {k: v for k, v in ind.attrs.items() if k != "sources"}
    out.attrs["sources"] = src
    return out


# --------------------------------------------------------------------------- BAD* annotations
#: Value of ``window_invalid_reason`` / ``closure_invalid_reason`` for a window overlapping a ``BAD*`` annotation.
BAD_ANNOTATION_REASON = "bad_annotation"


def apply_bad_annotations(rows: pd.DataFrame, onsets: Sequence[int] | np.ndarray, window_ms: tuple[float, float],
                          closure_window_ms: Sequence[float], bad_spans: np.ndarray | None, sfreq: float = 512.0
                          ) -> tuple[pd.DataFrame, dict]:
    """Invalidate the ocular indices whose own window overlaps a ``BAD*`` annotation of the cleaned data.

    The ocular copy carries no ``BAD_icafit`` spans (they are annotations that ``load_clean`` adds to the cleaned
    EEG), so the window rule is applied here, consistently with the alpha validity: on the BDF sample clock
    (``onset + round(ms x sfreq / 1000)``, :func:`feature_extraction.spectral.window_bounds`),

    - alpha window overlap -> ``valid_window`` False (``window_invalid_reason`` = ``bad_annotation`` if it was
      valid), ``heog_logsd`` / ``vfp_logsd`` NaN; ``valid_blink`` False and the blink columns missing;
    - closure window ([-1.5, -0.05] s) overlap -> ``valid_closure`` False (``closure_invalid_reason`` =
      ``bad_annotation``), ``closure`` / ``gaze_dev`` NaN.

    Returns the updated copy and the counts of windows newly invalidated (``n_window``, ``n_closure``, ``n_blink``).
    """
    from feature_extraction.spectral import overlaps_spans, window_bounds

    out = rows.reset_index(drop=True).copy()
    counts = {"n_window": 0, "n_closure": 0, "n_blink": 0}
    if bad_spans is None or len(bad_spans) == 0 or len(out) == 0:
        return out, counts
    on = np.asarray(onsets, np.int64)
    if on.size != len(out):
        raise ValueError("onsets must align with rows")
    bad = np.asarray(bad_spans, np.int64).reshape(-1, 2)
    a, b = window_bounds(on, float(window_ms[0]), float(window_ms[1]), sfreq)
    hit_w = overlaps_spans(a, b, bad)
    c0, c1 = window_bounds(on, float(closure_window_ms[0]), float(closure_window_ms[1]), sfreq)
    hit_c = overlaps_spans(c0, c1, bad)

    def _was(col: str) -> np.ndarray:
        return out[col].astype("boolean").fillna(False).to_numpy(bool) if col in out.columns else np.zeros(len(out), bool)

    w = hit_w & _was("valid_window")
    counts["n_window"] = int(w.sum())
    if "valid_window" in out.columns:
        out.loc[hit_w, "valid_window"] = False
        if "window_invalid_reason" in out.columns:
            out["window_invalid_reason"] = out["window_invalid_reason"].astype(object)
            out.loc[w, "window_invalid_reason"] = BAD_ANNOTATION_REASON
    for col in ("heog_logsd", "vfp_logsd"):
        if col in out.columns:
            out.loc[hit_w, col] = np.nan
    c = hit_c & _was("valid_closure")
    counts["n_closure"] = int(c.sum())
    if "valid_closure" in out.columns:
        out.loc[hit_c, "valid_closure"] = False
        if "closure_invalid_reason" in out.columns:
            out["closure_invalid_reason"] = out["closure_invalid_reason"].astype(object)
            out.loc[c, "closure_invalid_reason"] = BAD_ANNOTATION_REASON
    for col in ("closure", "gaze_dev"):
        if col in out.columns:
            out.loc[hit_c, col] = np.nan
    bl = hit_w & _was("valid_blink")
    counts["n_blink"] = int(bl.sum())
    if "valid_blink" in out.columns:
        out.loc[hit_w, "valid_blink"] = False
        for col, dt in (("blink_in_window", "boolean"), ("tslb_censored", "boolean"), ("n_blinks_in_window", "Int64")):
            if col in out.columns:
                out[col] = out[col].astype(dt)          # numpy bool / int columns cannot hold a missing value
                out.loc[hit_w, col] = pd.NA
        if "time_since_last_blink_s" in out.columns:
            out.loc[hit_w, "time_since_last_blink_s"] = np.nan
    return out, counts


# --------------------------------------------------------------------------- z-scoring and fallbacks
def zscore_within(df: pd.DataFrame, col: str, groups: Sequence[str], mask: pd.Series | np.ndarray,
                  min_rows: int = MIN_Z_ROWS) -> pd.Series:
    """z of ``col`` within ``groups``, with mean / SD (ddof 1) from the rows in ``mask`` with a finite value."""
    x = pd.to_numeric(df[col], errors="coerce").astype(float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x.to_numpy())
    ref = x.where(m)
    g = df[list(groups)].astype(str).agg("|".join, axis=1) if len(groups) > 1 else df[groups[0]].astype(str)
    mu = ref.groupby(g).transform("mean")
    sd = ref.groupby(g).transform(lambda s: s.std(ddof=1))
    n = ref.groupby(g).transform("count")
    z = (x - mu) / sd
    bad = (n < min_rows) | ~(sd > 0)
    return z.where(~bad)


def assemble_ocular_features(rows: pd.DataFrame, *, predictor_mask: str = "in_predictor_set",
                             extra_valid: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """z-score the raw indices within child x task (x source) over the predictor set; apply the EXG fallbacks.

    Parameters
    ----------
    rows
        Concatenated output of :func:`recording_ocular_indices` for the recordings of one or more children of one
        task.
    predictor_mask
        Boolean column marking the predictor set from the trial table (lookback and edge rules).
    extra_valid
        Optional boolean mask aligned with ``rows`` for the remaining window rules that need the EEG (ROI
        peak-to-peak, >= 5 ROI channels), from the alpha table; ANDed into the predictor set.
    """
    df = rows.reset_index(drop=True).copy()
    ps = df[predictor_mask].fillna(False).astype(bool)
    if extra_valid is not None:
        ev = pd.Series(np.asarray(extra_valid, dtype=bool))
        if len(ev) != len(df):
            raise ValueError("extra_valid must align with rows")
        ps &= ev.to_numpy()
    df["in_predictor_set"] = ps.to_numpy()
    base = ["subject", "task"]
    df["z_stratum"] = df["heog_source"].astype(str) + "/" + df["veog_ref"].astype(str)
    multi = df.groupby(base)["z_stratum"].transform("nunique") > 1
    df.loc[~multi, "z_stratum"] = "all"
    for raw, z in RAW_TO_Z.items():
        if raw in ("heog_logsd", "gaze_dev"):
            strat = df["heog_source"].astype(str).where(multi, "all")
        else:
            strat = df["veog_ref"].astype(str).where(multi, "all")
        tmp = df.assign(_s=strat)
        df[z] = zscore_within(tmp, raw, [*base, "_s"], df["in_predictor_set"])
    fb = df["gaze_dev_fallback"].fillna(True).astype(bool)
    df.loc[fb, "gaze_dev_z"] = 0.0
    df["gaze_dev_fallback"] = fb
    df["gaze_dev_fallback_child"] = fb.groupby([df["subject"], df["task"]]).transform("any").astype(bool)
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[cols]


def apply_missing_rule(z: pd.DataFrame, mask: pd.Series | np.ndarray, *, suffix: str = "",
                       rule: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Missing-covariate rule (module docstring): 0 plus a missingness indicator.

    Parameters
    ----------
    z
        Table with the z columns ``<col><suffix>`` of :data:`MISSING_RULE` (other columns are passed through).
    mask
        Boolean, aligned with ``z``: the window's predictor set with a valid alpha window (the model rows).
    suffix
        Window suffix (``""`` primary, ``"__s_w500"`` etc.); the indicator gets the same suffix.

    Returns
    -------
    A copy of ``z`` with each covered z column set to 0 where ``mask`` and missing, and the indicator column
    ``<indicator><suffix>`` (bool; True exactly there).
    """
    rule = dict(MISSING_RULE if rule is None else rule)
    m = np.asarray(mask, dtype=bool)
    if len(m) != len(z):
        raise ValueError("mask must align with z")
    out = z.copy()
    for col, ind in rule.items():
        c = f"{col}{suffix}"
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce").to_numpy(float)
        miss = m & ~np.isfinite(x)
        out[c] = np.where(miss, 0.0, x)
        out[f"{ind}{suffix}"] = miss
    return out


# --------------------------------------------------------------------------- recordings
def _extra_valid_mask(rows: pd.DataFrame, extra_valid: pd.DataFrame | None, valid_col: str | None
                      ) -> np.ndarray | None:
    """Align a keys + validity table (e.g. the alpha table's ``valid``) with ``rows``; missing rows are invalid."""
    if extra_valid is None:
        return None
    col = valid_col or next((c for c in VALID_COLUMN_CANDIDATES if c in extra_valid.columns), None)
    if col is None or col not in extra_valid.columns:
        raise KeyError(f"extra_valid needs one of {VALID_COLUMN_CANDIDATES} (or valid_col)")
    m = rows[list(KEY_COLUMNS)].merge(extra_valid[[*KEY_COLUMNS, col]], on=list(KEY_COLUMNS), how="left",
                                      validate="one_to_one")
    return m[col].astype("boolean").fillna(False).astype(bool).to_numpy()


def iter_recordings(trials: pd.DataFrame, task: str, preproc_root: str | os.PathLike[str], *, strict: bool = True,
                    counter: dict | None = None):
    """Yield ``(subject, run, trial rows, ocular copy, QC record)`` for each recording of ``trials``.

    Recordings without a stored QC record or ocular copy raise (``strict``) or are skipped and counted in
    ``counter['n_skipped']``. Each ocular copy is read once (MNE, preloaded).
    """
    import mne

    if counter is not None:
        counter.setdefault("n_skipped", 0)
    for (sub, run), g in trials.groupby(["subject", "run"], sort=True):
        try:
            paths = find_recording_outputs(preproc_root, str(sub), task, int(run))
            if not paths["ocular"].is_file():
                raise FileNotFoundError(paths["ocular"].name)
        except FileNotFoundError:
            if strict:
                raise
            if counter is not None:
                counter["n_skipped"] += 1
            continue
        qc = json.loads(Path(paths["qc"]).read_text(encoding="utf-8"))
        oc = mne.io.read_raw_fif(paths["ocular"], preload=True, verbose="error")
        yield str(sub), int(run), g, oc, qc


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype("boolean").fillna(False).astype(bool)


def missingness_summary(feats: pd.DataFrame) -> dict:
    """Pooled fractions over the predictor-set rows (counts only; no IDs)."""
    ps = _as_bool(feats["in_predictor_set"])
    n = int(ps.sum())
    out: dict[str, Any] = {"n_predictor_set": n}
    if not n:
        return out
    for z in Z_COLUMNS:
        out[f"frac_nan_{z}"] = float(feats.loc[ps, z].isna().mean())
    out["frac_gaze_dev_fallback"] = float(_as_bool(feats.loc[ps, "gaze_dev_fallback"]).mean())
    vb = _as_bool(feats["valid_blink"])
    out["frac_valid_blink"] = float(vb[ps].mean())
    bw = feats.loc[ps & vb, "blink_in_window"]
    out["frac_blink_in_window_of_valid"] = float(_as_bool(bw).mean()) if len(bw) else None
    return out


