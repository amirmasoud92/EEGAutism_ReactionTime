"""Saccade check of the FAST group difference (Additional file 1, Supplementary Methods S4, R10; Table S7).

Extraction (EEG side only): for each FAST recording, the horizontal-EOG saccade rule of the ocular quality control
(:func:`feature_extraction.ocular_index_validation.recording_events`) counts detected saccades in two windows around
every target onset with a valid alpha window: (a) [-550, -50) ms (the alpha window) and (b) [-550, +300] ms.

Models: M3-H2 refitted with the final M3 random-effects structure on targets with valid saccade detection and no
saccade in window (a), and separately (b); plus the per-child rate of flagged trials by group.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import sensitivity as SENS
from statistical_analysis.follow_up_analyses import H2T, fit_targets, welch
from statistical_analysis.group_difference_checks import Run

log = logging.getLogger(__name__)

WINDOWS = {"a": (-550.0, -50.0), "b": (-550.0, 300.0)}


def extract_recording(job: tuple) -> pd.DataFrame:
    subject, run, samples = job
    from feature_extraction.ocular_index_validation import recording_events
    from feature_extraction.ocular_trial import exg_status_from_qc
    from feature_extraction.trial_features import load_settings
    from preprocessing import io as pio

    st = load_settings()
    stem = f"{subject}_task-FAST_run-{int(run):02d}"
    outs = pio.recording_outputs(st.preproc_root, ("FAST", subject, stem))
    qc = pio.read_qc(outs["qc"]) if outs["qc"].is_file() else None
    oc = pio.read_raw_fif(outs["ocular"])
    ok = exg_status_from_qc(qc).heog_exg_ok if qc else False
    df = pd.DataFrame({"subject": subject, "run": run, "sample": np.asarray(samples, np.int64)})
    for key, win in WINDOWS.items():
        ev, info = recording_events(oc, df["sample"].to_numpy(), win, st.ocular, detect_saccades_here=ok)
        df[f"n_sacc_{key}"] = ev["n_saccades_alpha"].to_numpy()
        df[f"sacc_valid_{key}"] = ev["det_valid_saccade_alpha"].to_numpy().astype(bool)
    df["saccades_detectable"] = bool(info["saccades_detectable"])
    return df


def extract(predictor_table, out_path, n_jobs: int = 10) -> Path:
    """Saccade counts in the two windows for all FAST targets with a valid prestimulus window; trial-level parquet."""
    out_path = Path(out_path)
    pred = pd.read_parquet(predictor_table,
                           columns=["subject", "run", "sample", "in_predictor_set", "valid", "is_target"])
    pred = pred[pred["in_predictor_set"].astype(bool) & pred["valid"].astype(bool) & pred["is_target"].astype(bool)]
    jobs = [(s, r, g["sample"].to_numpy()) for (s, r), g in pred.groupby(["subject", "run"])]
    log.info("extracting %d recordings", len(jobs))
    with ProcessPoolExecutor(n_jobs) as ex:
        parts = list(ex.map(extract_recording, jobs))
    d = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_path)
    log.info("wrote %d trials; detectable %d", len(d), int(d["saccades_detectable"].sum()))
    return out_path


def saccade_refits(r: Run, saccade_table) -> dict:
    """M3-H2 without trials with a detected saccade in each window; flagged-trial rates by group."""
    sac = pd.read_parquet(saccade_table)
    sac["pid"] = sac["subject"].map(r.provider.ps)
    md0 = r.provider.model("M3-H2")
    f = md0.frame.merge(sac.drop(columns=["subject"]), on=["pid", "run", "sample"], how="left", validate="one_to_one")
    out: dict = {"rows_m3h2": int(len(f))}
    st = r.state("S2")
    for key in WINDOWS:
        valid = f[f"sacc_valid_{key}"].fillna(False).astype(bool)
        flag = (f[f"n_sacc_{key}"].fillna(0) > 0) & valid
        # per-child rate of flagged trials among detection-valid trials, by group
        g = f.loc[valid].assign(flag=flag[valid].astype(float)).groupby("pid").agg(grp=("grp", "first"),
                                                                                   rate=("flag", "mean"))
        rate = welch(g.loc[g["grp"] < 0, "rate"].to_numpy(), g.loc[g["grp"] > 0, "rate"].to_numpy())
        keep = valid & ~flag
        md = SENS.refinalise(SENS._replace(md0, frame=f.loc[keep.to_numpy()].reset_index(drop=True)),
                             reason=f"no saccade in window {key}")
        res = fit_targets(r, md, f"saccade_{key}", H2T(), state=st)
        out[key] = {"window_ms": list(WINDOWS[key]), "rows_kept": int(keep.sum()),
                    "rows_detection_valid": int(valid.sum()), "rows_flagged": int(flag.sum()),
                    "flagged_rate_by_group": rate, "fit": res}
    out["definition"] = ("horizontal-EOG saccade rule of the ocular quality control; M3-H2 on targets with valid "
                         "detection and no detected saccade in the window; final M3 random-effects structure")
    r.write("saccade_check", out)
    return out
