"""Channel-level alpha for the scalp maps of alpha-RT coupling (Additional file 1, Supplementary Methods S4, R13).

Extraction: for every row of M1-H2 / M3-H2 / M4-H2, log10 IAF-band power of each of the 64 scalp channels in the
stored primary window (``win_start``, ``win_stop``, ``band_lo_hz``, ``band_hi_hz`` of the predictor table), with the
predictor's cleaned data (ICA applied, bad channels left out) and DPSS estimator; a channel-window above 200 uV peak
to peak is dropped. Check: the primary-ROI value recomputed from the same spectra reproduces the stored ``alpha``
(|difference| < 1e-6); failure control: the same windows shifted by +100 ms must not. FAST also accumulates
target-locked evoked-response sums per child and within-child ``a_wb`` tercile (-100 to 500 ms, baseline -50 to 0 ms,
the rejection rule of :mod:`statistical_analysis.evoked_response_pathway`). The analysis is in
:mod:`statistical_analysis.cluster_permutation`.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from statistical_analysis.evoked_response_pathway import ERP_SETTINGS

log = logging.getLogger(__name__)

TASKS = {"FAST": "M3-H2", "IC": "M4-H2", "AVSRT": "M1-H2"}
ERP_MS = (-100.0, 500.0)
ERP_BASE_MS = (-50.0, 0.0)
SHIFT_MS = 100.0          # failure control of the ROI check
SEED = 20261101
N_PERM = 5000
MIN_ROWS = 30


def channels() -> list[str]:
    import mne

    return list(mne.channels.make_standard_montage("biosemi64").ch_names)


def _ms(ms: float, sf: float) -> int:
    return int(round(ms * sf / 1000.0))


# extraction -----------------------------------------------------------------------------------------------------------
def extract_recording(job: tuple) -> dict:
    task, subject, run, rows = job
    from feature_extraction import spectral as sp
    from feature_extraction.trial_features import load_clean_checked, load_settings
    from preprocessing import io as pio

    st = load_settings()
    A = st.features.alpha
    chans = channels()
    stem = f"{subject}_task-{task}_run-{int(run):02d}"
    outs = pio.recording_outputs(st.preproc_root, (task, subject, stem))
    raw = load_clean_checked(str(outs["qc"]), apply_ica=True, root=st.preproc_root)
    seg = pio.read_segments(outs["segments"])
    sf = float(raw.info["sfreq"])
    bads = set(raw.info["bads"])
    good = [c for c in chans if c in raw.ch_names and c not in bads]
    X = raw.get_data(picks=good) * A.data_scale
    bad_sp = sp.bad_annotation_spans(raw, A.bad_annotation_prefix)
    n_times = raw.n_times
    del raw
    col = np.array([chans.index(c) for c in good])
    prim = [c for c in A.roi("primary").channels if c in good]
    pidx = [good.index(c) for c in prim]
    thr = A.artefact_ptp_uv * A.data_scale / 1e6
    ws = np.asarray(rows["win_start"], np.int64)
    lens = np.asarray(rows["win_stop"], np.int64) - ws
    if len(set(lens.tolist())) != 1:
        raise ValueError(f"{stem}: window lengths differ {set(lens.tolist())}")
    n = int(lens[0])
    band = (float(rows["band_lo"][0]), float(rows["band_hi"][0]))
    chan = np.full((len(ws), len(chans)), np.nan, np.float32)
    roi = np.full(len(ws), np.nan)
    roi_shift = np.full(len(ws), np.nan)
    shift = _ms(SHIFT_MS, sf)
    for c0 in range(0, len(ws), 64):
        sl = slice(c0, c0 + 64)
        W = sp.extract_windows(X, ws[sl], n)
        P, f = sp.psd_estimate(W, sf, "dpss", A)
        bp = sp.band_mean_power(P, f, *band, tol=A.band_tol_hz)               # (rows, good channels), linear
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.log10(bp)
            roi[sl] = np.log10(bp[:, pidx].mean(axis=1))
        lp[np.ptp(W, axis=-1) > thr] = np.nan
        chan[sl, col] = lp.astype(np.float32)
        W2 = sp.extract_windows(X, ws[sl] + shift, n)
        P2, f2 = sp.psd_estimate(W2, sf, "dpss", A)
        with np.errstate(divide="ignore", invalid="ignore"):
            roi_shift[sl] = np.log10(sp.band_mean_power(P2, f2, *band, tol=A.band_tol_hz)[:, pidx].mean(axis=1))
    out = {"row": np.asarray(rows["row"]), "chan": chan, "roi": roi, "roi_shift": roi_shift,
           "n_good": len(good), "sfreq": sf}
    if task == "FAST":
        s = np.asarray(rows["sample"], np.int64)
        a, b = s + _ms(ERP_MS[0], sf), s + _ms(ERP_MS[1], sf)
        margin = int(round(A.edge_margin_s * sf))
        reason = sp.window_validity(a, b, n_times=n_times, seg=seg, margin=margin, bad_spans=bad_sp)
        n_ep = int(b[0] - a[0])
        t = (np.arange(n_ep) + _ms(ERP_MS[0], sf)) / sf * 1000.0
        bl = (t >= ERP_BASE_MS[0]) & (t < ERP_BASE_MS[1])
        rej = [good.index(c) for c in ERP_SETTINGS["p1"]["channels"] + ERP_SETTINGS["n1"]["channels"] if c in good]
        terc = np.asarray(rows["tercile"], int)
        sums = np.zeros((3, len(chans), n_ep))
        cnt = np.zeros((3, len(chans)))
        for i in np.flatnonzero(reason == ""):
            ep = X[:, a[i]:b[i]]
            if ep.shape[1] != n_ep or not rej or np.ptp(ep[rej], axis=1).max() > ERP_SETTINGS["reject_uv"]:
                continue
            ep = ep - ep[:, bl].mean(axis=1, keepdims=True)
            sums[terc[i], col] += ep
            cnt[terc[i], col] += 1
        out.update(erp_sums=sums, erp_cnt=cnt, erp_t=t)
    return out


def _model_rows(r, task: str, features_root) -> pd.DataFrame:
    """Group-model rows of ``task`` joined to their predictor-table window, band and IAF; within-child a_wb tercile."""
    f = r.provider.model(TASKS[task]).frame
    keep = ["pid", "run", "sample", "a_wb", "lrt", "grp", "pid_run_block"]
    f = f[keep].copy()
    pred = pd.read_parquet(Path(features_root) / task / f"{task}_conf_features_predictor.parquet",
                           columns=["subject", "run", "sample", "win_start", "win_stop", "band_lo_hz", "band_hi_hz",
                                    "alpha", "iaf_hz"])
    pred["pid"] = pred["subject"].map(r.provider.ps)
    m = f.merge(pred, on=["pid", "run", "sample"], how="left", validate="one_to_one")
    if m["subject"].isna().any():
        raise SystemExit(f"{task}: {int(m['subject'].isna().sum())} model rows without a predictor row")
    m["row"] = np.arange(len(m))
    rk = m.groupby("pid")["a_wb"].rank(method="first")
    m["tercile"] = (3 * (rk - 1) // m.groupby("pid")["a_wb"].transform("count")).astype(int)
    return m


def extract(r, tasks: list[str], features_root, work_dir, n_jobs: int = 12) -> None:
    """Channel-level alpha of every group-model row of ``tasks`` (and FAST evoked-response sums) into ``work_dir``."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    chans = channels()
    for task in tasks:
        t0 = time.time()
        m = _model_rows(r, task, features_root)
        jobs = []
        for (subj, run), g in m.groupby(["subject", "run"], sort=True):
            jobs.append((task, subj, int(run), {"row": g["row"].to_numpy(), "sample": g["sample"].to_numpy(),
                                                 "win_start": g["win_start"].to_numpy(),
                                                 "win_stop": g["win_stop"].to_numpy(),
                                                 "band_lo": g["band_lo_hz"].to_numpy(),
                                                 "band_hi": g["band_hi_hz"].to_numpy(),
                                                 "tercile": g["tercile"].to_numpy()}))
        log.info("%s: %d model rows, %d recordings", task, len(m), len(jobs))
        chan = np.full((len(m), len(chans)), np.nan, np.float32)
        roi = np.full(len(m), np.nan)
        roi_shift = np.full(len(m), np.nan)
        erp: dict = {}
        sub_pid = dict(zip(m["subject"], m["pid"]))
        with ProcessPoolExecutor(n_jobs) as ex:
            for i, (job, res) in enumerate(zip(jobs, ex.map(extract_recording, jobs))):
                idx = res["row"]
                chan[idx], roi[idx], roi_shift[idx] = res["chan"], res["roi"], res["roi_shift"]
                if "erp_sums" in res:
                    pid = sub_pid[job[1]]
                    if pid not in erp:
                        erp[pid] = [np.zeros_like(res["erp_sums"]), np.zeros_like(res["erp_cnt"])]
                    erp[pid][0] += res["erp_sums"]
                    erp[pid][1] += res["erp_cnt"]
                    erp_t = res["erp_t"]
                if (i + 1) % 25 == 0:
                    log.info("%s: %d/%d", task, i + 1, len(jobs))
        done = np.isfinite(roi)
        d_ok = np.abs(roi[done] - m["alpha"].to_numpy(float)[done])
        d_sh = np.abs(roi_shift[done] - m["alpha"].to_numpy(float)[done])
        check = {"rows_extracted": int(done.sum()), "rows_model": int(len(m)),
                 "max_abs_diff_roi_vs_stored": float(np.nanmax(d_ok)), "pass": bool(np.nanmax(d_ok) < 1e-6),
                 "failure_control": {"shift_ms": SHIFT_MS, "max_abs_diff": float(np.nanmax(d_sh)),
                                     "median_abs_diff": float(np.nanmedian(d_sh)),
                                     "fails_as_required": bool(np.nanmax(d_sh) >= 1e-6)}}
        log.info("%s ROI check: %s", task, json.dumps(check))
        if not (check["pass"] and check["failure_control"]["fails_as_required"]):
            raise SystemExit(f"{task}: ROI check failed; stop")
        out = m[["pid", "run", "sample"]].copy()
        out["roi_recomputed"] = roi
        cdf = pd.DataFrame(chan, columns=chans)
        pd.concat([out, cdf], axis=1).to_parquet(work_dir / f"{task}_channel_alpha.parquet")
        (work_dir / f"{task}_channel_alpha_check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
        if erp:
            pids = sorted(erp)
            np.savez_compressed(work_dir / "FAST_erp_sums.npz", pids=np.array(pids),
                                sums=np.stack([erp[p][0] for p in pids]), cnt=np.stack([erp[p][1] for p in pids]),
                                t=erp_t, channels=np.array(chans))
        log.info("%s done in %.0f s", task, time.time() - t0)
