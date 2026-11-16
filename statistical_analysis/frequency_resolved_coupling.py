"""Frequency-resolved coupling, sex and age-overlap refits (Additional file 1, Supplementary Methods S4, R14a-c;
Table S12, Figure S9).

Extraction: for every row of M3-H2 / M4-H2 / M1-H2, the DPSS power spectrum of the stored primary window averaged over
the primary posterior ROI (cleaned data, ICA applied, bad channels left out), log10, from 2 to 40
Hz on the estimator's frequency grid.
Check: the IAF-band mean of the same ROI spectrum reproduces the stored ``alpha`` (|diff| < 1e-6); failure control:
windows shifted by +100 ms must not.
Analysis: per child and frequency, the two-stage standardised slope (% change in RT per SD of within-block log power);
group means, ASD - TD Welch t, 1-D cluster-based permutation tests over adjacent frequencies (TD vs 0, ASD vs 0,
ASD - TD; 5,000 permutations), on the absolute and the IAF-aligned axis; the interpretation rule ("alpha-specific" if
every significant difference cluster lies within 7-14 Hz).
Sex: M3-H2 / M4-H2 / M1-H2 refitted on boys only, and with group x sex x alpha terms on the full sample.
Age: the three task models refitted on children no older than the oldest TD child.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

from modeling import sensitivity as SENS
from statistical_analysis.cluster_permutation import child_slopes, cluster_test
from statistical_analysis.follow_up_analyses import H2T, add_cols, fit_targets
from statistical_analysis.scalp_coupling_maps import N_PERM, SEED, SHIFT_MS, TASKS, _model_rows, _ms

log = logging.getLogger(__name__)

F_LO, F_HI = 2.0, 40.0
ALIGN = np.round(np.arange(-6.0, 20.0 + 1e-9, 0.5), 2)
STATES = {"FAST": "S2", "IC": "S3", "AVSRT": "P1"}


# extraction -----------------------------------------------------------------------------------------------------------
def extract_recording(job: tuple) -> dict:
    task, subject, run, rows = job
    from feature_extraction import spectral as sp
    from feature_extraction.trial_features import load_clean_checked, load_settings
    from preprocessing import io as pio

    st = load_settings()
    A = st.features.alpha
    stem = f"{subject}_task-{task}_run-{int(run):02d}"
    outs = pio.recording_outputs(st.preproc_root, (task, subject, stem))
    raw = load_clean_checked(str(outs["qc"]), apply_ica=True, root=st.preproc_root)
    sf = float(raw.info["sfreq"])
    bads = set(raw.info["bads"])
    prim = [c for c in A.roi("primary").channels if c in raw.ch_names and c not in bads]
    X = raw.get_data(picks=prim) * A.data_scale
    del raw
    ws = np.asarray(rows["win_start"], np.int64)
    lens = np.asarray(rows["win_stop"], np.int64) - ws
    if len(set(lens.tolist())) != 1:
        raise ValueError(f"{stem}: window lengths differ {set(lens.tolist())}")
    n = int(lens[0])
    band = (float(rows["band_lo"][0]), float(rows["band_hi"][0]))
    shift = _ms(SHIFT_MS, sf)
    spec, roi, roi_shift, freqs = [], [], [], None
    for c0 in range(0, len(ws), 64):
        sl = slice(c0, c0 + 64)
        P, f = sp.psd_estimate(sp.extract_windows(X, ws[sl], n), sf, "dpss", A)
        Pm = P.mean(axis=1)                                                     # ROI mean, linear (rows, freqs)
        keep = (f >= F_LO - 1e-9) & (f <= F_HI + 1e-9)
        freqs = f[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            spec.append(np.log10(Pm[:, keep]).astype(np.float32))
            roi.append(np.log10(sp.band_mean_power(Pm, f, *band, tol=A.band_tol_hz)))
            P2, f2 = sp.psd_estimate(sp.extract_windows(X, ws[sl] + shift, n), sf, "dpss", A)
            roi_shift.append(np.log10(sp.band_mean_power(P2.mean(axis=1), f2, *band, tol=A.band_tol_hz)))
    return {"row": np.asarray(rows["row"]), "spec": np.concatenate(spec), "roi": np.concatenate(roi),
            "roi_shift": np.concatenate(roi_shift), "freqs": freqs}


def extract(r, tasks: list[str], features_root, work_dir, n_jobs: int = 12) -> None:
    """Posterior-ROI spectra (2-40 Hz) of every group-model row of ``tasks`` into ``work_dir``."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        t0 = time.time()
        m = _model_rows(r, task, features_root)
        jobs = [(task, s, int(run), {"row": g["row"].to_numpy(), "win_start": g["win_start"].to_numpy(),
                                     "win_stop": g["win_stop"].to_numpy(), "band_lo": g["band_lo_hz"].to_numpy(),
                                     "band_hi": g["band_hi_hz"].to_numpy()})
                for (s, run), g in m.groupby(["subject", "run"], sort=True)]
        log.info("%s: %d model rows, %d recordings", task, len(m), len(jobs))
        spec = roi = roi_shift = freqs = None
        with ProcessPoolExecutor(n_jobs) as ex:
            for i, res in enumerate(ex.map(extract_recording, jobs)):
                if spec is None:
                    freqs = res["freqs"]
                    spec = np.full((len(m), len(freqs)), np.nan, np.float32)
                    roi, roi_shift = np.full(len(m), np.nan), np.full(len(m), np.nan)
                if not np.allclose(res["freqs"], freqs):
                    raise ValueError("frequency grids differ between recordings")
                idx = res["row"]
                spec[idx], roi[idx], roi_shift[idx] = res["spec"], res["roi"], res["roi_shift"]
                if (i + 1) % 25 == 0:
                    log.info("%s: %d/%d", task, i + 1, len(jobs))
        done = np.isfinite(roi)
        stored = m["alpha"].to_numpy(float)
        d_ok, d_sh = np.abs(roi[done] - stored[done]), np.abs(roi_shift[done] - stored[done])
        check = {"rows_extracted": int(done.sum()), "rows_model": int(len(m)),
                 "max_abs_diff_roi_vs_stored": float(np.nanmax(d_ok)), "pass": bool(np.nanmax(d_ok) < 1e-6),
                 "failure_control": {"shift_ms": SHIFT_MS, "max_abs_diff": float(np.nanmax(d_sh)),
                                     "median_abs_diff": float(np.nanmedian(d_sh)),
                                     "fails_as_required": bool(np.nanmedian(d_sh) >= 1e-6)}}
        log.info("%s ROI check: %s", task, json.dumps(check))
        if not (check["pass"] and check["failure_control"]["fails_as_required"] and done.all()):
            raise SystemExit(f"{task}: ROI check failed or rows missing; stop")
        out = m[["pid", "run", "sample"]].copy()
        out[[f"f{x:.1f}" for x in freqs]] = spec
        out.to_parquet(work_dir / f"{task}_roi_spectra.parquet")
        (work_dir / f"{task}_roi_spectra_check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
        log.info("%s done in %.0f s", task, time.time() - t0)


# analysis -------------------------------------------------------------------------------------------------------------
def chain(n: int):
    return sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="csr")


def interpret(diff_clusters: list[dict], freqs: np.ndarray) -> str:
    sig = [c for c in diff_clusters if c["p"] < 0.05]
    if not sig:
        return "not resolved"
    spans = [(min(map(float, c["channels"])), max(map(float, c["channels"]))) for c in sig]
    if any(hi - lo > 10 and (lo < 6 or hi > 15) for lo, hi in spans):
        return "broadband"
    inside = all(lo >= 7 and hi <= 14 for lo, hi in spans)
    outside = any(hi < 6 or lo > 15 or lo < 6 or hi > 15 for lo, hi in spans)
    return "alpha-specific" if inside and not outside else "not resolved"


def describe(S: np.ndarray, asd: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(labels):
        a, b = S[~asd, j], S[asd, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        rec = {"x": float(lab), "n_TD": len(a), "n_ASD": len(b)}
        for g, v in (("TD", a), ("ASD", b)):
            q = stats.t.ppf(0.975, len(v) - 1)
            se = v.std(ddof=1) / np.sqrt(len(v))
            rec.update({f"mean_{g}": v.mean(), f"lo_{g}": v.mean() - q * se, f"hi_{g}": v.mean() + q * se})
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
        q = stats.t.ppf(0.975, df)
        d = b.mean() - a.mean()
        rec.update({"diff": d, "lo_diff": d - q * se, "hi_diff": d + q * se, "t_diff": d / se})
        rows.append(rec)
    return pd.DataFrame(rows)


def frequency_coupling(r, tasks: list[str], features_root, work_dir) -> dict:
    """Two-stage coupling at each frequency, cluster tests on the absolute and IAF-aligned axes; writes
    ``frequency_coupling.json`` and the per-frequency group tables ``frequency_coupling_<task>_<axis>.csv`` to
    ``r.out``."""
    work_dir = Path(work_dir)
    r.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"definition": "per child and frequency: OLS slope of log RT on log10 ROI power, both "
                 "demeaned within child x run x block, x 100 x the child's SD of the demeaned power (% RT per SD); "
                 "cluster-based permutation over adjacent bins, cluster-forming two-sided p < 0.05",
                 "n_perm": N_PERM, "seed": SEED, "tasks": {}}
    for task in tasks:
        m = _model_rows(r, task, features_root)
        sp_ = pd.read_parquet(work_dir / f"{task}_roi_spectra.parquet")
        if not (sp_["pid"].to_numpy() == m["pid"].to_numpy()).all() or len(sp_) != len(m):
            raise SystemExit(f"{task}: spectra rows do not match the model rows")
        fcols = [c for c in sp_.columns if c.startswith("f")]
        freqs = np.array([float(c[1:]) for c in fcols])
        df = pd.concat([m[["pid", "pid_run_block", "lrt"]].reset_index(drop=True), sp_[fcols]], axis=1)
        S = child_slopes(df, fcols, standardise=True)
        grp = m.groupby("pid")["grp"].first().reindex(S.index)
        asd = (grp > 0).to_numpy()
        iaf = m.groupby("pid")["iaf_hz"].median().reindex(S.index).to_numpy(float)
        Sv = S.to_numpy(float)
        Sa = np.full((len(Sv), len(ALIGN)), np.nan)
        for i in range(len(Sv)):
            x = iaf[i] + ALIGN
            ok = (x >= freqs[0]) & (x <= freqs[-1])
            Sa[i, ok] = np.interp(x[ok], freqs, Sv[i])
        rec: dict = {"n_TD": int((~asd).sum()), "n_ASD": int(asd.sum()), "freq_step_hz": float(np.diff(freqs)[0]),
                     "check": json.loads((work_dir / f"{task}_roi_spectra_check.json").read_text(encoding="utf-8"))}
        for axis, M, labs in (("absolute", Sv, [f"{x:.1f}" for x in freqs]),
                              ("iaf_aligned", Sa, [f"{x:.1f}" for x in ALIGN])):
            adj = chain(len(labs))
            out = {}
            for kind, sub, a in (("diff", M, asd), ("TD", M[~asd], None), ("ASD", M[asd], None)):
                rng = np.random.default_rng(SEED)
                ct = cluster_test(sub, adj, "diff" if kind == "diff" else "one", a, labs, rng)
                ct.pop("t")
                for c in ct["clusters"]:
                    c["range"] = [float(c["channels"][0]), float(c["channels"][-1])]
                    c["n_bins"] = len(c.pop("channels"))
                out[kind] = ct
            tab = describe(M, asd, labs)
            tab.to_csv(r.out / f"frequency_coupling_{task}_{axis}.csv", index=False)
            pk = tab.loc[tab["t_diff"].abs().idxmax()]
            out["peak_difference"] = {"x": float(pk["x"]), "diff": float(pk["diff"]),
                                      "ci95": [float(pk["lo_diff"]), float(pk["hi_diff"])], "t": float(pk["t_diff"])}
            if axis == "absolute":
                out["interpretation"] = interpret([{**c, "channels": [str(c["range"][0]), str(c["range"][1])]}
                                                   for c in out["diff"]["clusters"]], freqs)
            rec[axis] = out
        res["tasks"][task] = rec
        log.info("%s interpretation: %s; peak %s", task, rec["absolute"]["interpretation"],
                 rec["absolute"]["peak_difference"])
    r.write("frequency_coupling", res)
    return res


# sex ------------------------------------------------------------------------------------------------------------------
def sex_refits(r, tasks: list[str]) -> dict:
    """Group models refitted in boys only and with group x sex x alpha terms (R14b)."""
    res: dict = {"coding": "sex_c +0.5 boy / -0.5 girl; grp +0.5 ASD / -0.5 TD", "tasks": {}}
    for task in tasks:
        md = r.provider.model(TASKS[task])
        st = r.state(STATES[task])
        f = md.frame
        kids = f.groupby("pid")[["sex_c", "grp"]].first()
        counts = {f"{'boys' if s > 0 else 'girls'}_{'ASD' if g > 0 else 'TD'}": int(n)
                  for (s, g), n in kids.value_counts().items()}
        boys = SENS._drop_terms(SENS._filter(md, (f["sex_c"] > 0).to_numpy(), "boys only"),
                                ["sex_c", "a_wb:sex_c"])
        rb = fit_targets(r, boys, f"boys_{task}", H2T(), state=st)
        a, g, s = (f[c].to_numpy(float) for c in ("a_wb", "grp", "sex_c"))
        m3 = add_cols(md, {"sex_x_grp": s * g, "awb_x_sex_x_grp": a * s * g})
        tg = {"H2|boys": {"a_wb:grp": 1.0, "awb_x_sex_x_grp": 0.5},
              "H2|girls": {"a_wb:grp": 1.0, "awb_x_sex_x_grp": -0.5},
              "H2|boys-girls": {"awb_x_sex_x_grp": 1.0}}
        for gl, gv in (("TD", -0.5), ("ASD", 0.5)):
            for sl, sv in (("boys", 0.5), ("girls", -0.5)):
                tg[f"a_wb|{gl}|{sl}"] = {"a_wb": 1.0, "a_wb:grp": gv, "a_wb:sex_c": sv, "awb_x_sex_x_grp": gv * sv}
        r3 = fit_targets(r, m3, f"sex_group_{task}", tg, state=st)
        res["tasks"][task] = {"model": TASKS[task], "children": counts, "boys_only": rb, "group_x_sex": r3}
    r.write("sex_refits", res)
    return res


# age overlap ----------------------------------------------------------------------------------------------------------
def age_overlap_refits(r, tasks: list[str]) -> dict:
    """Refit on children no older than the oldest TD child of the task's model data (R14c)."""
    res: dict = {"tasks": {}}
    for task in tasks:
        md = r.provider.model(TASKS[task])
        f = md.frame
        kids = f.groupby("pid")[["age", "grp"]].first()
        cap = float(kids.loc[kids["grp"] < 0, "age"].max())
        keep = (f["age"] <= cap + 1e-9).to_numpy()
        sub = SENS._filter(md, keep, f"age <= {cap}")
        k2 = sub.frame.groupby("pid")["grp"].first()
        out = fit_targets(r, sub, f"age_overlap_{task}", H2T(), state=r.state(STATES[task]))
        res["tasks"][task] = {"model": TASKS[task], "age_cap": cap, "n_TD": int((k2 < 0).sum()),
                              "n_ASD": int((k2 > 0).sum()), "fit": out}
    r.write("age_overlap_refits", res)
    return res
