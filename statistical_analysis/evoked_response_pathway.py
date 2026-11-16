"""Prestimulus alpha -> single-trial visual evoked response -> RT in FAST, by group (Additional file 1,
Supplementary Methods S4, R7 / R7b).

Extraction (EEG side only): for every FAST target with a valid primary prestimulus window in the predictor table, the
single-trial visual evoked response is measured on the cleaned analysis data (ICA applied, bad channels not
interpolated and left out): epoch -100 to +300 ms; P1 = mean over 80-130 ms of the O1/Oz/O2 average, N1 window =
mean over 150-220 ms of the P7/P8/PO7/PO8 average, each with a -100 to 0 ms baseline, without baseline and with a
-50 to 0 ms baseline (which does not overlap the alpha window). An epoch is used only if it passes the window-validity
rules (segment, edge margin, no BAD* overlap) and its peak-to-peak on these channels is <= 200 uV.

Models, joined to the M3-H2 rows (FAST hits) by (pid, run, sample):
(a) evoked response ~ a_wb x grp (+ the M3 nuisance terms without the RT-history terms), simplification path;
(b) M3-H2 + evoked response (z within child) and its x grp product, final M3 structure: does the evoked response
predict RT, by group, and how much of the alpha slope remains (direct effect);
(c) indirect effect a x b per group with a Monte Carlo 95% interval (independent normal a, b; fixed effects only).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import sensitivity as SENS
from modeling.model_specs import RandomDef
from statistical_analysis.follow_up_analyses import H2T, add_cols, by_group, fit_targets
from statistical_analysis.group_difference_checks import Run

log = logging.getLogger(__name__)

ERP_SETTINGS = {"baseline_ms": [-100, 0],
                "p1": {"window_ms": [80, 130], "channels": ["O1", "Oz", "O2"]},
                "n1": {"window_ms": [150, 220], "channels": ["P7", "P8", "PO7", "PO8"]},
                "reject_uv": 200}      # peak-to-peak in [-100, 300] ms on the evoked-response channels
CH = {"p1": list(ERP_SETTINGS["p1"]["channels"]), "n1": list(ERP_SETTINGS["n1"]["channels"])}


def _ms(ms: float, sf: float) -> int:
    return int(round(ms * sf / 1000.0))


def extract_recording(job: tuple) -> pd.DataFrame:
    subject, run, samples = job
    from feature_extraction import spectral as sp
    from feature_extraction.trial_features import load_clean_checked, load_settings
    from preprocessing import io as pio

    st = load_settings()
    stem = f"{subject}_task-FAST_run-{int(run):02d}"
    outs = pio.recording_outputs(st.preproc_root, ("FAST", subject, stem))
    raw = load_clean_checked(str(outs["qc"]), apply_ica=True, root=st.preproc_root)
    seg = pio.read_segments(outs["segments"])
    sf = float(raw.info["sfreq"])
    bads = set(raw.info["bads"])
    good = {k: [c for c in v if c in raw.ch_names and c not in bads] for k, v in CH.items()}
    allc = sorted(set(good["p1"]) | set(good["n1"]))
    bad_sp = sp.bad_annotation_spans(raw, "BAD")
    n_times = raw.n_times
    X = raw.get_data(picks=allc) * 1e6
    del raw
    s = np.asarray(samples, np.int64)
    a, b = s + _ms(-100, sf), s + _ms(300, sf)
    margin = int(round(st.features.alpha.edge_margin_s * sf))
    reason = sp.window_validity(a, b, n_times=n_times, seg=seg, margin=margin, bad_spans=bad_sp)
    out = {"subject": subject, "run": run, "sample": s, "erp_valid": reason == "", "erp_reason": reason}
    n_ep = b[0] - a[0]
    t = (np.arange(n_ep) + _ms(-100, sf)) / sf * 1000.0
    bl = (t >= ERP_SETTINGS["baseline_ms"][0]) & (t < ERP_SETTINGS["baseline_ms"][1])
    bl50 = (t >= -50.0) & (t < 0.0)          # does not overlap the alpha window (ends at -50 ms)
    res = {f"{k}{suf}": np.full(len(s), np.nan) for k in ("p1", "n1") for suf in ("", "_nobl", "_bl50")}
    ptp = np.full(len(s), np.nan)
    for i in np.flatnonzero(reason == ""):
        ep = X[:, a[i]:b[i]]
        if ep.shape[1] != n_ep:
            out["erp_valid"][i] = False
            continue
        ptp[i] = float(np.max(np.ptp(ep, axis=1)))
        epb = ep - ep[:, bl].mean(axis=1, keepdims=True)
        for k in ("p1", "n1"):
            idx = [allc.index(c) for c in good[k]]
            if len(idx) < 2:
                continue
            w0, w1 = ERP_SETTINGS[k]["window_ms"]
            win = (t >= w0) & (t <= w1)
            res[k][i] = float(epb[idx][:, win].mean())
            res[f"{k}_nobl"][i] = float(ep[idx][:, win].mean())
            res[f"{k}_bl50"][i] = float((ep[idx] - ep[idx][:, bl50].mean(axis=1, keepdims=True))[:, win].mean())
    out.update(res)
    out["erp_ptp_uv"] = ptp
    out["n_p1_channels"], out["n_n1_channels"] = len(good["p1"]), len(good["n1"])
    df = pd.DataFrame(out)
    df["erp_valid"] = df["erp_valid"] & (df["erp_ptp_uv"] <= ERP_SETTINGS["reject_uv"])
    return df


def paths_baseline_50ms(r: Run, erp_table) -> dict:
    """The evoked-response paths with P1 / N1-window amplitudes measured against a -50 to 0 ms baseline (R7b)."""
    erp = pd.read_parquet(erp_table)
    erp = erp[erp["erp_valid"]].copy()
    erp["pid"] = erp["subject"].map(r.provider.ps)
    md0 = r.provider.model("M3-H2")
    f = md0.frame.merge(erp[["pid", "run", "sample", "p1_bl50", "n1_bl50"]], on=["pid", "run", "sample"],
                        how="left", validate="one_to_one")
    ok = np.isfinite(f["p1_bl50"].to_numpy(float)) & np.isfinite(f["n1_bl50"].to_numpy(float))
    md = SENS.refinalise(SENS._replace(md0, frame=f.loc[ok].reset_index(drop=True)),
                         reason="valid evoked response (-50 to 0 ms baseline)")
    mf = md.frame
    out: dict = {"rows": int(len(mf)), "baseline_ms": [-50, 0]}
    st = r.state("S2")
    keep_terms = [t for t in md.fixed_terms if t in ("a_wb", "a_slow", "cat", "half_status", "heog_z", "vfp_z",
                                                      "heog_missing", "vfp_missing", "age_c", "sex_c", "grp",
                                                      "a_wb:grp")
                  or t.startswith("ns_t_run") or t.startswith("ns_tib")]
    g = mf["grp"].to_numpy(float)
    for comp in ("p1_bl50", "n1_bl50"):
        z = ((mf[comp] - mf.groupby("pid")[comp].transform("mean")) / mf.groupby("pid")[comp].transform("std"))
        ff = mf.copy()
        ff["erp_y"] = z.to_numpy(float)
        mdef = replace(md.model_def, family="lmm", outcome="erp_y",
                       random=RandomDef(corr=("1", "a_wb"), uncorrelated=("t_run", "tib")),
                       targets={}, grp_targets={}, sesoi={})
        ma = replace(md, frame=ff, model_def=mdef, outcome_column="erp_y", fixed_terms=keep_terms)
        out[f"a_alpha_to_{comp}_z"] = fit_targets(r, ma, f"erp_bl50_a_{comp}_z", H2T(), cr2=True)
        mb = add_cols(md, {f"{comp}_z": z.to_numpy(float), f"{comp}_z_x_grp": z.to_numpy(float) * g})
        out[f"b_{comp}_to_rt"] = fit_targets(r, mb, f"erp_bl50_b_{comp}", {**by_group(f"{comp}_z", f"{comp}_z_x_grp"),
                                                                           **H2T()}, state=st)
    out["c_total_same_rows"] = fit_targets(r, md, "erp_bl50_c_total", H2T(), state=st)
    out["units"] = ("a: within-child SD of the evoked response per SD_i of alpha; b: log RT per within-child SD of the "
                    "evoked response; c: log RT per SD_i of alpha")
    r.write("evoked_response_pathway_baseline_50ms", out)
    return out


def extract(predictor_table, out_path, n_jobs: int = 10) -> Path:
    """Single-trial evoked responses of all FAST targets with a valid prestimulus window; trial-level parquet."""
    out_path = Path(out_path)
    pred = pd.read_parquet(predictor_table,
                           columns=["subject", "run", "sample", "in_predictor_set", "valid", "is_target"])
    pred = pred[pred["in_predictor_set"].astype(bool) & pred["valid"].astype(bool) & pred["is_target"].astype(bool)]
    jobs = [(subj, run, g["sample"].to_numpy()) for (subj, run), g in pred.groupby(["subject", "run"])]
    log.info("extracting %d recordings", len(jobs))
    parts = []
    with ProcessPoolExecutor(n_jobs) as ex:
        for i, df in enumerate(ex.map(extract_recording, jobs)):
            parts.append(df)
            if (i + 1) % 20 == 0:
                log.info("%d/%d recordings", i + 1, len(jobs))
    d = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_path)
    log.info("wrote %d trials (%d valid) to %s", len(d), int(d["erp_valid"].sum()), out_path)
    return out_path


def evoked_response_paths(r: Run, erp_table) -> dict:
    """Paths alpha -> evoked response -> RT with the -100 to 0 ms baseline and the Monte Carlo indirect effect (R7)."""
    erp = pd.read_parquet(erp_table)
    erp = erp[erp["erp_valid"]].copy()
    erp["pid"] = erp["subject"].map(r.provider.ps)
    erp = erp.drop(columns=["subject"])
    md0 = r.provider.model("M3-H2")
    f = md0.frame.merge(erp[["pid", "run", "sample", "p1", "n1", "p1_nobl", "n1_nobl"]],
                        on=["pid", "run", "sample"], how="left", validate="one_to_one")
    ok = np.isfinite(f["p1"].to_numpy(float)) & np.isfinite(f["n1"].to_numpy(float))
    md = SENS.refinalise(SENS._replace(md0, frame=f.loc[ok].reset_index(drop=True)), reason="valid evoked response")
    mf = md.frame
    out: dict = {"rows": int(len(mf)), "rows_m3h2": int(len(md0.frame)),
                 "children": {"TD": int(mf.loc[mf["grp"] < 0, "pid"].nunique()),
                              "ASD": int(mf.loc[mf["grp"] > 0, "pid"].nunique())}}
    # descriptives: mean evoked responses by group (child means)
    cm = mf.groupby("pid").agg(grp=("grp", "first"), p1=("p1", "mean"), n1=("n1", "mean"))
    out["child_mean_uv"] = {k: {"TD": float(cm.loc[cm["grp"] < 0, k].mean()),
                                "ASD": float(cm.loc[cm["grp"] > 0, k].mean())} for k in ("p1", "n1")}
    st = r.state("S2")
    for comp in ("p1", "n1"):
        # z within child
        z = (mf[comp] - mf.groupby("pid")[comp].transform("mean")) / mf.groupby("pid")[comp].transform("std")
        # (a) alpha -> evoked response
        keep_terms = [t for t in md.fixed_terms if t in ("a_wb", "a_slow", "cat", "half_status", "heog_z", "vfp_z",
                                                          "heog_missing", "vfp_missing", "age_c", "sex_c", "grp",
                                                          "a_wb:grp")
                      or t.startswith("ns_t_run") or t.startswith("ns_tib")]
        for scale, y in (("uv", mf[comp].to_numpy(float)), ("z", z.to_numpy(float))):
            ff = mf.copy()
            ff["erp_y"] = y
            mdef = replace(md.model_def, family="lmm", outcome="erp_y",
                           random=RandomDef(corr=("1", "a_wb"), uncorrelated=("t_run", "tib")),
                           targets={}, grp_targets={}, sesoi={})
            ma = replace(md, frame=ff, model_def=mdef, outcome_column="erp_y", fixed_terms=keep_terms)
            out[f"a_alpha_to_{comp}_{scale}"] = fit_targets(r, ma, f"erp_a_{comp}_{scale}", H2T(), cr2=True)
        # (b) evoked response -> RT, and the direct alpha effect
        g = mf["grp"].to_numpy(float)
        mb = add_cols(md, {f"{comp}_z": z.to_numpy(float), f"{comp}_z_x_grp": z.to_numpy(float) * g})
        out[f"b_{comp}_to_rt"] = fit_targets(r, mb, f"erp_b_{comp}", {**by_group(f"{comp}_z", f"{comp}_z_x_grp"),
                                                                          **H2T()}, state=st)
    # the alpha slope on the same rows without the evoked response (reference for the direct effect)
    out["c_total_same_rows"] = fit_targets(r, md, "erp_c_total", H2T(), state=st)
    # (c) indirect effects (Monte Carlo)
    rng = np.random.default_rng(20261101)
    out["indirect"] = {}
    for comp in ("p1", "n1"):
        ta = out[f"a_alpha_to_{comp}_z"].get("targets") or {}
        tb = out[f"b_{comp}_to_rt"].get("targets") or {}
        for lab in ("TD", "ASD"):
            A, Bb = ta.get(f"a_wb|{lab}"), tb.get(f"{comp}_z|{lab}")
            if not A or not Bb:
                continue
            da = rng.normal(A["estimate"], A["se"], 20000)
            db = rng.normal(Bb["estimate"], Bb["se"], 20000)
            ab = da * db
            out["indirect"][f"{comp}|{lab}"] = {"a": A["estimate"], "b": Bb["estimate"],
                                                "ab": A["estimate"] * Bb["estimate"],
                                                "ci95_mc": [float(np.percentile(ab, 2.5)),
                                                            float(np.percentile(ab, 97.5))]}
    out["units"] = ("a: uV or within-child SD of the evoked response per SD_i of alpha; b: log RT per within-child SD "
                    "of the evoked response; indirect: log RT per SD_i of alpha")
    r.write("evoked_response_pathway", out)
    return out


def paths_without_baseline(r: Run, erp_table) -> dict:
    """Path (a) with the evoked-response amplitudes without baseline correction (the -100 to 0 ms baseline overlaps
    the alpha window by 50 ms; baseline-shift artefacts would show here)."""
    erp = pd.read_parquet(erp_table)
    erp = erp[erp["erp_valid"]].copy()
    erp["pid"] = erp["subject"].map(r.provider.ps)
    md0 = r.provider.model("M3-H2")
    f = md0.frame.merge(erp[["pid", "run", "sample", "p1_nobl", "n1_nobl"]], on=["pid", "run", "sample"],
                        how="left", validate="one_to_one")
    ok = np.isfinite(f["p1_nobl"].to_numpy(float)) & np.isfinite(f["n1_nobl"].to_numpy(float))
    md = SENS.refinalise(SENS._replace(md0, frame=f.loc[ok].reset_index(drop=True)),
                         reason="valid evoked response (no baseline)")
    mf = md.frame
    keep_terms = [t for t in md.fixed_terms if t in ("a_wb", "a_slow", "cat", "half_status", "heog_z", "vfp_z",
                                                      "heog_missing", "vfp_missing", "age_c", "sex_c", "grp",
                                                      "a_wb:grp")
                  or t.startswith("ns_t_run") or t.startswith("ns_tib")]
    out: dict = {"rows": int(len(mf))}
    for comp in ("p1_nobl", "n1_nobl"):
        z = (mf[comp] - mf.groupby("pid")[comp].transform("mean")) / mf.groupby("pid")[comp].transform("std")
        ff = mf.copy()
        ff["erp_y"] = z.to_numpy(float)
        mdef = replace(md.model_def, family="lmm", outcome="erp_y",
                       random=RandomDef(corr=("1", "a_wb"), uncorrelated=("t_run", "tib")),
                       targets={}, grp_targets={}, sesoi={})
        ma = replace(md, frame=ff, model_def=mdef, outcome_column="erp_y", fixed_terms=keep_terms)
        out[f"a_alpha_to_{comp}_z"] = fit_targets(r, ma, f"erp_a_{comp}_z", H2T(), cr2=True)
    r.write("evoked_response_pathway_no_baseline", out)
    return out
