"""Channel-level alpha-RT coupling with cluster-based permutation tests (Additional file 1, Supplementary Methods
S4, R13): two-stage per-child slopes at each of the 64 electrodes, cluster-based permutation over neighbouring
electrodes (biosemi64 adjacency; 5,000 permutations of group labels or sign flips; cluster-forming threshold
two-sided P < 0.05; seed 20261101), alpha-power topographies, posterior-ROI spectra (absolute and IAF-aligned) and
FAST group evoked-response averages. Input: the channel-level alpha extracted by
:mod:`statistical_analysis.scalp_coupling_maps`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import connected_components

from statistical_analysis.scalp_coupling_maps import MIN_ROWS, N_PERM, SEED, _model_rows, channels

log = logging.getLogger(__name__)


# two-stage slopes -----------------------------------------------------------------------------------------------------
def child_slopes(df: pd.DataFrame, cols: list[str], standardise: bool = True) -> pd.DataFrame:
    """Per child and column: OLS slope of log RT on the column, both demeaned within child x block over the rows
    where the column is finite, x 100; times the child's SD of the demeaned column when ``standardise``."""
    out = {}
    for pid, g in df.groupby("pid", sort=True):
        blk = pd.factorize(g["pid_run_block"])[0]
        nb = blk.max() + 1
        y = g["lrt"].to_numpy(float)
        X = g[cols].to_numpy(float)
        M = np.isfinite(X)
        Xz = np.where(M, X, 0.0)
        cnt = np.zeros((nb, X.shape[1]))
        np.add.at(cnt, blk, M)
        sx = np.zeros((nb, X.shape[1]))
        np.add.at(sx, blk, Xz)
        sy = np.zeros((nb, X.shape[1]))
        np.add.at(sy, blk, M * y[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            xd = np.where(M, X - (sx / cnt)[blk], np.nan)
            yd = np.where(M, y[:, None] - (sy / cnt)[blk], np.nan)
            n = M.sum(axis=0)
            b = np.nansum(xd * yd, axis=0) / np.nansum(xd * xd, axis=0)
            sd = np.nanstd(xd, axis=0, ddof=1)
        s = 100.0 * b * (sd if standardise else 1.0)
        s[n < MIN_ROWS] = np.nan
        out[pid] = s
    return pd.DataFrame.from_dict(out, orient="index", columns=cols)


# statistics -----------------------------------------------------------------------------------------------------------
def _mv(S: np.ndarray):
    n = np.isfinite(S).sum(axis=0)
    m = np.nanmean(S, axis=0)
    v = np.nanvar(S, axis=0, ddof=1)
    return n, m, v


def welch_t(S: np.ndarray, asd: np.ndarray) -> np.ndarray:
    na, ma, va = _mv(S[~asd])
    nb, mb, vb = _mv(S[asd])
    return (mb - ma) / np.sqrt(va / na + vb / nb)


def one_t(S: np.ndarray) -> np.ndarray:
    n, m, v = _mv(S)
    return m / np.sqrt(v / n)


def adjacency(chans: list[str]):
    import mne

    info = mne.create_info(chans, 512.0, "eeg")
    info.set_montage("biosemi64")
    adj, names = mne.channels.find_ch_adjacency(info, "eeg")
    if list(names) != chans:
        raise ValueError("adjacency channel order differs")
    return adj.tocsr()


def find_clusters(t: np.ndarray, thr: float, adj) -> list[dict]:
    out = []
    for sign in (1, -1):
        sup = np.flatnonzero(np.nan_to_num(sign * t) > thr)
        if not len(sup):
            continue
        k, lab = connected_components(adj[sup][:, sup], directed=False)
        for c in range(k):
            idx = sup[lab == c]
            out.append({"sign": sign, "idx": idx, "mass": float(np.nansum(t[idx]))})
    return out


def cluster_test(S: np.ndarray, adj, kind: str, asd: np.ndarray | None, chans: list[str], rng) -> dict:
    if kind == "diff":
        n = int((~asd).sum() + asd.sum())
        thr = float(stats.t.ppf(0.975, n - 2))
        t_obs = welch_t(S, asd)
    else:
        n = S.shape[0]
        thr = float(stats.t.ppf(0.975, n - 1))
        t_obs = one_t(S)
    obs = find_clusters(t_obs, thr, adj)
    null = np.zeros(N_PERM)
    for i in range(N_PERM):
        if kind == "diff":
            tt = welch_t(S, rng.permutation(asd))
        else:
            tt = one_t(S * rng.choice([-1.0, 1.0], size=S.shape[0])[:, None])
        cl = find_clusters(tt, thr, adj)
        null[i] = max((abs(c["mass"]) for c in cl), default=0.0)
    clusters = [{"sign": int(c["sign"]), "channels": [chans[j] for j in c["idx"]], "mass": c["mass"],
                 "p": float(np.mean(null >= abs(c["mass"])))} for c in sorted(obs, key=lambda c: -abs(c["mass"]))]
    return {"threshold_t": thr, "t": t_obs.tolist(), "clusters": clusters, "n_perm": N_PERM,
            "null_max_mass_q95": float(np.quantile(null, 0.95))}


def summary(S: np.ndarray) -> dict:
    n, m, v = _mv(S)
    se = np.sqrt(v / n)
    q = stats.t.ppf(0.975, np.maximum(n - 1, 1))
    return {"n": n.tolist(), "mean": m.tolist(), "ci_lo": (m - q * se).tolist(), "ci_hi": (m + q * se).tolist()}


# spectra --------------------------------------------------------------------------------------------------------------
def spectra(r, task: str, m: pd.DataFrame, features_root) -> dict:
    """Prestimulus posterior-ROI spectra by group, on the absolute axis and aligned to each child's IAF."""
    sub = m.groupby("subject").agg(pid=("pid", "first"), grp=("grp", "first"), iaf=("iaf_hz", "first"))
    rel = np.round(np.arange(-6.0, 10.0001, 0.25), 2)
    rows, rrows = {}, {}
    for subj, row in sub.iterrows():
        parts = []
        for p in sorted((Path(features_root) / task / subj).glob(
                "*_features_spectra_predictor.parquet")):
            s = pd.read_parquet(p)
            parts.append(s[s["kind"] == "pre_predictor_set"])
        if not parts:
            continue
        s = pd.concat(parts)
        w = s.groupby("freq_hz").apply(lambda g: np.average(g["mean_log10_psd"], weights=g["n_windows"]),
                                       include_groups=False)
        rows[subj] = w
        rrows[subj] = np.interp(row["iaf"] + rel, w.index.to_numpy(float), w.to_numpy(float), left=np.nan, right=np.nan)
    F = pd.DataFrame(rows).T
    R = pd.DataFrame(rrows, index=rel).T
    asd = (sub.loc[F.index, "grp"] > 0).to_numpy()
    out = {"freq_hz": F.columns.to_list(), "rel_hz": rel.tolist(),
           "iaf": {g: summary(sub.loc[F.index, "iaf"].to_numpy(float)[sel][:, None]) for g, sel in
                   (("TD", ~asd), ("ASD", asd))}}
    for g, sel in (("TD", ~asd), ("ASD", asd)):
        out[f"abs_{g}"] = summary(F.to_numpy(float)[sel])
        out[f"rel_{g}"] = summary(R.to_numpy(float)[sel])
        out[f"iaf_values_{g}"] = sorted(sub.loc[F.index, "iaf"].to_numpy(float)[sel].tolist())
    return out


# analysis -------------------------------------------------------------------------------------------------------------
def scalp_coupling(r, tasks: list[str], features_root, work_dir) -> dict:
    """Channel-level coupling, cluster tests, alpha-power maps, spectra and FAST group evoked-response averages;
    writes ``scalp_coupling.json`` and ``fast_erp_group_average.npz`` to ``r.out``."""
    work_dir = Path(work_dir)
    chans = channels()
    adj = adjacency(chans)
    r.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"channels": chans,
                 "units": "coupling: % RT per within-child SD of channel alpha (two-stage, as the ROI two-stage slope)",
                 "seed": SEED, "tasks": {}}
    for task in tasks:
        rng = np.random.default_rng(SEED)
        m = _model_rows(r, task, features_root)
        ch = pd.read_parquet(work_dir / f"{task}_channel_alpha.parquet")
        if not (ch[["pid", "run", "sample"]].reset_index(drop=True).equals(m[["pid", "run", "sample"]]
                                                                           .reset_index(drop=True))):
            raise SystemExit(f"{task}: channel rows are not aligned with the model rows")
        df = pd.concat([m.reset_index(drop=True), ch[chans + ["roi_recomputed"]].reset_index(drop=True)], axis=1)
        grp = df.groupby("pid")["grp"].first()
        S = child_slopes(df, chans)
        asd = (grp.loc[S.index] > 0).to_numpy()
        Sv = S.to_numpy(float)
        roi = child_slopes(df, ["a_wb"], standardise=False)["a_wb"]
        roi_re = child_slopes(df, ["roi_recomputed"])["roi_recomputed"]
        t = {"n_children": {"TD": int((~asd).sum()), "ASD": int(asd.sum())},
             "check": json.loads((work_dir / f"{task}_channel_alpha_check.json").read_text(encoding="utf-8")),
             "TD": summary(Sv[~asd]), "ASD": summary(Sv[asd]),
             "diff": (np.array(summary(Sv[asd])["mean"]) - np.array(summary(Sv[~asd])["mean"])).tolist()}
        ra, rb = roi[~asd].to_numpy(), roi[asd].to_numpy()
        wt = stats.ttest_ind(rb, ra, equal_var=False)
        t["roi_two_stage_a_wb"] = {"TD": float(ra.mean()), "ASD": float(rb.mean()), "diff": float(rb.mean() - ra.mean()),
                                   "welch_p": float(wt.pvalue), "values_TD": sorted(ra.tolist()),
                                   "values_ASD": sorted(rb.tolist())}
        rr = roi_re.to_numpy()
        t["roi_two_stage_recomputed_per_sd"] = {"TD": float(np.nanmean(rr[~asd])), "ASD": float(np.nanmean(rr[asd]))}
        # permutation null of the ROI two-stage difference on the model rows (display)
        obs = rb.mean() - ra.mean()
        allv = roi.to_numpy()
        nullp = np.array([(lambda g: allv[g].mean() - allv[~g].mean())(rng.permutation(asd)) for _ in range(10000)])
        hist, edges = np.histogram(nullp, bins=60)
        t["roi_permutation"] = {"n": 10000, "p_two_sided": float(np.mean(np.abs(nullp) >= abs(obs))),
                                "hist": hist.tolist(), "edges": edges.tolist(), "observed": float(obs)}
        t["clusters"] = {"TD": cluster_test(Sv[~asd], adj, "one", None, chans, rng),
                         "ASD": cluster_test(Sv[asd], adj, "one", None, chans, rng),
                         "diff": cluster_test(Sv, adj, "diff", asd, chans, rng)}
        # alpha-power topography (child means over the model rows)
        P = df.groupby("pid")[chans].mean().loc[S.index].to_numpy(float)
        t["alpha_power"] = {"all": summary(P), "TD": summary(P[~asd]), "ASD": summary(P[asd]),
                            "units": "log10(uV^2/Hz), mean over each child's windows"}
        t["spectra"] = spectra(r, task, m, features_root)
        res["tasks"][task] = t
        log.info("%s: ROI two-stage diff %.3f; clusters %s", task, t["roi_two_stage_a_wb"]["diff"],
                 json.dumps({k: [(c["sign"], len(c["channels"]), round(c["p"], 4)) for c in v["clusters"]]
                             for k, v in t["clusters"].items()}))
    # FAST ERP group averages
    if "FAST" in tasks:
        z = np.load(work_dir / "FAST_erp_sums.npz", allow_pickle=False)
        m = _model_rows(r, "FAST", features_root)
        grp = m.groupby("pid")["grp"].first()
        pids = [str(p) for p in z["pids"]]
        sums, cnt = z["sums"], z["cnt"]                   # (child, tercile, channel, time), (child, tercile, channel)
        with np.errstate(invalid="ignore", divide="ignore"):
            terc = sums / cnt[..., None]
            allc = sums.sum(axis=1) / cnt.sum(axis=1)[..., None]
        asd = np.array([grp.loc[p] > 0 for p in pids])
        erp = {"t_ms": z["t"].tolist(), "baseline_ms": [-50, 0], "n_children": {"TD": int((~asd).sum()),
                                                                                  "ASD": int(asd.sum())},
               "trials_per_child_median": float(np.median(cnt.sum(axis=1).max(axis=1)))}
        arrays = {}
        sites = {"p1": ["O1", "Oz", "O2"], "n1": ["P7", "P8", "PO7", "PO8"]}     # the evoked-response site sets
        for g, sel in (("TD", ~asd), ("ASD", asd)):
            arrays[f"{g}_mean"] = np.nanmean(allc[sel], axis=0)
            arrays[f"{g}_sem"] = np.nanstd(allc[sel], axis=0, ddof=1) / np.sqrt(np.isfinite(allc[sel]).sum(axis=0))
            for k in range(3):
                arrays[f"{g}_t{k + 1}_mean"] = np.nanmean(terc[sel, k], axis=0)
            # site averages within each child first, then mean and SEM across children (adjacent sites correlate)
            for s, names in sites.items():
                idx = [chans.index(c) for c in names]
                for lab, A in (("all", allc[sel]), *((f"t{k + 1}", terc[sel, k]) for k in range(3))):
                    cs = np.nanmean(A[:, idx, :], axis=1)
                    n = np.isfinite(cs).sum(axis=0)
                    arrays[f"{g}_{s}_{lab}_mean"] = np.nanmean(cs, axis=0)
                    arrays[f"{g}_{s}_{lab}_sem"] = np.nanstd(cs, axis=0, ddof=1) / np.sqrt(n)
        np.savez_compressed(r.out / "fast_erp_group_average.npz", t_ms=z["t"], channels=np.array(chans), **arrays)
        res["erp_FAST"] = erp
    r.write("scalp_coupling", res)
    return res
