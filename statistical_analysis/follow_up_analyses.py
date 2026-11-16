"""Follow-up analyses of the FAST group difference (Additional file 1, Supplementary Methods S4, R1-R9).

Model fits use the model engine (lme4 / glmmTMB through :mod:`modeling.engine`) with the final random-effects
structure of the task's model unless stated. Group code: TD -0.5, ASD +0.5, so ``a_wb:grp`` = ASD - TD. Every
function takes the analysis context :class:`statistical_analysis.group_difference_checks.Run` and returns a pooled result dict.
Child keys of the model data are keyed identifiers (:mod:`utils.child_ids`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from modeling import sensitivity as SENS
from modeling.model_specs import RandomDef
from modeling.r_bridge import EngineError
from statistical_analysis import coupling_analyses as ST
from statistical_analysis.group_difference_checks import Run, _counts, _t

SEED = 20261101
PERMUTATIONS = 10000          # child-level permutation of group labels
BOOTSTRAP = 2000              # child-level bootstrap
#: task -> group model and the analysis whose final random-effects structure it uses
TASKS = {"FAST": {"model": "M3-H2", "state": "S2"},
         "AVSRT": {"model": "M1-H2", "state": "P1"},
         "IC": {"model": "M4-H2", "state": "S3"}}
AGES = (9, 11, 13)
NNT = {"<=2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "6": 6.0, "7": 7.0, ">=8": 8.0}
RT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
ALPHA_BINS = 3                # within-child terciles of a_wb
MEASURES = ("mabc_total_ss", "cpt_response_style")   # participants.tsv; *_dup used where the original is missing
GROUP_NORMALISATION = {"ASD SIBLING": "SIB", "NAN": "n/a"}
GT = {"TD": -0.5, "ASD": 0.5}


# helpers --------------------------------------------------------------------------------------------------------------
def read_group_labels(participants_tsv) -> pd.DataFrame:
    """``participant_id`` and normalised ``group`` (``ASD SIBLING`` -> ``SIB``) from participants.tsv."""
    df = pd.read_csv(participants_tsv, sep="\t", dtype=str, usecols=["participant_id", "group"],
                     keep_default_na=False, encoding="utf-8-sig")
    df["group"] = df["group"].str.strip().replace(GROUP_NORMALISATION)
    return df


def add_cols(md, cols: dict):
    """Add plain columns as fixed terms (no ':' in names, so R cannot reorder them)."""
    return SENS._add(md, cols, list(cols))


def by_group(term: str, gx: str) -> dict:
    """Group-specific values of ``term`` whose group interaction is the product column ``gx``."""
    return {f"{term}|TD": {term: 1.0, gx: -0.5}, f"{term}|ASD": {term: 1.0, gx: 0.5}, f"{term}|ASD-TD": {gx: 1.0}}


def fit_targets(r: Run, md, label: str, targets: dict, *, state=None, cr2=True, subdir="improve", **kw) -> dict:
    try:
        res = r.fit(md, label, reduction=state is None, fixed_state=state, cr2=cr2, extra_targets=targets,
                    subdir=subdir, **kw)
    except EngineError as e:
        return {"estimable": False, "error": str(e)[-600:]}
    out = {"estimable": True, **_counts(md), "flag": res.get("flag"),
           "final_structure": (res.get("final_state") or {}).get("re"),
           "targets": {k: _t(res, k) for k in targets}}
    tau = (ST._core(res) or {}).get("a_wb_slope_sd")
    if tau is not None:
        out["tau_a_wb"] = tau
    return out


def H2T() -> dict:
    return {"H2": {"a_wb:grp": 1.0}, "a_wb|TD": {"a_wb": 1.0, "a_wb:grp": -0.5},
            "a_wb|ASD": {"a_wb": 1.0, "a_wb:grp": 0.5}}


def child_slopes(f: pd.DataFrame, y: str = "lrt", x: str = "a_wb", block: str = "pid_run_block") -> pd.DataFrame:
    """Per-child OLS slope of y on x after demeaning both within block; with its sampling variance."""
    d = f[["pid", block, x, y]].dropna().copy()
    d["xc"] = d[x] - d.groupby(block)[x].transform("mean")
    d["yc"] = d[y] - d.groupby(block)[y].transform("mean")
    rows = []
    for pid, g in d.groupby("pid"):
        sxx = float((g["xc"] ** 2).sum())
        if sxx <= 0 or len(g) < 10:
            continue
        b = float((g["xc"] * g["yc"]).sum() / sxx)
        nb = g[block].nunique()
        resid = g["yc"] - b * g["xc"]
        dof = max(len(g) - nb - 1, 1)
        rows.append({"pid": pid, "slope": b, "var": float((resid ** 2).sum() / dof / sxx), "n": len(g)})
    return pd.DataFrame(rows).set_index("pid")


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """b - a (ASD - TD) with Welch t, Hedges g, AUC (P(b > a))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(b, a, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    g = (b.mean() - a.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = float(t.df) if hasattr(t, "df") else len(a) + len(b) - 2
    q = stats.t.ppf(0.975, df)
    u = stats.mannwhitneyu(b, a)
    return {"n_TD": int(len(a)), "n_ASD": int(len(b)), "mean_TD": float(a.mean()), "mean_ASD": float(b.mean()),
            "diff": float(b.mean() - a.mean()), "ci95": [float(b.mean() - a.mean() - q * se),
                                                         float(b.mean() - a.mean() + q * se)],
            "welch_t": float(t.statistic), "welch_df": df, "welch_p": float(t.pvalue), "hedges_g": float(g),
            "auc_ASD_gt_TD": float(u.statistic / (len(a) * len(b))), "mann_whitney_p": float(u.pvalue)}


def child_boot(values_by_child: pd.DataFrame, fn, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kids = values_by_child.index.to_numpy()
    out = []
    for _ in range(B):
        out.append(fn(values_by_child.loc[rng.choice(kids, len(kids), replace=True)]))
    return np.asarray(out, float)


def nnt_numeric(f: pd.DataFrame) -> np.ndarray:
    return f["nnt_level"].astype(str).map(NNT).astype(float).to_numpy()


# two-stage slopes, permutation, leave-one-out -------------------------------------------------------------------------
def two_stage_permutation_leave_one_out(r: Run, predictor_table, trial_table, participants_tsv) -> dict:
    """Two-stage per-child slopes from the FAST predictor (feature) table and trial table; Welch test, Hedges g, AUC;
    10,000-permutation test of group labels; leave-one-child-out refits of M3-H2 (R1)."""
    md = r.provider.model("M3-H2")
    kids = set(md.frame["pid"])
    pred = pd.read_parquet(predictor_table,
                           columns=["subject", "run", "sample", "a_wb", "block", "in_predictor_set", "valid"])
    trials = pd.read_parquet(trial_table,
                             columns=["subject", "run", "sample", "is_target", "responded", "rt_ms"])
    g = read_group_labels(participants_tsv)
    grp = {r.provider.ps(s): lab for s, lab in zip(g["participant_id"], g["group"])}
    del g
    d = pred.merge(trials, on=["subject", "run", "sample"], how="inner", validate="one_to_one")
    d["pid"] = d["subject"].map(r.provider.ps)
    d = d.drop(columns=["subject"])
    d = d[d["pid"].isin(kids) & d["in_predictor_set"].astype(bool) & d["valid"].astype(bool)
          & d["is_target"].astype(bool) & d["responded"].astype(bool)]
    d = d[(d["rt_ms"] >= 150) & (d["rt_ms"] <= 1000) & np.isfinite(d["a_wb"])]
    d["lrt"] = np.log(d["rt_ms"] / 1000.0)
    d["blk"] = d["pid"] + "|" + d["run"].astype(str) + "|" + d["block"].astype(str)
    cs = child_slopes(d, block="blk")
    cs["grp"] = cs.index.map(grp)
    cs = cs[cs["grp"].isin(["TD", "ASD"])]
    a, b = cs.loc[cs["grp"] == "TD", "slope"].to_numpy(), cs.loc[cs["grp"] == "ASD", "slope"].to_numpy()
    out: dict = {"two_stage": {"rows": int(len(d)), "rows_model_frame": int(len(md.frame)), **welch(a, b),
                               "definition": "per child OLS slope of log RT on a_wb, both demeaned within child x "
                                             "run x block, FAST target hits 150-1000 ms with a valid predictor window; "
                                             "built from the feature and trial tables, not the model-data builder"}}
    # one-sample tests per group
    for lab, v in (("TD", a), ("ASD", b)):
        tt = stats.ttest_1samp(v, 0.0)
        out["two_stage"][f"mean_{lab}_vs_0_p"] = float(tt.pvalue)
    # child-level permutation of group labels
    rng = np.random.default_rng(SEED)
    allv = np.concatenate([a, b])
    obs = b.mean() - a.mean()
    nperm = PERMUTATIONS
    null = np.empty(nperm)
    for i in range(nperm):
        p = rng.permutation(allv)
        null[i] = p[len(a):].mean() - p[:len(a)].mean()
    out["permutation"] = {"n": nperm, "observed": float(obs),
                          "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + nperm)), "seed": SEED}
    # leave-one-child-out refits of M3-H2
    st = r.state("S2")
    pids = sorted(kids)

    def loo(pid):
        sub = SENS.refinalise(SENS._replace(md, frame=md.frame[md.frame["pid"] != pid].reset_index(drop=True)),
                              reason="leave-one-child-out")
        res = r.fit(sub, f"loo_{pids.index(pid):03d}", reduction=False, fixed_state=st, cr2=False,
                    subdir="improve_loo")
        return res["targets"]["H2"]["estimate"], res["targets"]["H2"]["p"]

    with ThreadPoolExecutor(r.n_jobs) as ex:
        est = list(ex.map(loo, pids))
    e = np.array([x[0] for x in est])
    pv = np.array([x[1] for x in est])
    full = ((r.c.r("H2") or {}).get("M3-H2") or {}).get("core", {}).get("targets", {}).get("H2", {})
    out["leave_one_out"] = {"n_refits": len(e), "full_estimate": full.get("estimate"),
                            "min": float(e.min()), "max": float(e.max()), "max_p": float(pv.max()),
                            "max_abs_change": float(np.max(np.abs(e - full.get("estimate", np.nan)))),
                            "n_p_above_0.05": int(np.sum(pv >= 0.05))}
    return out


# disjoint trial subsets -----------------------------------------------------------------------------------------------
def _split_model(md, s: np.ndarray, name: str):
    f = md.frame
    a, g = f["a_wb"].to_numpy(float), f["grp"].to_numpy(float)
    cols = {f"s_{name}": s, f"s_{name}_x_grp": s * g, f"awb_x_s_{name}": a * s, f"awb_x_grp_x_s_{name}": a * g * s}
    return add_cols(md, cols)


def disjoint_subsets(r: Run) -> dict:
    """Face vs object blocks; first vs second half of each child's FAST trials: joint split x group x alpha models and
    separate subset fits (R2)."""
    st = r.state("S2")
    md = r.provider.model("M3-H2")
    f = md.frame
    out: dict = {}
    # (a) category: cat is +-0.5; the label of +0.5 from the 'category' column
    lab_pos = f.loc[f["cat"] > 0, "category"].astype(str).mode().iloc[0]
    lab_neg = f.loc[f["cat"] < 0, "category"].astype(str).mode().iloc[0]
    s_cat = f["cat"].to_numpy(float)
    order = f.sort_values(["pid", "run", "sample"]).groupby("pid").cumcount()
    n = f.groupby("pid")["pid"].transform("size")
    s_half = np.where(order.reindex(f.index).to_numpy() < n.to_numpy() / 2.0, -0.5, 0.5)
    for name, s, labs in (("category", s_cat, {-0.5: lab_neg, 0.5: lab_pos}),
                          ("session_half", s_half, {-0.5: "first half", 0.5: "second half"})):
        m = _split_model(md, s, name) if name != "category" else add_cols(md, {
            "cat_x_grp": s * f["grp"].to_numpy(float), "awb_x_cat": f["a_wb"].to_numpy(float) * s,
            "awb_x_grp_x_cat": f["a_wb"].to_numpy(float) * f["grp"].to_numpy(float) * s})
        gx = "awb_x_grp_x_s_session_half" if name != "category" else "awb_x_grp_x_cat"
        sx = "awb_x_s_session_half" if name != "category" else "awb_x_cat"
        tg = {}
        for sv, lab in labs.items():
            tg[f"H2|{lab}"] = {"a_wb:grp": 1.0, gx: sv}
            tg[f"a_wb|TD|{lab}"] = {"a_wb": 1.0, "a_wb:grp": -0.5, sx: sv, gx: -0.5 * sv}
            tg[f"a_wb|ASD|{lab}"] = {"a_wb": 1.0, "a_wb:grp": 0.5, sx: sv, gx: 0.5 * sv}
        tg["H2_difference_between_splits"] = {gx: 1.0}
        out[name] = fit_targets(r, m, f"split_{name}", tg, state=st)
        out[name]["levels"] = {str(k): v for k, v in labs.items()}
        # separate subset fits for display
        sub = {}
        for sv, lab in labs.items():
            keep = s == sv
            mm = SENS.refinalise(SENS._replace(md, frame=f.loc[keep].reset_index(drop=True)),
                                 reason=f"subset {name}={lab}")
            if name == "category":
                mm = SENS._drop_terms(mm, ["cat"])
            sub[lab] = fit_targets(r, mm, f"split_{name}_{'pos' if sv > 0 else 'neg'}", H2T(), state=st)
        out[name]["subset_fits"] = sub
    return out


# siblings -------------------------------------------------------------------------------------------------------------
def siblings(r: Run, participants_tsv) -> dict:
    """Unaffected siblings: three-group models (TD reference) in each task (R3)."""
    g = read_group_labels(participants_tsv)
    sib = {r.provider.ps(s) for s, lab in zip(g["participant_id"], g["group"]) if lab == "SIB"}
    del g
    out: dict = {"caveat": "SIB = unaffected siblings, the cohort used to develop and check the processing; "
                           "descriptive"}
    for task, model, base in (("FAST", "M3", "S2"), ("AVSRT", "M1", "P1"), ("IC", "M4", "S3")):
        st = r.state(base)
        conf = r.provider.model(model).frame
        dev = r.provider.build(model, cohort="dev").frame
        dev = dev[dev["pid"].isin(sib)].copy()
        dev["grp"] = 0.0
        cols = sorted(set(conf.columns) & set(dev.columns))
        f = pd.concat([conf[cols], dev[cols]], ignore_index=True)
        base_md = r.provider.model(model)
        terms = [t for t in base_md.fixed_terms if "grp" not in t.split(":")]
        md = SENS._replace(base_md, frame=f, fixed=terms)
        md = replace(md, has_grp=False)
        md = SENS.refinalise(md, reason="three groups")
        md = add_cols(md, {"is_asd": (md.frame["grp"] > 0).astype(float).to_numpy(),
                           "is_sib": md.frame["pid"].isin(sib).astype(float).to_numpy()})
        mf = md.frame
        md = add_cols(md, {"awb_x_asd": mf["a_wb"].to_numpy(float) * mf["is_asd"].to_numpy(float),
                           "awb_x_sib": mf["a_wb"].to_numpy(float) * mf["is_sib"].to_numpy(float)})
        tg = {"slope|TD": {"a_wb": 1.0}, "slope|ASD": {"a_wb": 1.0, "awb_x_asd": 1.0},
              "slope|SIB": {"a_wb": 1.0, "awb_x_sib": 1.0}, "ASD-TD": {"awb_x_asd": 1.0},
              "SIB-TD": {"awb_x_sib": 1.0}, "SIB-ASD": {"awb_x_sib": 1.0, "awb_x_asd": -1.0}}
        res = fit_targets(r, md, f"siblings_{task}", tg, state=st)
        res["n_children"] = {"TD": int(mf.loc[(mf["grp"] < 0), "pid"].nunique()),
                             "ASD": int(mf.loc[mf["is_asd"] > 0, "pid"].nunique()),
                             "SIB": int(mf.loc[mf["is_sib"] > 0, "pid"].nunique())}
        res["note"] = "task model without the group moderators; TD reference; final random-effects structure of the " \
                      "task model"
        out[task] = res
    return out


# age ------------------------------------------------------------------------------------------------------------------
def age_moderation(r: Run) -> dict:
    """a_wb x age x group in each task; slopes at 9, 11 and 13 years (R4)."""
    out: dict = {}
    for task, cfg in TASKS.items():
        md = r.provider.model(cfg["model"])
        f = md.frame
        m0 = float(((md.meta or {}).get("centring") or {}).get("age_mean", np.nan))
        a, g, ag = f["a_wb"].to_numpy(float), f["grp"].to_numpy(float), f["age_c"].to_numpy(float)
        m = add_cols(md, {"age_x_grp": ag * g, "awb_x_age_x_grp": a * ag * g})
        tg = {"age_slope_of_coupling|TD": {"a_wb:age_c": 1.0, "awb_x_age_x_grp": -0.5},
              "age_slope_of_coupling|ASD": {"a_wb:age_c": 1.0, "awb_x_age_x_grp": 0.5},
              "age_slope_of_coupling|ASD-TD": {"awb_x_age_x_grp": 1.0}}
        for age in AGES:
            c = float(age) - m0
            tg[f"a_wb|TD|age{age}"] = {"a_wb": 1.0, "a_wb:grp": -0.5, "a_wb:age_c": c, "awb_x_age_x_grp": -0.5 * c}
            tg[f"a_wb|ASD|age{age}"] = {"a_wb": 1.0, "a_wb:grp": 0.5, "a_wb:age_c": c, "awb_x_age_x_grp": 0.5 * c}
            tg[f"H2|age{age}"] = {"a_wb:grp": 1.0, "awb_x_age_x_grp": c}
        out[task] = fit_targets(r, m, f"age_{task}", tg, state=r.state(cfg["state"]))
        out[task]["age_mean"] = m0
        out[task]["units"] = "log RT per SD_i per year of age (age slopes); log RT per SD_i (slopes at ages)"
    return out


# temporal expectation -------------------------------------------------------------------------------------------------
def temporal_expectation(r: Run) -> dict:
    """Temporal expectation in FAST: RT per additional non-target, alpha per additional non-target, coupling x
    non-target count, by group (R5)."""
    st = r.state("S2")
    out: dict = {}
    md = r.provider.model("M3-H2")
    f = md.frame
    nn = nnt_numeric(f)
    c0 = float(np.mean(nn))
    nc = nn - c0
    g, a = f["grp"].to_numpy(float), f["a_wb"].to_numpy(float)
    # (i) RT hazard: replace the 7-level nnt factor by a linear trend and its group interaction
    m = SENS._drop_terms(md, [t for t in md.fixed_terms if t.startswith("nnt")])
    m = add_cols(m, {"nnt_c": nc, "nnt_x_grp": nc * g})
    out["i_rt_hazard"] = fit_targets(r, m, "sequence_rt_hazard", by_group("nnt_c", "nnt_x_grp"), state=st)
    out["i_rt_hazard"]["units"] = "log RT per additional non-target since the previous target (linear, 2..8)"
    # (iii) coupling x hazard (nnt factor kept)
    m = add_cols(md, {"awb_x_nnt": a * nc, "awb_x_nnt_x_grp": a * nc * g})
    out["iii_coupling_x_hazard"] = fit_targets(r, m, "sequence_coupling_x_hazard",
                                               {**by_group("awb_x_nnt", "awb_x_nnt_x_grp"), **H2T()}, state=st)
    # (ii) alpha ~ hazard on the predictor set (hits and misses), outcome = a_wb
    pm = r.provider.miss_model("M3")
    pf = pm.frame.copy()
    nn2 = nnt_numeric(pf) - c0
    pf["alpha_y"] = pf["a_wb"].to_numpy(float)
    pf["nnt_c"] = nn2
    pf["nnt_x_grp"] = nn2 * pf["grp"].to_numpy(float)
    keep_terms = ["nnt_c", "nnt_x_grp", "grp", "cat", "half_status", "age_c", "sex_c"] + \
        [t for t in pm.fixed_terms if t.startswith("ns_t_run") or t.startswith("ns_tib")]
    mdef = replace(pm.model_def, family="lmm", outcome="alpha_y",
                   random=RandomDef(corr=("1", "nnt_c"), uncorrelated=("t_run", "tib")),
                   targets={}, grp_targets={}, sesoi={})
    ma = replace(pm, frame=pf, model_def=mdef, outcome_column="alpha_y", fixed_terms=keep_terms)
    out["ii_alpha_x_hazard"] = fit_targets(r, ma, "sequence_alpha_hazard", by_group("nnt_c", "nnt_x_grp"))
    out["ii_alpha_x_hazard"]["units"] = "SD_i of a_wb per additional non-target since the previous target"
    out["nnt_centre"] = c0
    return out


# RT quantiles and misses ----------------------------------------------------------------------------------------------
def rt_quantiles_and_misses(r: Run) -> dict:
    """Alpha effect at RT quantiles by group (two-stage, highest minus lowest within-child alpha tercile); alpha and
    misses by group (binomial GLMM) (R6)."""
    md = r.provider.model("M3-H2")
    f = md.frame.copy()
    f["y"] = f["lrt"] - f.groupby("pid_run_block")["lrt"].transform("mean")
    k = ALPHA_BINS
    f["bin"] = f.groupby("pid")["a_wb"].transform(lambda s: pd.qcut(s.rank(method="first"), k, labels=False))
    qs = [float(q) for q in RT_QUANTILES]
    rows = []
    for pid, gdf in f.groupby("pid"):
        lo, hi = gdf.loc[gdf["bin"] == 0, "y"], gdf.loc[gdf["bin"] == k - 1, "y"]
        if len(lo) < 10 or len(hi) < 10:
            continue
        rows.append({"pid": pid, "grp": gdf["grp"].iloc[0],
                     **{f"q{int(q * 100)}": float(np.quantile(hi, q) - np.quantile(lo, q)) for q in qs}})
    d = pd.DataFrame(rows).set_index("pid")
    res = {}
    ps = []
    for q in qs:
        col = f"q{int(q * 100)}"
        a, b = d.loc[d["grp"] < 0, col].to_numpy(), d.loc[d["grp"] > 0, col].to_numpy()
        w = welch(a, b)
        for lab, v in (("TD", a), ("ASD", b)):
            tt = stats.ttest_1samp(v, 0.0)
            se = v.std(ddof=1) / np.sqrt(len(v))
            qq = stats.t.ppf(0.975, len(v) - 1)
            w[f"{lab}_mean_ci95"] = [float(v.mean() - qq * se), float(v.mean() + qq * se)]
            w[f"{lab}_vs_0_p"] = float(tt.pvalue)
        res[col] = w
        ps.append(w["welch_p"])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[idx]))
        adj[idx] = run
    for i, q in enumerate(qs):
        res[f"q{int(q * 100)}"]["welch_p_holm5"] = float(adj[i])
    out = {"quantiles": res, "definition": "per child: log RT deviation from the child x run x block mean; quantile "
                                           "of the highest minus the lowest within-child a_wb tercile; group means "
                                           "and Welch tests (ASD - TD); Holm over the five quantiles",
           "n_children": int(len(d))}
    # misses by group (glmmTMB), predictor-set rows
    pm = r.provider.miss_model("M3")
    pf = pm.frame.copy()
    if "responded" in pf.columns:
        pf["miss"] = (~pf["responded"].astype(bool)).astype(float)
    mdef = replace(pm.model_def, family="glmm", outcome="miss", random=RandomDef(corr=("1", "a_wb")),
                   targets={}, grp_targets={}, sesoi={})
    mm = replace(pm, frame=pf, model_def=mdef, outcome_column="miss")
    out["miss_glmm"] = fit_targets(r, mm, "quantile_miss", H2T(), cr2=False)
    out["miss_glmm"]["units"] = "log-OR of a miss per SD_i"
    out["miss_glmm"]["miss_rate"] = float(pf["miss"].mean())
    return out


# slope reliability ----------------------------------------------------------------------------------------------------
def slope_reliability(r: Run) -> dict:
    """Odd / even split-half reliability of per-child coupling slopes (Spearman-Brown), by task and group; implied
    attenuation (R8)."""
    out: dict = {}
    B = BOOTSTRAP
    for task, cfg in TASKS.items():
        f = r.provider.model(cfg["model"]).frame.copy()
        f = f.sort_values(["pid", "run", "sample"])
        f["half"] = f.groupby("pid").cumcount() % 2
        s0 = child_slopes(f[f["half"] == 0])
        s1 = child_slopes(f[f["half"] == 1])
        full = child_slopes(f)
        j = s0[["slope"]].join(s1[["slope"]], lsuffix="_odd", rsuffix="_even", how="inner")
        j["grp"] = f.groupby("pid")["grp"].first().reindex(j.index)

        def sb(dd):
            rr = np.corrcoef(dd["slope_odd"], dd["slope_even"])[0, 1]
            return 2 * rr / (1 + rr) if rr > -1 else np.nan

        rec = {}
        for lab, sel in (("all", j.index == j.index), ("TD", j["grp"] < 0), ("ASD", j["grp"] > 0)):
            dd = j.loc[sel]
            boot = child_boot(dd, sb, B, SEED)
            rec[lab] = {"n_children": int(len(dd)), "r_odd_even": float(np.corrcoef(dd["slope_odd"],
                                                                                    dd["slope_even"])[0, 1]),
                        "spearman_brown": float(sb(dd)),
                        "ci95_boot": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]}
        # signal / noise view from the full-data per-child slopes
        sv = full["var"].median()
        tot = full["slope"].var(ddof=1)
        rec["variance_decomposition"] = {"observed_between_child_var": float(tot),
                                         "median_sampling_var": float(sv),
                                         "implied_true_var": float(max(tot - full["var"].mean(), 0.0)),
                                         "implied_reliability": float(max(tot - full["var"].mean(), 0.0) / tot)}
        out[task] = rec
    out["implication"] = ("an observed correlation of a per-child slope with a symptom score is bounded by "
                          "sqrt(rel_slope x rel_score)")
    return out


# dimensional measures -------------------------------------------------------------------------------------------------
def _participants_measure(participants_tsv, col: str) -> dict:
    p = pd.read_csv(participants_tsv, sep="\t", dtype=str)
    v = pd.to_numeric(p[col], errors="coerce")
    d = pd.to_numeric(p.get(f"{col}_dup"), errors="coerce") if f"{col}_dup" in p else None
    if d is not None:
        v = v.where(v.notna(), d)
    return {r.provider.ps(s): float(x) for s, x in zip(p["participant_id"], v) if np.isfinite(x)}


def dimensional_measures(r: Run, participants_tsv) -> dict:
    """Coupling x CPT-3 response style and x MABC-2 total standard score in FAST, adjusted for group (R9)."""
    st = r.state("S2")
    md0 = r.provider.model("M3-H2")
    out: dict = {}
    for col in MEASURES:
        vals = _participants_measure(participants_tsv, col)
        f = md0.frame
        v = f["pid"].map(vals).astype(float)
        ok = np.isfinite(v.to_numpy(float))
        md = SENS.refinalise(SENS._replace(md0, frame=f.loc[ok].reset_index(drop=True)), reason=f"measure {col}")
        mf = md.frame
        kids = mf.groupby("pid")["pid"].first().index
        kv = np.array([vals[k] for k in kids])
        cen, sd = float(kv.mean()), float(kv.std(ddof=1))
        x = (mf["pid"].map(vals).astype(float).to_numpy() - cen) / sd
        a, g = mf["a_wb"].to_numpy(float), mf["grp"].to_numpy(float)
        m = add_cols(md, {"meas_z": x, "awb_x_meas": a * x, "awb_x_meas_x_grp": a * x * g})
        tg = {**by_group("awb_x_meas", "awb_x_meas_x_grp"), "awb_x_meas|pooled": {"awb_x_meas": 1.0}, **H2T()}
        res = fit_targets(r, m, f"dimensional_{col}", tg, state=st)
        grp = mf.groupby("pid")["grp"].first()
        res["n_children"] = {"TD": int((grp < 0).sum()), "ASD": int((grp > 0).sum())}
        res["measure"] = {"centre": cen, "sd": sd,
                          "TD_mean": float(np.mean([vals[k] for k in grp.index[grp < 0]])),
                          "ASD_mean": float(np.mean([vals[k] for k in grp.index[grp > 0]]))}
        res["units"] = "log RT per SD_i per SD of the measure (pooled over the children with the measure)"
        out[col] = res
    return out


def without_fallback_iaf(r: Run, predictor_table) -> dict:
    """M3-H2 without the children whose IAF came from a fallback source (R1b)."""
    pred = pd.read_parquet(predictor_table,
                           columns=["subject", "iaf_source"]).drop_duplicates("subject")
    src = {r.provider.ps(s): v for s, v in zip(pred["subject"], pred["iaf_source"])}
    md = r.provider.model("M3-H2")
    f = md.frame
    keep = f["pid"].map(src).astype(str).eq("restingIAF_PAF").to_numpy()
    kids = f.groupby("pid")["grp"].first()
    fb = kids.index.map(src).astype(str) != "restingIAF_PAF"
    out = {"fallback_children_excluded": {"TD": int(((kids < 0) & fb).sum()), "ASD": int(((kids > 0) & fb).sum())}}
    sub = SENS.refinalise(SENS._replace(md, frame=f.loc[keep].reset_index(drop=True)), reason="restingIAF PAF only")
    out["fit"] = fit_targets(r, sub, "paf_iaf_only", H2T(), state=r.state("S2"))
    return out


def cross_child_pairing_null(r: Run) -> dict:
    """Cross-child pairing null (shared temporal structure) for the two-stage slope (R1c)."""
    f = r.provider.model("M3-H2").frame.copy().sort_values(["pid", "run", "sample"])
    f["xc"] = f["a_wb"] - f.groupby("pid_run_block")["a_wb"].transform("mean")
    f["yc"] = f["lrt"] - f.groupby("pid_run_block")["lrt"].transform("mean")
    seqs = {pid: (g["xc"].to_numpy(float), g["yc"].to_numpy(float), float(g["grp"].iloc[0]))
            for pid, g in f.groupby("pid")}

    def slope(x, y):
        n = min(len(x), len(y))
        x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
        return float((x * y).sum() / (x * x).sum())

    obs = {lab: np.mean([slope(x, y) for x, y, g in seqs.values() if np.sign(g) == np.sign(s)]) for lab, s in GT.items()}
    rng = np.random.default_rng(SEED)
    null = {"TD": [], "ASD": [], "diff": []}
    for _ in range(1000):
        m = {}
        for lab, sgn in GT.items():
            ids = [k for k, v in seqs.items() if np.sign(v[2]) == np.sign(sgn)]
            while True:
                perm = rng.permutation(len(ids))
                if not np.any(perm == np.arange(len(ids))):
                    break
            m[lab] = np.mean([slope(seqs[ids[i]][0], seqs[ids[j]][1]) for i, j in zip(range(len(ids)), perm)])
            null[lab].append(m[lab])
        null["diff"].append(m["ASD"] - m["TD"])
    out = {"observed": {"TD": obs["TD"], "ASD": obs["ASD"], "diff": obs["ASD"] - obs["TD"]}}
    for k, v in null.items():
        v = np.asarray(v)
        out[f"null_{k}"] = {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)),
                           "p97.5": float(np.percentile(v, 97.5))}
    od = out["observed"]["diff"]
    out["p_diff_vs_null"] = float((1 + np.sum(np.abs(np.asarray(null["diff"])) >= abs(od))) / 1001)
    out["p_TD_vs_null"] = float((1 + np.sum(np.asarray(null["TD"]) >= out["observed"]["TD"])) / 1001)
    out["definition"] = ("child i's block-demeaned a_wb (trial order) paired with child j's block-demeaned log RT over "
                         "common trial positions, j a random derangement within group; 1,000 draws")
    return out
