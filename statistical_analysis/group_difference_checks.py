"""Checks of the FAST group difference in alpha-RT coupling (M3-H2, ``a_wb:grp`` = ASD - TD).

- ``demographics``: sample description of the three group-model samples by group (Table 1);
- ``group_slopes``: TD and ASD slopes of M1-H2, M3-H2 and M4-H2;
- ``interaction``: the joint three-task model with ``a_wb:task:grp`` and the group x task contrasts;
- ``severity``: ADOS-2 CSS and SRS-2 T within the autistic group in FAST;
- ``quality``: child-level data-quality metrics by group in FAST (Additional file 1, Table S11) and M3-H2 with
  alpha x child data-quality terms;
- ``robustness``: the sensitivity analyses that apply to FAST refitted on M3-H2 with the final M3 random-effects
  structure, plus S-HET, S-MATCH and S-REL-H2;
- ``figdata``: pooled alpha-quintile curves (Fig. 2A).

Group code: TD -0.5, ASD +0.5. Results are pooled (no child-level rows).
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from modeling import sensitivity as SENS
from modeling import srel as SREL
from modeling.model_data import combine_m5h
from modeling.r_bridge import EngineError
from statistical_analysis import coupling_analyses as ST
from utils.output import write_json

#: sensitivity analyses that apply to FAST or to the group model, refitted on M3-H2
ROBUST_IDS = ("S-W400", "S-LAG", "S-ERP", "S-HH", "S-PRT", "S-EST1", "S-EST2", "S-BAND", "R-AP", "S-ICA",
              "S-RT-OWN-100-1500", "S-RT-FAST-150-1200", "S-RT-INV", "S-RET", "S-ALT", "S-NOM", "S-MIN", "S-ROI",
              "S-IAF", "S-BLINK-H2", "S-OCU-H2", "S-FSIQ", "S-FSIQ-EXCL", "S-MED", "S-MED2")
GROUP_TARGETS = {"TD": {"a_wb": 1.0, "a_wb:grp": -0.5}, "ASD": {"a_wb": 1.0, "a_wb:grp": 0.5}}
H2_OF = {"M1-H2": ("P1", "AVSRT"), "M3-H2": ("S2", "FAST"), "M4-H2": ("S3", "IC")}


class Run:
    """Analysis context: the model-data provider, the model engine and the stored results of
    :mod:`statistical_analysis.coupling_analyses` (``{key: {"result": dict}}``, keys as in :class:`coupling_analyses.Ctx`)."""

    def __init__(self, provider, engine, results: Mapping, *, out_dir=Path("results"), work_dir=Path("work"),
                 n_jobs: int = 12, log=print):
        self.log, self.n_jobs = log, n_jobs
        self.provider = provider
        self.engine = engine
        self.work = Path(work_dir)
        self.out = Path(out_dir)
        self.results = dict(results)
        self.c = ST.Ctx(provider=self.provider, engine=self.engine, settings=None, results=self.results,
                        n_jobs=n_jobs, log=log)

    def state(self, base: str) -> dict:
        st = ((self.c.r(base) or {}).get("core") or {}).get("final_state")
        if not st:
            raise RuntimeError(f"no final random-effects structure stored for {base}")
        return st

    def fit(self, md, label: str, **kw) -> dict:
        kw.setdefault("second_impl", False)
        return self.engine.fit(md, label, **kw)

    def write(self, name: str, obj: dict) -> Path:
        return write_json(self.out / f"{name}.json", obj)


def _t(res: dict, key: str) -> dict | None:
    row = ST.target_row((res.get("targets") or {}).get(key))
    cr2 = ST.target_row((((res.get("cr2") or {}).get("targets")) or {}).get(key))
    if row is not None and cr2 is not None:
        row = {**row, "cr2_p": cr2.get("p"), "cr2_ci95": cr2.get("ci95"), "cr2_se": cr2.get("se")}
    return row


def _counts(md) -> dict:
    return ST._group_counts(md) | {"rows": int(len(md.frame))}


# group slopes ---------------------------------------------------------------------------------------------------------
def group_slopes(r: Run) -> dict:
    out = {"definition": "TD = a_wb - 0.5 a_wb:grp, ASD = a_wb + 0.5 a_wb:grp, at the centre of the a_wb moderators "
                         "(age, sex, reliability) of the H2 models; final random-effects structure of the task's "
                         "model"}
    for model, (base, task) in H2_OF.items():
        md = r.provider.model(model)
        res = r.fit(md, f"fast_slopes_{model.replace('-', '_')}", reduction=False, fixed_state=r.state(base),
                    cr2=True, extra_targets=GROUP_TARGETS)
        out[task] = {"model": model, **_counts(md), **{k: _t(res, k) for k in ("TD", "ASD", "H2")}}
    return out


# group x task ---------------------------------------------------------------------------------------------------------
def interaction(r: Run) -> dict:
    parts = {"AVSRT": r.provider.model("M1"), "FAST": r.provider.model("M3"), "IC": r.provider.model("M4")}
    m5 = ST.m5h_with_grp(combine_m5h(parts), parts)
    tg = {"GxT_FAST_minus_AVSRT": {"aF:grp": 1.0, "aA:grp": -1.0},
          "GxT_FAST_minus_IC": {"aF:grp": 1.0, "aI:grp": -1.0},
          "GxT_FASTIC_minus_AVSRT": {"aF:grp": 0.5, "aI:grp": 0.5, "aA:grp": -1.0},
          "H2_A": {"aA:grp": 1.0}, "H2_F": {"aF:grp": 1.0}, "H2_I": {"aI:grp": 1.0}}
    res = r.fit(m5, "fast_M5h_GxT", cr2=True, extra_targets=tg, export_fixef_vcov=True)
    out = {"model": "M5h + a_wb:task:grp", **_counts(m5),
           "units": "slope of the task-standardised log RT (y = (lrt - child x task mean) / s_task) per SD_i",
           "s_task": {k: float(v) for k, v in __import__("modeling.model_data",
                                                         fromlist=["S_TASK"]).S_TASK.items()},
           "final_structure": (res.get("final_state") or {}).get("re"),
           "targets": {k: _t(res, k) for k in tg}}
    # 2-df Wald test that the group difference is equal in the three tasks
    names, V = res.get("coef_names"), res.get("vcov")
    if names and V is not None and all(t in names for t in ("aA:grp", "aF:grp", "aI:grp")):
        V = np.asarray(V, float)
        idx = [names.index(t) for t in ("aA:grp", "aF:grp", "aI:grp")]
        b = np.array([out["targets"][k]["estimate"] for k in ("H2_A", "H2_F", "H2_I")])
        L = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        Lb = L @ b
        w = float(Lb @ np.linalg.solve(L @ V[np.ix_(idx, idx)] @ L.T, Lb))
        out["omnibus_equal_group_difference"] = {"chi2": w, "df": 2, "p": float(stats.chi2.sf(w, 2)),
                                                 "definition": "H0: aA:grp = aF:grp = aI:grp (model-based Wald)"}
    return out


# severity -------------------------------------------------------------------------------------------------------------
def severity(r: Run) -> dict:
    out: dict = {"definition": "coupling x severity within the autistic children, FAST (M3); random-effects "
                               "simplification path"}
    base = r.provider.model("M3")
    cov = r.provider.child_covariates()
    f = base.frame
    asd = f[f["grp"] > 0].reset_index(drop=True)
    for key, col in (("ados_css", "ados_css_c"), ("srs2", "srs2_c")):
        vals = cov.get(key) or {}
        v = asd["pid"].map(vals).astype(float)
        sub = asd.loc[np.isfinite(v.to_numpy(float))].reset_index(drop=True)
        if sub["pid"].nunique() < 10:
            out[key] = {"estimable": False, "n_children": int(sub["pid"].nunique())}
            continue
        md = SENS.refinalise(ST._drop_grp(SENS._replace(base, frame=sub)), reason=f"severity {key}")
        kids = md.frame.groupby("pid")["pid"].first().index
        centre = float(np.mean([vals[k] for k in kids]))
        sd = float(np.std([vals[k] for k in kids], ddof=1))
        md = SENS._add(md, {col: md.frame["pid"].map(vals).astype(float).to_numpy() - centre}, [f"a_wb:{col}"])
        tg = {"per_point": {f"a_wb:{col}": 1.0}, "per_SD": {f"a_wb:{col}": sd},
              "ASD_slope_at_centre": {"a_wb": 1.0}}
        try:
            res = r.fit(md, f"fast_severity_FAST_{key}", cr2=True, extra_targets=tg)
        except EngineError as e:
            out[key] = {"estimable": False, "error": str(e)[:300]}
            continue
        out[key] = {"estimable": True, **_counts(md), "centre": centre, "sd_children": sd,
                    "targets": {k: _t(res, k) for k in tg},
                    "units": f"log RT per SD_i (per point / per SD of {key})"}
    return out


# data quality ---------------------------------------------------------------------------------------------------------
def _child_slopes(f: pd.DataFrame) -> pd.Series:
    """Descriptive within-block slope of lrt on a_wb per child (both demeaned within child x run x block)."""
    x = f["a_wb"] - f.groupby("pid_run_block")["a_wb"].transform("mean")
    y = f["lrt"] - f.groupby("pid_run_block")["lrt"].transform("mean")
    d = pd.DataFrame({"pid": f["pid"], "xy": x * y, "xx": x * x})
    g = d.groupby("pid")[["xy", "xx"]].sum()
    return g["xy"] / g["xx"]


def child_metrics(r: Run, task: str = "FAST") -> pd.DataFrame:
    model = {"AVSRT": "M1-H2", "FAST": "M3-H2", "IC": "M4-H2"}[task]
    f = r.provider.model(model).frame
    g = f.groupby("pid")
    m = pd.DataFrame({"grp": g["grp"].first(), "n_trials": g.size(), "rt_median_ms": g["lrt"].median().pipe(
        lambda s: np.exp(s) * 1000.0), "lrt_sd": g["lrt"].std(ddof=1), "slope_descriptive": _child_slopes(f)})
    if "a_wb_raw" in f:
        m["alpha_sd_raw"] = g["a_wb_raw"].std(ddof=1)
    rt = SREL.rho_table(r.provider.rel_windows(task), task).set_index("pid")["rho"]
    m["alpha_reliability"] = m.index.map(rt).astype(float)
    try:
        mm = r.provider.miss_model({"FAST": "M3", "IC": "M4"}[task]).frame if task != "AVSRT" else None
    except Exception:  # noqa: BLE001 - miss rate is optional (recorded as missing)
        mm = None
    if mm is not None and "miss" not in mm and "responded" in mm:
        mm = mm.assign(miss=(~mm["responded"].astype(bool)).astype(float))    # as E8
    if mm is not None and "miss" in mm:
        m["miss_rate"] = mm.groupby("pid")["miss"].mean().reindex(m.index).astype(float)
    return m


def _compare(m: pd.DataFrame, col: str) -> dict:
    a = m.loc[m["grp"] < 0, col].dropna().to_numpy(float)
    b = m.loc[m["grp"] > 0, col].dropna().to_numpy(float)
    if len(a) < 3 or len(b) < 3:
        return {"n_TD": int(len(a)), "n_ASD": int(len(b))}
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    g = (b.mean() - a.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9)) if sp > 0 else None
    q = lambda v: [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))]  # noqa: E731
    return {"n_TD": int(len(a)), "n_ASD": int(len(b)), "TD_q25_median_q75": q(a), "ASD_q25_median_q75": q(b),
            "TD_mean": float(a.mean()), "ASD_mean": float(b.mean()), "hedges_g_ASD_minus_TD": g,
            "welch_p": float(stats.ttest_ind(b, a, equal_var=False).pvalue),
            "mann_whitney_p": float(stats.mannwhitneyu(b, a).pvalue)}


def quality(r: Run) -> dict:
    m = child_metrics(r, "FAST")
    cols = [c for c in ("n_trials", "rt_median_ms", "lrt_sd", "miss_rate", "alpha_sd_raw", "alpha_reliability")
            if c in m]
    out: dict = {"definition": "child-level FAST metrics over the M3-H2 rows (miss rate: E8 predictor-set rows); "
                               "ASD vs TD; Hedges g, Welch and Mann-Whitney p (uncorrected, descriptive)",
                 "by_group": {c: _compare(m, c) for c in cols}}
    # M3-H2 + a_wb x (child-centred quality metrics): does the group term survive?
    md = r.provider.model("M3-H2")
    add, terms = {}, []
    for c in ("lrt_sd", "n_trials", "miss_rate", "rt_median_ms"):
        if c not in m:
            continue
        v = m[c].astype(float)
        if c == "n_trials":
            v = np.log(v)
        v = v - v.mean()
        col = f"q_{c}_c"
        add[col] = md.frame["pid"].map(v.fillna(0.0)).to_numpy(float)
        terms.append(f"a_wb:{col}")
    mdq = SENS._add(md, add, terms)
    try:
        res = r.fit(mdq, "fast_quality_M3_H2", reduction=False, fixed_state=r.state("S2"), cr2=True,
                    extra_targets={**GROUP_TARGETS, **{t.replace(":", "_x_"): {t: 1.0} for t in terms}})
        out["adjusted_model"] = {"added_terms": terms, **_counts(mdq),
                                 "targets": {k: _t(res, k) for k in ("H2", "TD", "ASD",
                                                                     *[t.replace(":", "_x_") for t in terms])},
                                 "note": "child metrics centred on the child mean; log(n_trials); missing -> 0"}
    except EngineError as e:
        out["adjusted_model"] = {"estimable": False, "error": str(e)[:300]}
    return out


# robustness -----------------------------------------------------------------------------------------------------------
def _extend_registry() -> None:
    for sid in ROBUST_IDS + ("S-HET",):
        s = SENS.REGISTRY.get(sid)
        if s is not None and "M3-H2" not in s.models:
            SENS.REGISTRY[sid] = replace(s, models=(*s.models, "M3-H2"))


def _robust_one(r: Run, sid: str, st: dict) -> dict:
    try:
        md = r.provider.sensitivity(sid, "M3-H2", r.c.sens_ctx("FAST"))
    except Exception as e:  # noqa: BLE001 - recorded, not hidden
        return {"id": sid, "status": "not run", "reason": f"{type(e).__name__}: {e}"[:300]}
    try:
        res = r.fit(md, f"fast_sens_{sid}_M3_H2".replace("-", "_"), reduction=False, fixed_state=st, cr2=True,
                    subdir="sens")
    except EngineError as e:
        return {"id": sid, "status": "not estimable", "reason": str(e)[:300]}
    return {"id": sid, "status": "done", "change": SENS.get(sid).change, **_counts(md), "H2": _t(res, "H2")}


def robustness(r: Run) -> dict:
    _extend_registry()
    st = r.state("S2")
    ids = [s for s in ROBUST_IDS if s in SENS.REGISTRY]
    with ThreadPoolExecutor(r.n_jobs) as ex:
        rows = list(ex.map(lambda s: _robust_one(r, s, st), ids))
    out: dict = {"definition": "sensitivity analyses applicable to FAST / the group model on M3-H2, final M3 "
                               "random-effects structure; target a_wb:grp (ASD - TD)",
                 "entries": {row["id"]: row for row in rows}}
    # S-HET: residual dispersion by group (glmmTMB)
    try:
        md = r.provider.sensitivity("S-HET", "M3-H2", r.c.sens_ctx("FAST"))
        res = r.engine.s_het(md, "fast_S_HET_M3_H2", primary_state=st)
        out["entries"]["S-HET"] = {"id": "S-HET", "status": "done", "change": SENS.get("S-HET").change,
                                   "targets": {k: ST.target_row(v) for k, v in (res.get("targets") or {}).items()},
                                   "dispersion": res.get("dispersion")}
    except Exception as e:  # noqa: BLE001
        out["entries"]["S-HET"] = {"id": "S-HET", "status": "not estimable", "reason": f"{e}"[:300]}
    # S-MATCH: trial-count matching (20 subsamples, as S-MATCH)
    try:
        md = r.provider.model("M3-H2")
        tc = SENS.smatch_target_counts(md)
        draws = ST.smatch_draws(md, tc, reps=20, seed=20261020)

        def one(i):
            m = SENS.refinalise(SENS._replace(md, frame=md.frame.iloc[draws[i]].reset_index(drop=True)),
                                reason="S-MATCH subsample")
            res = r.fit(m, f"fast_smatch_M3_H2_rep{i + 1:02d}", reduction=False, fixed_state=st, cr2=False,
                        subdir="smatch")
            return res["targets"]["H2"]["estimate"], res["targets"]["H2"]["se"]

        with ThreadPoolExecutor(r.n_jobs) as ex:
            est = list(ex.map(one, range(len(draws))))
        e = np.array([x[0] for x in est])
        out["entries"]["S-MATCH"] = {"id": "S-MATCH", "status": "done", "reps": len(e),
                                     "change": "children subsampled to the FAST target trial count",
                                     "H2": {"estimate": float(np.median(e)),
                                            "se": float(np.median([x[1] for x in est])),
                                            "rep_percentiles_2.5_97.5": [float(np.percentile(e, 2.5)),
                                                                         float(np.percentile(e, 97.5))]}}
    except Exception as e:  # noqa: BLE001
        out["entries"]["S-MATCH"] = {"id": "S-MATCH", "status": "not estimable", "reason": f"{e}"[:300]}
    # S-REL-H2: regression calibration, child bootstrap (B = 499, as S-REL)
    try:
        md = r.provider.model("M3-H2")
        win = r.provider.rel_windows("FAST")

        def fit(m, lab):
            return r.fit(m, f"fast_srel_M3_H2_{lab}", reduction=False, fixed_state=st, cr2=False, subdir="srel")

        s = SREL.srel_bootstrap(md, win, fit=fit, target="H2", B=499, seed=20261020, workers=r.n_jobs)
        out["entries"]["S-REL-H2"] = {"id": "S-REL-H2", "status": "done",
                                      "change": "reliability calibration (split-channel rho x 0.741), child bootstrap",
                                      "H2": {"estimate": s.get("estimate"), "se": s.get("se_model"),
                                             "ci95": s.get("ci95_percentile"), "B_ok": s.get("B_ok"),
                                             "bootstrap_sd": s.get("bootstrap_sd")}}
    except Exception as e:  # noqa: BLE001
        out["entries"]["S-REL-H2"] = {"id": "S-REL-H2", "status": "not estimable", "reason": f"{e}"[:300]}
    return out


# figure data ----------------------------------------------------------------------------------------------------------
def figdata(r: Run) -> dict:
    """Pooled alpha-quintile curves (Fig. 2A); the child-level metrics table is written next to the work
    directory."""
    out: dict = {"quintiles": {}}
    figdir = r.work.parent / "figdata"
    figdir.mkdir(parents=True, exist_ok=True)
    for task in ("AVSRT", "FAST", "IC"):
        m = child_metrics(r, task)
        m.assign(task=task).reset_index().to_parquet(figdir / f"child_metrics_{task}.parquet")
        f = r.provider.model({"AVSRT": "M1-H2", "FAST": "M3-H2", "IC": "M4-H2"}[task]).frame
        y = f["lrt"] - f.groupby("pid_run_block")["lrt"].transform("mean")
        q = f.groupby("pid")["a_wb"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        d = pd.DataFrame({"pid": f["pid"], "grp": f["grp"], "q": q, "y": y})
        per = d.groupby(["pid", "q"]).agg(y=("y", "mean"), grp=("grp", "first")).reset_index()
        rows = {}
        for lab, sgn in (("TD", -1), ("ASD", 1)):
            s = per[np.sign(per["grp"]) == sgn].groupby("q")["y"]
            rows[lab] = {"mean_lrt_dev": s.mean().tolist(), "se": s.sem().tolist(), "n_children": int(
                per.loc[np.sign(per["grp"]) == sgn, "pid"].nunique())}
        out["quintiles"][task] = rows
    out["definition"] = ("per child, quintiles of a_wb; mean log RT deviation from the child x run x block mean per "
                         "quintile; then mean and SE over children of each group (descriptive)")
    return out


def demographics(r: Run) -> dict:
    """Sample description of the three H2 model samples by group (Table 1)."""
    cov = r.provider.child_covariates()
    out: dict = {}
    for task, model in (("AVSRT", "M1-H2"), ("FAST", "M3-H2"), ("IC", "M4-H2")):
        md = r.provider.model(model)
        f = md.frame
        age0 = float(((md.meta or {}).get("centring") or {}).get("age_mean", np.nan))
        k = f.groupby("pid").agg(grp=("grp", "first"), age_c=("age_c", "first"), sex_c=("sex_c", "first"),
                                 n=("pid", "size"))
        k["age"] = k["age_c"] + age0
        for key in ("fsiq", "ados_css", "srs2"):
            k[key] = k.index.map(cov.get(key) or {}).astype(float)
        rows = {}
        for lab, sgn in (("TD", -1), ("ASD", 1)):
            g = k[np.sign(k["grp"]) == sgn]

            def ms(v):
                v = v.dropna()
                return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                        "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "min": float(v.min()) if len(v) else None, "max": float(v.max()) if len(v) else None}

            rows[lab] = {"children": int(len(g)), "male": int((g["sex_c"] > 0).sum()), "age": ms(g["age"]),
                         "fsiq": ms(g["fsiq"]), "ados_css": ms(g["ados_css"]), "srs2": ms(g["srs2"]),
                         "trials_per_child": ms(g["n"].astype(float)), "trials_total": int(g["n"].sum())}
        out[task] = rows
    return out
