"""Coupling models of the three tasks: pooled coupling, group differences, the joint three-task model, multiplicity
and equivalence, alternative inference, sensitivity analyses, trial-count matching, reliability calibration, the
censored-RT model, severity, and the exploratory task models (Additional file 1, Tables S1-S3).

Each analysis function receives the context :class:`Ctx` (model-data provider, model engine, settings, and the
outputs of analyses it builds on) and returns a pooled result dict. Model data come from the provider
(:mod:`modeling.provider`), fits from the engine (:mod:`modeling.engine`).

Model names: M1 (AVSRT RT), M2 (AVSRT misses), M3 (FAST RT), M4 (IC RT), M1-hist (AVSRT, alpha over the previous
five trials), M5h (joint model of the three tasks, outcomes standardised within task); ``-H2`` variants carry the
group terms (``grp``: TD -0.5, ASD +0.5; ``a_wb:grp`` = ASD - TD).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from modeling import extras as X
from modeling import inference as I
from modeling import sensitivity as SENS
from modeling import srel as SREL
from modeling.model_data import combine_m5h
from modeling.model_specs import RandomDef
from modeling.r_bridge import EngineError
from modeling.settings import A2_SEEDS, PC3_EXPECTED, SESOI_OF, TRUE_SCORE_RHO_BAR, AnalysisSettings
from modeling.splines import ns_basis, ns_column_names

__all__ = ["Ctx", "pc3_targets", "m5h_with_grp", "smatch_draws", "SENS_PLAN", "model_summary", "target_row",
           "row_estimate", "primary", "pc3", "secondary", "cross_task", "contrast_power",
           "avsrt_coupling", "avsrt_alternative_inference", "avsrt_expected_effects", "avsrt_cr2_refit",
           "avsrt_miss_coupling", "fast_coupling", "ic_coupling", "avsrt_alpha_history_coupling", "joint_task_model",
           "task_tests_multiplicity", "group_differences", "alpha_variability", "sensitivity_analyses",
           "trial_count_matching", "reliability_calibration", "censored_rt_model", "severity_and_siblings_avsrt",
           "exploratory_task_models", "ex_gaussian_model"]

log = logging.getLogger(__name__)


# context --------------------------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    """Context of the analysis functions.

    ``results`` holds the outputs of earlier analyses as ``{key: {"result": dict}}``, keyed P1 (avsrt_coupling),
    S1 (avsrt_miss_coupling), S2 (fast_coupling), S3 (ic_coupling), S4 (joint_task_model), S6
    (avsrt_alpha_history_coupling), H2 (group_differences), SENS (sensitivity_analyses), S-REL
    (reliability_calibration), S-CENS (censored_rt_model).
    """

    provider: object
    engine: object
    settings: AnalysisSettings
    results: dict = field(default_factory=dict)
    n_jobs: int = 4
    log: Callable[[str], None] = log.info
    _ctx: dict | None = None

    def r(self, step: str) -> dict | None:
        rec = self.results.get(step)
        return None if rec is None else rec.get("result")

    def sub(self, step: str, sub: str, fn: Callable[[], dict]) -> dict:
        return fn()

    def sens_ctx(self, task: str) -> dict:
        """Inputs of the sensitivity analyses S-RET, S-ALT, S-FSIQ and S-MED for ``task``."""
        if self._ctx is None:
            cov = self.provider.child_covariates()
            self._ctx = {"retfrac_by_task": self.provider.retfrac(), "alt_step": self.provider.alt_step()["step"],
                         "fsiq": cov["fsiq"], "medication": cov["medication"]}
        b = self._ctx
        return {"retfrac": b["retfrac_by_task"].get(task, {}), "alt_step": b["alt_step"], "fsiq": b["fsiq"],
                "medication": b["medication"]}


# helpers --------------------------------------------------------------------------------------------------------------
def target_row(t: Mapping | None) -> dict | None:
    if not t:
        return None
    keys = ("estimate", "se", "df", "p", "ci95", "ci90", "df_method", "weights", "median",
            "p_two_sided_posterior", "rep_percentiles_2.5_97.5")
    return {k: t.get(k) for k in keys if k in t}


def _group_counts(md) -> dict:
    f = md.frame
    if "grp" not in f.columns:
        return {"children": int(f["pid"].nunique())}
    g = f.groupby("pid")["grp"].first()
    return {"children": int(len(g)), "TD": int((g < 0).sum()), "ASD": int((g > 0).sum())}


def model_summary(md, res: Mapping, *, keep_random: bool = True) -> dict:
    f = md.frame
    oc = res.get("optimum_check") or {}
    conv = res.get("convergence") or {}
    out = {"model": md.model_def.name, "task": md.task, "rows": int(len(f)), **_group_counts(md),
           "blocks": int(f["pid_run_block"].nunique()) if "pid_run_block" in f else None,
           "rows_per_child_median": float(f.groupby("pid").size().median()),
           "has_grp_term": bool(md.has_grp), "outcome_kind": md.outcome_kind,
           "children_excluded": (md.log or {}).get("children_excluded"),
           "final_structure": (res.get("final_state") or {}).get("re"), "flag": res.get("flag"),
           "steps": [s.get("step") for s in res.get("reduction_log") or []], "n_fits": res.get("n_fits"),
           "singular": res.get("singular"), "converged": conv.get("converged"),
           "allfit_used": conv.get("allfit_used"),
           "optimum_check": {k: oc.get(k) for k in ("ran", "switched", "kept", "loglik_gain", "n_refits")},
           "targets": {k: target_row(v) for k, v in (res.get("targets") or {}).items()},
           "tau_hat_a_wb_slope_sd": res.get("a_wb_slope_sd"), "sigma": res.get("sigma"),
           "loglik": res.get("loglik"), "nobs": res.get("nobs"), "engine_seconds": res.get("seconds"),
           "cached": bool(res.get("cached", False))}
    if keep_random and res.get("random"):
        out["random"] = res.get("random")
    if res.get("cr2") and isinstance(res["cr2"], Mapping) and "targets" in res["cr2"]:
        out["cr2"] = {k: target_row(v) for k, v in res["cr2"]["targets"].items()}
    if res.get("second_impl"):
        si = res["second_impl"]
        out["second_impl_glmer"] = {k: si.get(k) for k in ("pass", "max_fe_diff_rel_se", "loglik_diff", "error")}
    return out


def _core(res: Mapping) -> dict:
    """What later analyses need from an engine fit (structure, formula, targets, slope SD; pooled)."""
    return {"final_state": res.get("final_state"), "full_formula": res.get("full_formula"),
            "a_wb_slope_sd": res.get("a_wb_slope_sd"),
            "targets": {k: target_row(v) for k, v in (res.get("targets") or {}).items()}}


def _fit_or_none(c: Ctx, md, label: str, **kw) -> tuple[dict | None, str | None]:
    try:
        return c.engine.fit(md, label, **kw), None
    except EngineError as e:
        c.log(f"{label}: not estimable ({str(e)[:200]})")
        return None, str(e)[:500]


def pc3_targets(md) -> dict:
    """Expected-effect contrasts (Additional file 1, Table S1d): M1 modality beta(cAV), the ns(soa, 4) prediction
    at the upper minus the lower interior quartile knot (2.732 s minus 1.816 s), beta(prev_miss); M3 beta(prev_miss),
    beta(soa_prev_c); M4 beta(prev_miss)."""
    out: dict = {}
    terms = set(md.fixed_terms)
    if md.model_def.name in ("M1", "PC3-MARGINAL") or "cAV" in terms:
        sp = next((s for s in md.splines if s["variable"] == "soa"), None)
        if "cAV" in terms:
            out["PC3_modality_cAV"] = {"cAV": 1.0}
        if sp is not None:
            kn = list(sp["interior_knots"])
            b = ns_basis(np.array([kn[-1], kn[0]]), kn, sp["boundary_knots"])
            cols = ns_column_names("soa", int(sp["df"]))
            out["PC3_soa_q75_minus_q25"] = {c: float(b[0, j] - b[1, j]) for j, c in enumerate(cols)}
    if "prev_miss" in terms:
        out["PC3_prev_miss"] = {"prev_miss": 1.0}
    if "soa_prev_c" in terms:
        out["PC3_soa_prev"] = {"soa_prev_c": 1.0}
    return out


BETA_V = {"beta_V": {"a_wb": 1.0, "a_wb:cVA": 0.5, "a_wb:cAV": -1.0 / 3.0}}


def _a_rows(a2a3: Mapping) -> dict:
    a2, a3 = a2a3["A2"], a2a3["A3"]
    return {"A2": {k: a2.get(k) for k in ("statistic", "p", "n_surrogates", "n_children", "null_sd",
                                          "null_q025_q975", "seed", "fft_vs_explicit_max_abs_diff")},
            "A3": {k: a3.get(k) for k in ("estimate", "se", "df", "p", "ci95", "ci90", "n_clusters", "method")}}


def _a1_rows(res: Mapping | None) -> dict | None:
    if res is None:
        return None
    return {"targets": {k: target_row(v) for k, v in (res.get("targets") or {}).items()},
            "optimizer_used": res.get("optimizer_used"), "optim_error": res.get("optim_error"),
            "phi": res.get("phi"), "note": "A1 (AR(1) residuals) omits the block intercept and the child time slopes"}


def _a1(c: Ctx, md, label: str, targets: Mapping) -> dict | None:
    if not c.settings.run_a1:
        return None
    try:
        return _a1_rows(c.engine.a1(md, label, targets=targets))
    except EngineError as e:
        return {"error": str(e)[:400]}


def _child_t_ci(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return {"n_children": n, "mean": None}
    m, s = float(v.mean()), float(v.std(ddof=1))
    q = stats.t.ppf(0.975, n - 1)
    return {"n_children": n, "mean": m, "sd": s, "ci95": [m - q * s / math.sqrt(n), m + q * s / math.sqrt(n)]}


# multiplicity and equivalence -----------------------------------------------------------------------------------------
PC3_LEG_NAMES = {"PC3_modality_cAV": "modality", "PC3_soa_q75_minus_q25": "SOA", "PC3_prev_miss": "previous outcome"}


def row_estimate(row: Mapping | None) -> I.Estimate | None:
    """Engine target row as an :class:`modeling.inference.Estimate` (``None`` if missing or not finite)."""
    if not row:
        return None
    try:
        e = float(row["estimate"])
        s = float(row["se"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(e) and math.isfinite(s) and s > 0):
        return None
    df = row.get("df")
    try:
        df = None if df is None else float(df)
    except (TypeError, ValueError):
        df = None
    if df is not None and not math.isfinite(df):
        df = None
    return I.Estimate(e, s, df)


def primary(p1_row: Mapping | None, h1a_row: Mapping | None, beta_v_row: Mapping | None = None) -> dict:
    """AVSRT coupling (P1) and auditory-trial coupling (H1A) with equivalence tests against the AVSRT SESOI
    (``inference.primary_family``); the visual-trial slope beta_V is descriptive."""
    p1, h1a = row_estimate(p1_row), row_estimate(h1a_row)
    sesoi = SESOI_OF["P1"]
    if p1 is None:
        out = {"P1": {"category": "not_estimable", "categories": ["not_estimable"]},
               "H1A": {"category": "estimation_only", "tested": False,
                       "note": "P1 not estimable: H1-A is estimation only"}}
        if h1a is not None:
            out["H1A"].update(I.primary_category(h1a, SESOI_OF["H1A"]))
            out["H1A"]["category"] = "estimation_only"
            out["H1A"]["tested"] = False
    elif h1a is None:
        out = {"P1": I.primary_category(p1, sesoi), "H1A": {"category": "not_estimable", "tested": False}}
    else:
        out = I.primary_family(p1, h1a, sesoi)
    bv = row_estimate(beta_v_row)
    if bv is not None:
        out["beta_V"] = {"estimate": bv.estimate, "se": bv.se, "df": bv.df, "ci95": list(bv.ci95),
                         "note": "visual-trial slope beta + 1/2 beta(a_wb:cVA) - 1/3 beta(a_wb:cAV); descriptive, "
                                 "without a test"}
    return out


def pc3(m1_targets: Mapping[str, Mapping] | None, m3_targets: Mapping | None = None,
        m4_targets: Mapping | None = None) -> dict:
    """Expected effects in the AVSRT RT model (modality, SOA, previous outcome), each with its expected sign and a
    95% CI excluding 0; M3 / M4 legs reported alongside (Additional file 1, Table S1d)."""
    def leg(row, name):
        e = row_estimate(row)
        sign = PC3_EXPECTED[name]
        if e is None:
            return {"available": False, "expected_sign": "+" if sign > 0 else "-", "pass": False}
        lo, hi = (row.get("ci95") or e.ci95)
        ok = bool(lo > 0) if sign > 0 else bool(hi < 0)
        return {"available": True, "estimate": e.estimate, "se": e.se, "ci95": [float(lo), float(hi)],
                "expected_sign": "+" if sign > 0 else "-", "pass": ok}

    m1 = {k: leg((m1_targets or {}).get(k), k) for k in ("PC3_modality_cAV", "PC3_soa_q75_minus_q25",
                                                         "PC3_prev_miss")}
    failed = [PC3_LEG_NAMES[k] for k, v in m1.items() if not v["pass"]]
    out = {"binding": "M1 legs: each with the expected sign and a 95% CI excluding 0",
           "M1": m1, "passed": not failed, "failed_legs": failed,
           "label": None if not failed else f"PC3 failed ({', '.join(failed)})"}
    out["M3"] = {k: leg((m3_targets or {}).get(k), k) for k in ("PC3_prev_miss", "PC3_soa_prev")} \
        if m3_targets is not None else None
    out["M4"] = {"PC3_prev_miss": leg((m4_targets or {}).get("PC3_prev_miss"), "PC3_prev_miss")} \
        if m4_targets is not None else None
    return out


def secondary(rows: Mapping[str, Mapping | None], *, a3nl_row: Mapping | None = None) -> dict:
    """The six task-level tests (S1-S6) with Benjamini-Hochberg correction, m = 6 (a missing or non-finite member
    enters with p = 1), and their equivalence tests (Additional file 1, Table S1a)."""
    members = {k: row_estimate(rows.get(k)) for k in ("S1", "S2", "S3", "S4", "S5", "S6")}
    return I.secondary_family(members, SESOI_OF, a3nl_s6=row_estimate(a3nl_row), rho_bar=TRUE_SCORE_RHO_BAR)


def contrast_power(se: float, delta: float = 0.03, alpha: float = 0.05) -> float:
    """Normal-approximation power of a two-sided Wald test of a contrast with standard error ``se``."""
    if not (se and math.isfinite(se) and se > 0):
        return float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(delta / se - z) + stats.norm.cdf(-delta / se - z))


def cross_task(c1_row: Mapping | None, c2_row: Mapping | None, *, p1_category: str, secondary_res: Mapping) -> dict:
    """The two contrasts between tasks with Holm correction (``inference.cross_task``; Additional file 1,
    Table S1b), with S4 / S2 / S3 from the BH-adjusted family and the normal-approximation power to detect a
    contrast of 0.03 r."""
    c1, c2 = row_estimate(c1_row), row_estimate(c2_row)
    if c1 is None or c2 is None:
        return {"label_id": None, "label": "not estimable (M5h)", "note": "M5h contrasts not estimable"}
    mem = secondary_res.get("members", {})
    pw = {"C1": contrast_power(c1.se), "C2": contrast_power(c2.se)}
    out = I.cross_task(c1, c2, p1_category=p1_category,
                       s4_practically_null=bool(mem.get("S4", {}).get("category") == "practically_null"),
                       s2_positive=bool(mem.get("S2", {}).get("category") == "positive"),
                       s3_positive=bool(mem.get("S3", {}).get("category") == "positive"),
                       power_c003=pw)
    out["power_definition"] = ("normal approximation, two-sided alpha 0.05 without the Holm step, at the model SE "
                               "of each contrast")
    return out


# AVSRT coupling -------------------------------------------------------------------------------------------------------
def avsrt_coupling(c: Ctx) -> dict:
    md = c.provider.model("M1")
    extra = {**pc3_targets(md), **BETA_V}
    res, err = _fit_or_none(c, md, "task_M1", cr2=False, extra_targets=extra)
    if res is None:
        return {"estimable": False, "error": err, "decision": primary(None, None)}
    t = res["targets"]
    return {"estimable": True, "model": model_summary(md, res), "core": _core(res),
            "decision": primary(t.get("P1"), t.get("H1A"), t.get("beta_V")),
            "h2_unadjusted": target_row(t.get("H2_unadjusted")),
            "note": "P1 = beta(a_wb) of M1 (Satterthwaite t, 95% CI; TOST on the 90% CI)"}


def avsrt_alternative_inference(c: Ctx) -> dict:
    md = c.provider.model("M1")
    n = c.settings.n_surrogates
    a2a3 = {t: _a_rows(X.a2_a3_model(md, target=t, seed=A2_SEEDS[t], n_surrogates=n)) for t in ("P1", "H1A", "S5")}
    a1 = _a1(c, md, "task_M1_a1", {k: md.model_def.targets[k] for k in ("P1", "H1A", "S5")})
    return {"A2_A3": a2a3, "A1": a1, "seeds": {t: A2_SEEDS[t] for t in ("P1", "H1A", "S5")},
            "n_surrogates": n}


def avsrt_expected_effects(c: Ctx) -> dict:
    p1 = c.r("P1") or {}
    m1_t = (p1.get("core") or {}).get("targets") or {}
    pc = pc3(m1_t)
    # M1 without ns(log_tsp, 4) and tsp_join (SOA effect without the time-since-press spline)
    md = c.provider.sensitivity("PC3-MARGINAL", "M1", {})
    extra = {k: v for k, v in pc3_targets(md).items() if k == "PC3_soa_q75_minus_q25"}
    res, err = _fit_or_none(c, md, "task_PC3_MARGINAL", cr2=False, second_impl=False, extra_targets=extra)
    if res is None:
        pc["PC3_marginal"] = {"estimable": False, "error": err}
    else:
        row = target_row(res["targets"].get("PC3_soa_q75_minus_q25"))
        lo, hi = row["ci95"] if row else (None, None)
        pc["PC3_marginal"] = {"estimable": True, "SOA": row, "expected_sign": "-",
                              "pass": bool(hi is not None and hi < 0), "model": model_summary(md, res, keep_random=False),
                              "note": "M1 without the spline of the time since the last press"}
    # within-child SOA slope and the within-child correlation of SOA with the time since the last press
    f = c.provider.model("M1").frame
    sl, rr = [], []
    for _, g in f.groupby("pid"):
        s = g["soa"].to_numpy(float)
        y = g["lrt"].to_numpy(float)
        if len(g) >= 10 and np.std(s) > 0:
            sl.append(float(np.polyfit(s, y, 1)[0]))
            ts = g["log_tsp"].to_numpy(float)
            if np.std(ts) > 0:
                rr.append(float(np.corrcoef(s, ts)[0, 1]))
    pc["investigation"] = {
        "unadjusted_within_child_soa_slope": {**_child_t_ci(np.array(sl)),
                                              "definition": "mean over children of the per-child OLS slope of lrt on "
                                                            "soa (M1 rows), log RT per s; t 95% CI"},
        "within_child_corr_soa_log_tsp": {"median": float(np.median(rr)) if rr else None, "n_children": len(rr),
                                          "definition": "median over children of the per-child Pearson r of soa and "
                                                        "log_tsp (M1 rows)"},
        "reported_as_investigation": not pc["passed"],
        "note": "M1 without the time-since-press spline and the two within-child descriptives"}
    return pc


def avsrt_cr2_refit(c: Ctx) -> dict:
    md = c.provider.model("M1")
    extra = {**pc3_targets(md), **BETA_V}
    res, err = _fit_or_none(c, md, "task_M1_cr2", cr2=True, extra_targets=extra)
    out: dict = {}
    if res is None:
        return {"estimable": False, "error": err}
    p1 = (c.r("P1") or {}).get("core", {}).get("targets", {}).get("P1") or {}
    same = p1 and abs(res["targets"]["P1"]["estimate"] - p1["estimate"]) <= 1e-8 * max(1.0, abs(p1["estimate"]))
    out["cr2"] = {k: target_row(v) for k, v in ((res.get("cr2") or {}).get("targets") or {}).items()}
    out["refit_reproduces_P1"] = bool(same)
    if c.settings.second_impl:
        out["statsmodels"] = c.engine.sm_check("task_M1_cr2")
    return out


# task-level tests -----------------------------------------------------------------------------------------------------
def avsrt_miss_coupling(c: Ctx) -> dict:
    md = c.provider.model("M2")
    res, err = _fit_or_none(c, md, "task_M2", second_impl=True)
    out: dict = {"estimable": res is not None}
    if res is None:
        out["error"] = err
    else:
        out["model"] = model_summary(md, res)
        out["core"] = _core(res)
    try:
        a3l = c.engine.a3l(md, "task_M2_a3l")
        a2l = X.a2l_surrogate(md, a3l["p0"], seed=A2_SEEDS["S1"], n_surrogates=c.settings.n_surrogates,
                              children=a3l["children_kept"])
        out["A3L"] = {"targets": {k: target_row(v) for k, v in (a3l.get("targets") or {}).items()},
                      "n_children_kept": a3l.get("n_children_kept", len(a3l.get("children_kept") or []))}
        out["A2L"] = {k: a2l.get(k) for k in ("statistic", "p", "n_surrogates", "n_children", "seed")}
    except EngineError as e:
        out["A3L"] = {"error": str(e)[:400]}
    m = c.provider.sensitivity("S1-A", "M2", {})
    r2, err2 = _fit_or_none(c, m, "task_S1A", second_impl=False)
    out["S1-A"] = ({"model": model_summary(m, r2, keep_random=False),
                    "note": "ocular diagnostic of the miss model"} if r2 else
                   {"estimable": False, "error": err2})
    return out


def _lmm_member(c: Ctx, model: str, target: str, label: str, *, extra_targets=None, a1_targets=None,
                x_col: str = "a_wb") -> dict:
    md = c.provider.model(model)
    res, err = _fit_or_none(c, md, label, cr2=True, extra_targets=extra_targets)
    out: dict = {"estimable": res is not None}
    if res is None:
        out["error"] = err
        return out
    out["model"] = model_summary(md, res)
    out["core"] = _core(res)
    if c.settings.second_impl:
        out["statsmodels"] = c.engine.sm_check(label)
    a = X.a2_a3_model(md, target=target, x_col=x_col, seed=A2_SEEDS[target], n_surrogates=c.settings.n_surrogates)
    out["A2_A3"] = _a_rows(a)
    out["A1"] = _a1(c, md, f"{label}_a1", a1_targets or {target: md.model_def.targets[target]})
    return out


def fast_coupling(c: Ctx) -> dict:
    md = c.provider.model("M3")
    return _lmm_member(c, "M3", "S2", "task_M3", extra_targets=pc3_targets(md))


def ic_coupling(c: Ctx) -> dict:
    md = c.provider.model("M4")
    return _lmm_member(c, "M4", "S3", "task_M4", extra_targets=pc3_targets(md))


def avsrt_alpha_history_coupling(c: Ctx) -> dict:
    out = _lmm_member(c, "M1-hist", "S6", "task_M1hist", x_col="a_hist5")
    if out.get("estimable"):
        md = c.provider.model("M1-hist")
        a = X.a3nl(md)
        out["A3NL"] = {k: a.get(k) for k in ("estimate", "se", "df", "p", "ci95", "ci90", "n_clusters", "method")}
    return out


def m5h_with_grp(m5h, parts: Mapping) -> object:
    """M5h with the ``a_wb:task:grp`` terms ``aA:grp``, ``aF:grp``, ``aI:grp`` and the group-difference targets
    (descriptive). ``grp`` is the child's TD / ASD code from the task parts."""
    gm: dict = {}
    for md in parts.values():
        if "grp" not in md.frame.columns:
            return m5h
        gm.update(dict(zip(md.frame["pid"], md.frame["grp"])))
    f = m5h.frame.copy()
    f["grp"] = f["pid"].map(gm).astype(float)
    if not np.isfinite(f["grp"].to_numpy(float)).all():
        raise ValueError("M5h rows without grp")
    terms = list(m5h.fixed_terms) + ["aA:grp", "aF:grp", "aI:grp"]
    tg = {"H2_S4": {"aF:grp": 0.5, "aI:grp": 0.5}, "H2_mean3": {"aA:grp": 1 / 3, "aF:grp": 1 / 3, "aI:grp": 1 / 3},
          "H2_A": {"aA:grp": 1.0}, "H2_F": {"aF:grp": 1.0}, "H2_I": {"aI:grp": 1.0}}
    mdef = replace(m5h.model_def, fixed=tuple(terms), grp_targets=tg,
                   sesoi={**dict(m5h.model_def.sesoi), **{k: 0.03 for k in tg}})
    return replace(m5h, frame=f, fixed_terms=terms, model_def=mdef, has_grp=True)


def joint_task_model(c: Ctx) -> dict:
    parts = {"AVSRT": c.provider.model("M1"), "FAST": c.provider.model("M3"), "IC": c.provider.model("M4")}
    m5 = m5h_with_grp(combine_m5h(parts), parts)
    res, err = _fit_or_none(c, m5, "task_M5h", cr2=True)
    out: dict = {"estimable": res is not None}
    if res is None:
        out["error"] = err
    else:
        out["model"] = model_summary(m5, res)
        out["core"] = _core(res)
        out["M5_H2_descriptive"] = {k: {"model": target_row(res["targets"].get(k)),
                                        "cr2": target_row(((res.get("cr2") or {}).get("targets") or {}).get(k))}
                                    for k in ("H2_S4", "H2_mean3", "H2_A", "H2_F", "H2_I")}
    try:
        a = X.a2_a3_s4({"FAST": parts["FAST"], "IC": parts["IC"]}, seed=A2_SEEDS["S4"],
                       n_surrogates=c.settings.n_surrogates)
        out["A2_A3"] = _a_rows(a)
    except Exception as e:  # noqa: BLE001 - recorded; S4 inference-only analyses then unavailable
        out["A2_A3"] = {"error": str(e)[:300]}
    out["note"] = ("beta_AVS is the M5h AVSRT slope, neither beta_A of H1-A nor P1; A1 is not applicable to S4; "
                   "the group-difference targets of M5h are descriptive")
    return out


def _row(step: Mapping | None, target: str) -> dict | None:
    if not step or not step.get("estimable", True):
        return None
    return ((step.get("core") or {}).get("targets") or {}).get(target)


def task_tests_multiplicity(c: Ctx) -> dict:
    p1, s1, s2, s3, s4, s6 = (c.r(k) for k in ("P1", "S1", "S2", "S3", "S4", "S6"))
    rows = {"S1": _row(s1, "S1"), "S2": _row(s2, "S2"), "S3": _row(s3, "S3"), "S4": _row(s4, "S4"),
            "S5": _row(p1, "S5"), "S6": _row(s6, "S6")}
    a3nl = (s6 or {}).get("A3NL")
    fam = secondary(rows, a3nl_row=a3nl)
    cat = ((p1 or {}).get("decision") or {}).get("P1", {}).get("category", "not_estimable")
    ct = cross_task(_row(s4, "C1"), _row(s4, "C2"), p1_category=cat, secondary_res=fam)
    desc = {k: target_row(_row(s4, k)) for k in ("beta_A", "beta_F", "beta_I", "mean3")}
    return {"family": fam, "rows": {k: target_row(v) for k, v in rows.items()}, "cross_task": ct,
            "m5h_descriptive": desc}


# group differences ----------------------------------------------------------------------------------------------------
def group_differences(c: Ctx) -> dict:
    out: dict = {}
    p1 = c.r("P1") or {}
    pooled = ((p1.get("core") or {}).get("targets") or {}).get("P1") or {}
    for model, base in (("M1-H2", "P1"), ("M3-H2", "S2"), ("M4-H2", "S3")):
        st = ((c.r(base) or {}).get("core") or {}).get("final_state")
        if not st:
            out[model] = {"estimable": False, "error": f"{base} not estimable: no final structure"}
            continue
        md = c.provider.model(model)
        res, err = _fit_or_none(c, md, f"task_{model.replace('-', '_')}", reduction=False, fixed_state=st, cr2=True)
        if res is None:
            out[model] = {"estimable": False, "error": err}
            continue
        rec = {"estimable": True, "model": model_summary(md, res), "core": _core(res),
               "fit": "final random-effects structure of the task model without group terms"}
        if model == "M1-H2" and c.settings.second_impl:
            rec["statsmodels"] = c.engine.sm_check("task_M1_H2")
        out[model] = rec
    out["unadjusted_M1"] = target_row(p1.get("h2_unadjusted"))
    out["pooled_beta_P1"] = pooled.get("estimate")
    out["note"] = "group difference in coupling (a_wb:grp, ASD - TD) in each task, estimation"
    return out


# alpha variability (E3a / E3b) ----------------------------------------------------------------------------------------
def alpha_variability(c: Ctx) -> dict:
    md = c.provider.model("M1")
    kids = set(md.frame["pid"].unique())
    grp = md.frame.groupby("pid")["grp"].first() if "grp" in md.frame else pd.Series(dtype=float)
    w = c.provider.alpha_windows("AVSRT")
    w = w[w["pid"].isin(kids)]
    sd = w.groupby("pid")["alpha"].std(ddof=1)
    rt = SREL.rho_table(c.provider.rel_windows("AVSRT").pipe(lambda d: d[d["pid"].isin(kids)]), "AVSRT")
    rho = dict(zip(rt["pid"], rt["rho"]))
    sdr = sd * np.sqrt(sd.index.map(lambda k: rho.get(k, SREL.POOLED_RHO["AVSRT"])).to_numpy(float))

    def summ(v: pd.Series) -> dict:
        v = v.dropna()
        if len(v) < 5:
            return {"n": int(len(v))}
        return {"n": int(len(v)), "median": float(v.median()), "q25": float(v.quantile(0.25)),
                "q75": float(v.quantile(0.75)), "mean": float(v.mean())}

    e3a = {"definition": "per child SD (ddof 1) of single-trial alpha (log10 power) over the primary predictor-set "
                         "windows with a valid window, M1 children; also x sqrt(rho_i) (split-time, S-REL rho_i)",
           "sd_alpha": {"all": summ(sd)}, "sd_alpha_x_sqrt_rho": {"all": summ(sdr)}}
    for lab, sign in (("TD", -1), ("ASD", 1)):
        ids = [k for k, g in grp.items() if np.sign(g) == sign]
        e3a["sd_alpha"][lab] = summ(sd[sd.index.isin(ids)])
        e3a["sd_alpha_x_sqrt_rho"][lab] = summ(sdr[sdr.index.isin(ids)])
    # E3b: M1 with unscaled alpha, on the final M1 random-effects structure
    st = ((c.r("P1") or {}).get("core") or {}).get("final_state")
    out = {"E3a": e3a}
    if st:
        f = md.frame.copy()
        f["a_wb_standardised"] = f["a_wb"]
        f["a_wb"] = f["a_wb_raw"].to_numpy(float)
        mr = SENS._replace(md, frame=f, meta={"e3b": "a_wb_raw in place of a_wb in every term (M1-raw)"})
        tg = {"TD": {"a_wb": 1.0, "a_wb:grp": -0.5}, "ASD": {"a_wb": 1.0, "a_wb:grp": 0.5}} if md.has_grp else None
        res, err = _fit_or_none(c, mr, "task_M1raw", reduction=False, fixed_state=st, cr2=False, extra_targets=tg)
        out["E3b"] = ({"model": model_summary(mr, res, keep_random=False),
                       "units": "log RT per log10 power (block-centred, unscaled alpha)",
                       "note": "fixed effect of a_wb_raw; group values from a_wb_raw:grp (not from BLUP ratios)"}
                      if res else {"estimable": False, "error": err})
    else:
        out["E3b"] = {"estimable": False, "error": "M1 not estimable"}
    return out


# sensitivity analyses -------------------------------------------------------------------------------------------------
#: (sensitivity ID, model) pairs fitted by sensitivity_analyses: every registry entry except those fitted by their own
#: function (PC3-MARGINAL, S1-A, S-MATCH, S-REL / S-REL-H2 / S-DTOST, S-CENS) and S-GAZE (per-trial gaze flags not
#: available). S-HARM's M5h is fitted after its M1 / M3 / M4 parts.
_OWN_STEP = {"PC3-MARGINAL", "S1-A", "S-MATCH", "S-REL", "S-REL-H2", "S-DTOST", "S-CENS", "S-GAZE"}
SENS_PLAN: tuple[tuple[str, str], ...] = tuple(
    (sid, m) for sid, s in SENS.REGISTRY.items() if sid not in _OWN_STEP for m in s.models if m != "M5h")


def _sens_one(c: Ctx, sid: str, model: str, p1_state, keep: dict | None = None) -> dict:
    spec = SENS.get(sid)
    task = {"M1": "AVSRT", "M1-H2": "AVSRT", "M2": "AVSRT", "M3": "FAST", "M4": "IC"}[model]
    label = f"task_sens_{sid}_{model}".replace("-", "_")
    try:
        md = c.provider.sensitivity(sid, model, c.sens_ctx(task))
    except SENS.InputMissingError as e:
        return {"id": sid, "model": model, "status": "not run", "reason": str(e)[:300]}
    except (ValueError, KeyError) as e:         # e.g. no child left after the row filter: recorded, not hidden
        return {"id": sid, "model": model, "status": "not estimable", "reason": f"{type(e).__name__}: {e}"[:300]}
    if keep is not None:
        keep[(sid, model)] = md
    kind = spec.fit_kind
    try:
        if kind == "path":
            res = c.engine.fit(md, label, second_impl=False, subdir="sens")
        elif kind == "fixed_structure":
            if not p1_state:
                return {"id": sid, "model": model, "status": "not estimable", "reason": "no final M1 structure"}
            res = c.engine.fit(md, label, reduction=False, fixed_state=p1_state, second_impl=False, subdir="sens")
        elif kind == "het":
            res = c.engine.s_het(md, label, primary_state=p1_state)
            return {"id": sid, "model": model, "status": "done", "fit_kind": kind,
                    "targets": {k: target_row(v) for k, v in (res.get("targets") or {}).items()},
                    "dispersion": res.get("dispersion"), "rows": int(len(md.frame)), **_group_counts(md),
                    "note": spec.note or "glmmTMB dispformula ~ grp; the CR2 SE of the group difference is the "
                                         "M1-H2 CR2 row"}
        else:
            raise ValueError(f"{sid}: fit kind {kind} is not run in SENS")
    except EngineError as e:
        return {"id": sid, "model": model, "status": "not estimable", "reason": str(e)[:300]}
    return {"id": sid, "model": model, "status": "done", "fit_kind": kind, "qualifier": spec.qualifier,
            "descriptive": spec.descriptive, "change": spec.change, "summary": model_summary(md, res, keep_random=False)}


def sensitivity_analyses(c: Ctx) -> dict:
    p1_state = ((c.r("P1") or {}).get("core") or {}).get("final_state")
    harm: dict = {}

    def one(pair):
        sid, model = pair
        return c.sub("SENS", f"{sid}|{model}", lambda: _sens_one(c, sid, model, p1_state,
                                                                  harm if sid == "S-HARM" else None))

    with ThreadPoolExecutor(max(1, c.n_jobs)) as ex:
        rows = list(ex.map(one, SENS_PLAN))
    out = {"entries": {f"{r['id']}|{r['model']}": r for r in rows}}

    def m5h_harm():
        parts = {}
        for task, m in (("AVSRT", "M1"), ("FAST", "M3"), ("IC", "M4")):
            md = harm.get(("S-HARM", m)) or c.provider.sensitivity("S-HARM", m, c.sens_ctx(task))
            parts[task] = md
        m5 = m5h_with_grp(combine_m5h(parts), parts)
        try:
            res = c.engine.fit(m5, "task_sens_S_HARM_M5h", second_impl=False, subdir="sens")
        except EngineError as e:
            return {"id": "S-HARM", "model": "M5h", "status": "not estimable", "reason": str(e)[:300]}
        return {"id": "S-HARM", "model": "M5h", "status": "done", "fit_kind": "path", "qualifier": True,
                "summary": model_summary(m5, res, keep_random=False)}

    out["entries"]["S-HARM|M5h"] = c.sub("SENS", "S-HARM|M5h", m5h_harm)
    out["not_run"] = {"S-GAZE": "not available: per-trial gaze flags of the data owners were not used"}
    out["note"] = ("random-effects simplification path of the main models (the final M1 structure for the -H2 "
                   "variants)")
    return out


def smatch_draws(md, target_counts: Mapping[str, int], *, reps: int, seed: int) -> list[np.ndarray]:
    """The S-MATCH subsamples exactly as ``sensitivity._fit_match`` draws them (same generator, same order)."""
    rng = np.random.default_rng(seed)
    f = md.frame
    by = {k: np.flatnonzero(f["pid"].to_numpy() == k) for k in f["pid"].unique()}
    draws = []
    for _ in range(reps):
        keep = []
        for k, idx in by.items():
            n = int(min(len(idx), target_counts.get(k, len(idx))))
            keep.append(np.sort(rng.choice(idx, n, replace=False)))
        draws.append(np.concatenate(keep))
    return draws


def trial_count_matching(c: Ctx) -> dict:
    out: dict = {"target_counts": dict(SENS.SMATCH_TARGET_COUNT), "reps": c.settings.smatch_reps,
                 "seed": c.settings.smatch_seed}
    for model in ("M1", "M3", "M4"):
        def run(model=model):
            md = c.provider.model(model)
            tc = SENS.smatch_target_counts(md)
            draws = smatch_draws(md, tc, reps=c.settings.smatch_reps, seed=c.settings.smatch_seed)
            tgts = [next(iter(md.model_def.targets))]
            if "a_wb:grp" in md.fixed_terms:
                tgts += [t for t in md.model_def.grp_targets if t not in tgts]
            f = md.frame
            n_sub = int(sum(len(np.flatnonzero(f["pid"].to_numpy() == k)) > n for k, n in tc.items()))

            def one(r):
                m = SENS.refinalise(SENS._replace(md, frame=f.iloc[draws[r]].reset_index(drop=True)),
                                    reason="S-MATCH subsample")
                res = c.engine.fit(m, f"task_smatch_{model}_rep{r + 1:02d}", second_impl=False, cr2=False,
                                   subdir="smatch")
                return {t: (res["targets"][t]["estimate"], res["targets"][t]["se"]) for t in tgts}

            with ThreadPoolExecutor(max(1, c.n_jobs)) as ex:
                est = list(ex.map(one, range(len(draws))))
            rec = {"n_children": int(f["pid"].nunique()), "n_children_subsampled": n_sub,
                   "target_count": SENS.SMATCH_TARGET_COUNT[md.task], "targets": {}}
            for t in tgts:
                e = np.array([x[t][0] for x in est], float)
                rec["targets"][t] = {"estimate": float(np.median(e)),
                                     "se": float(np.median([x[t][1] for x in est])),
                                     "rep_percentiles_2.5_97.5": [float(np.percentile(e, 2.5)),
                                                                  float(np.percentile(e, 97.5))],
                                     "estimates": e.tolist()}
            return rec

        try:
            out[model] = c.sub("S-MATCH", model, run)
        except EngineError as e:
            out[model] = {"status": "not estimable", "reason": str(e)[:300]}
    out["note"] = "median over the repetitions; the 2.5 / 97.5 percentiles describe subsampling variability"
    return out


def reliability_calibration(c: Ctx) -> dict:
    out: dict = {"B": c.settings.srel_B, "seed": c.settings.srel_seed}
    plan = (("M1", "P1", "P1"), ("M3", "S2", "S2"), ("M4", "S3", "S3"), ("M1-H2", "P1", "H2"))
    for model, base, tgt in plan:
        st = ((c.r(base) or {}).get("core") or {}).get("final_state")
        if not st:
            out[model] = {"status": "not estimable", "reason": f"{base} not estimable"}
            continue

        def run(model=model, st=st, tgt=tgt):
            md = c.provider.model(model)
            win = c.provider.rel_windows(md.task)
            sub = f"srel/{model.replace('-', '_')}"

            def fit(m, lab):
                return c.engine.fit(m, f"task_srel_{model.replace('-', '_')}_{lab}", reduction=False, fixed_state=st,
                                    cr2=False, second_impl=False, subdir=sub)

            r = SREL.srel_bootstrap(md, win, fit=fit, target=tgt, B=c.settings.srel_B, seed=c.settings.srel_seed,
                                    workers=max(1, c.n_jobs))
            rec = {k: r.get(k) for k in ("target", "estimate", "se_model", "ci95_percentile", "ci90_percentile",
                                         "B", "B_ok", "seed", "bootstrap_sd", "rho_children", "rho_fallback_children",
                                         "rho_median", "rho_min", "rho_max", "mean_bootstrap_fallbacks",
                                         "final_state", "flag_full")}
            rec["B_errors_n"] = len(r.get("B_errors") or [])
            if model == "M1":
                rec["S_DTOST"] = SREL.dtost(r, SESOI_OF["P1"])
                # group slope divided by the square root of the group median reliability
                f = md.frame
                if "grp" in f:
                    g = f.groupby("pid")["grp"].first()
                    rt = SREL.rho_table(win[win["pid"].isin(g.index)], "AVSRT").set_index("pid")["rho"]
                    p1 = ((c.r("P1") or {}).get("core") or {}).get("targets") or {}
                    b, dlt = (p1.get("P1") or {}).get("estimate"), (p1.get("H2_unadjusted") or {}).get("estimate")
                    if b is not None and dlt is not None:
                        rec["group_slope_over_sqrt_rho"] = {
                            lab: {"rho_bar": float(rt[rt.index.isin(g.index[np.sign(g) == s])].median()),
                                  "beta_g": b + s * 0.5 * dlt,
                                  "beta_g_over_sqrt_rho_bar": (b + s * 0.5 * dlt) / math.sqrt(float(
                                      rt[rt.index.isin(g.index[np.sign(g) == s])].median()))}
                            for lab, s in (("TD", -1), ("ASD", 1))}
            if md.task != "AVSRT":
                rec["note"] = "unanchored correction: split-channel rho_i x 0.741"
            return rec

        try:
            out[model] = c.sub("S-REL", model, run)
        except EngineError as e:
            out[model] = {"status": "not estimable", "reason": str(e)[:300]}
    return out


def censored_rt_model(c: Ctx) -> dict:
    md = c.provider.sensitivity("S-CENS", "M1", c.sens_ctx("AVSRT"))
    try:
        r = c.engine.s_cens(md, "task_S_CENS", random_string="(1 + a_wb | pid) + (0 + t_run | pid) + (0 + tib | pid)",
                            iter=c.settings.cens_iter, chains=c.settings.cens_chains)
    except EngineError as e:
        return {"estimable": False, "error": str(e)[:400]}
    return {"estimable": True, "targets": {k: target_row(v) for k, v in (r.get("targets") or {}).items()},
            "max_rhat": r.get("max_rhat"), "min_bulk_ess": r.get("min_bulk_ess"), "converged": r.get("converged"),
            "n_divergent": r.get("n_divergent"), "n_censored": r.get("n_censored"), "nobs": r.get("nobs"),
            "chains": r.get("chains"), "iter": r.get("iter"), "warmup": r.get("warmup"),
            "note": "95% equal-tailed posterior interval as the 95% CI, posterior mean as the estimate"}


# severity and siblings (AVSRT) ----------------------------------------------------------------------------------------
def _drop_grp(md):
    terms = [t for t in md.fixed_terms if "grp" not in t.split(":")]
    return replace(md, fixed_terms=terms, has_grp=False)


def severity_and_siblings_avsrt(c: Ctx) -> dict:
    out: dict = {}
    base = c.provider.model("M1")
    cov = c.provider.child_covariates()
    f = base.frame
    asd = f[f["grp"] > 0].reset_index(drop=True) if "grp" in f else f.iloc[0:0]
    for key, col, lab in (("ados_css", "ados_css_c", "H4b_ados"), ("srs2", "srs2_c", "H4b_srs2")):
        vals = cov.get(key) or {}
        v = asd["pid"].map(vals).astype(float)
        sub = asd.loc[np.isfinite(v.to_numpy(float))].reset_index(drop=True)
        if sub.empty or sub["pid"].nunique() < 10:
            out[lab] = {"estimable": False, "error": f"fewer than 10 ASD children with {key}",
                        "n_children": int(sub["pid"].nunique())}
            continue
        md = SENS.refinalise(_drop_grp(SENS._replace(base, frame=sub)), reason=f"H4b: ASD children with {key}")
        kids = md.frame.groupby("pid")["pid"].first().index
        centre = float(np.mean([vals[k] for k in kids]))
        md = SENS._add(md, {col: md.frame["pid"].map(vals).astype(float).to_numpy() - centre}, [f"a_wb:{col}"])
        res, err = _fit_or_none(c, md, f"task_{lab}", cr2=True, extra_targets={lab: {f"a_wb:{col}": 1.0}})
        out[lab] = ({"estimable": True, "model": model_summary(md, res, keep_random=False),
                     "estimate": target_row(res["targets"].get(lab)), "centre": centre,
                     "units": f"log RT per SD_i per point of {key}", "fit": "random-effects simplification path"}
                    if res else {"estimable": False, "error": err})
    out["ados_counts"] = {k: v for k, v in (cov.get("counts") or {}).items()}
    # H4a: unaffected siblings (descriptive)
    try:
        sib = c.provider.dev_m1_sib()
        res, err = _fit_or_none(c, sib, "task_H4a_SIB", cr2=False)
        out["H4a"] = ({"label": "descriptive (SIB: unaffected siblings, the cohort used to develop the processing)",
                       "model": model_summary(sib, res, keep_random=False)} if res else
                      {"estimable": False, "error": err})
    except Exception as e:  # noqa: BLE001 - H4a is descriptive; recorded
        out["H4a"] = {"estimable": False, "error": f"{type(e).__name__}: {e}"[:300]}
    out["note"] = "H4b: estimation (95% CIs)"
    return out


# exploratory analyses (Table S3) --------------------------------------------------------------------------------------
_EXPL_COLS = ("a_ctx10", "burst_occ", "burst_occ_p95_d3", "burst_occ_p99_d2", "burst_occ_p99_d3", "burst_valid",
              "alpha_e7_mu_left", "alpha_e7_mu_right", "valid_e7_mu_left", "valid_e7_mu_right")


def _expl_base(c: Ctx):
    try:
        return c.provider.build("M1", extra_predictor_columns=_EXPL_COLS)
    except TypeError:
        return c.provider.model("M1")


def _fit_expl(c: Ctx, md, label: str, targets: Mapping, **kw) -> dict:
    res, err = _fit_or_none(c, md, label, cr2=False, second_impl=False, extra_targets=targets, subdir="expl", **kw)
    if res is None:
        return {"estimable": False, "error": err}
    out = {"estimable": True, "targets": {k: target_row(res["targets"].get(k)) for k in targets},
           "rows": int(len(md.frame)), "children": int(md.frame["pid"].nunique()), "flag": res.get("flag"),
           "final_structure": (res.get("final_state") or {}).get("re")}
    if kw.get("export_fixef_vcov") and res.get("vcov"):
        out["_vcov"] = res["vcov"]
        out["_coef_names"] = res.get("coef_names")
    return out


def _nonlinearity_wald(res: Mapping, terms, c_lin: np.ndarray) -> dict:
    """2-df Wald test that the a_wb x ns(t_run, 3) interaction is linear in t_run: with t_run = c0 + sum_k c_k ns_k
    exactly on the model rows, linearity means gamma = lambda c; the test is L gamma = 0 for a basis L of the
    orthogonal complement of c."""
    names = list(res.pop("_coef_names", None) or [])
    V = res.pop("_vcov", None)
    if not names or V is None or not all(t in names for t in terms):
        return {}
    V = np.asarray(V, float)
    idx = [names.index(t) for t in terms]
    tg = res.get("targets") or {}
    if not all(tg.get(t.replace(":", "_x_")) for t in terms):
        return {}
    g = np.array([tg[t.replace(":", "_x_")]["estimate"] for t in terms])
    u, s_, vt = np.linalg.svd(np.asarray(c_lin, float).reshape(1, -1))
    L = vt[1:]
    Vs = V[np.ix_(idx, idx)]
    Lg = L @ g
    w = float(Lg @ np.linalg.solve(L @ Vs @ L.T, Lg))
    return {"chi2": w, "df": int(L.shape[0]), "p": float(stats.chi2.sf(w, L.shape[0])),
            "definition": "H0: the a_wb x time interaction is linear in t_run (2 df)"}


def _child_centre(md, col: str) -> np.ndarray:
    v = md.frame[col].to_numpy(float)
    return v - pd.Series(v).groupby(md.frame["pid"].to_numpy()).transform("mean").to_numpy()


def exploratory_task_models(c: Ctx) -> dict:
    out: dict = {"label": "exploratory: unadjusted for multiplicity"}
    base = _expl_base(c)
    p1 = ((c.r("P1") or {}).get("core") or {})

    def e1():
        v = base.frame["a_wb"].to_numpy(float) ** 2
        md = SENS._add(base, {"a_wb_sq": v - v.mean()}, ["a_wb_sq"])
        return _fit_expl(c, md, "task_E1", {"E1_a_wb_sq": {"a_wb_sq": 1.0}})

    def e2():
        m2 = c.provider.model("M2")
        f = m2.frame
        ok = np.isfinite(pd.to_numeric(f.get("a_hist5"), errors="coerce").to_numpy(float))
        md = SENS._filter(m2, ok, "E2: a_hist5 defined")
        md = SENS._add(md, {}, ["a_hist5"])
        rnd = md.model_def.random
        nr = RandomDef(corr=rnd.corr, droppable=rnd.droppable, uncorrelated=tuple(rnd.uncorrelated) + ("a_hist5",),
                       block=rnd.block, group=rnd.group, block_factor=rnd.block_factor, start_corr=rnd.start_corr)
        md = replace(md, model_def=replace(md.model_def, random=nr))
        return _fit_expl(c, md, "task_E2", {"E2_a_hist5": {"a_hist5": 1.0}})

    def e2b():
        cc = c.provider.context_counts("AVSRT")
        f = base.frame.drop(columns=[x for x in ("n_stim10", "n_press10") if x in base.frame.columns]).merge(
            cc, on=["pid", "run", "sample"], how="left", validate="one_to_one")
        ok = np.isfinite(pd.to_numeric(f["a_ctx10"], errors="coerce").to_numpy(float)) \
            & np.isfinite(f["n_stim10"].to_numpy(float)) & np.isfinite(f["n_press10"].to_numpy(float))
        md = SENS._filter(SENS._replace(base, frame=f), ok, "E2b: a_ctx10 and the context counts defined")
        res = {}
        md_w = SENS._add(md, {}, ["a_ctx10", "n_stim10", "n_press10"])
        res["with_a_wb"] = _fit_expl(c, md_w, "task_E2b_with", {"E2b_a_ctx10": {"a_ctx10": 1.0}})
        g = md.frame.copy()
        g["a_wb_primary"] = g["a_wb"]
        g["a_wb"] = pd.to_numeric(g["a_ctx10"], errors="coerce").to_numpy(float)
        md_a = SENS._add(SENS._replace(md, frame=g), {}, ["n_stim10", "n_press10"])
        res["alone"] = _fit_expl(c, md_a, "task_E2b_alone", {"E2b_a_ctx10_alone": {"a_wb": 1.0}})
        res["note"] = ("alone: a_ctx10 in place of a_wb in every term and the random slope; with a_wb: + a_ctx10; "
                       "both + n_stim10 + n_press10")
        return res

    def e3():
        ok = pd.array(base.frame["burst_valid"], dtype="boolean").fillna(False).to_numpy(bool) \
            & np.isfinite(pd.to_numeric(base.frame["burst_occ"], errors="coerce").to_numpy(float))
        md = SENS._filter(base, ok, "E3: burst_valid")
        res = {}
        for col in ("burst_occ", "burst_occ_p95_d3", "burst_occ_p99_d2", "burst_occ_p99_d3"):
            m = SENS._add(md, {"burst_c": _child_centre(md, col)}, ["burst_c"])
            res[f"predictor_{col}"] = _fit_expl(c, m, f"task_E3_{col}", {"E3_burst": {"burst_c": 1.0}})
        m = SENS._add(md, {"burst_c": _child_centre(md, "burst_occ")}, ["burst_c", "a_wb:burst_c"])
        res["moderator_burst_occ"] = _fit_expl(c, m, "task_E3_mod", {"E3_a_wb_x_burst": {"a_wb:burst_c": 1.0}})
        res["note"] = "burst occupancy centred within child; known ceiling limitation"
        return res

    def e5():
        m2 = c.provider.model("M2").frame
        dd = []
        for _, g in m2.groupby("pid"):
            a, h = g.loc[g["prev_miss"] > 0, "a_wb"], g.loc[g["prev_miss"] < 0, "a_wb"]
            if len(a) >= 5 and len(h) >= 5:
                dd.append(float(a.mean() - h.mean()))
        md = SENS._add(base, {}, ["a_wb:prev_miss"])
        return {"post_miss_alpha": {**_child_t_ci(np.array(dd)),
                                    "definition": "per child mean a_wb after a miss minus after a hit (M2 rows, >= 5 "
                                                  "each); SD_i units"},
                "post_miss_rt_beta_prev_miss": (p1.get("targets") or {}).get("PC3_prev_miss"),
                "coupling_after_miss": _fit_expl(c, md, "task_E5", {"E5_a_wb_x_prev_miss": {"a_wb:prev_miss": 1.0}})}

    def e6():
        md = SENS._add(base, {"soa_c": base.frame["soa"].to_numpy(float) - 2.28125}, ["a_wb:soa_c"])
        r1 = _fit_expl(c, md, "task_E6_soa", {"E6_a_wb_x_soa": {"a_wb:soa_c": 1.0}})
        cols = sorted(col for col in base.frame.columns if col.startswith("ns_t_run_3_"))
        terms = [f"a_wb:{col}" for col in cols]
        md2 = SENS._add(SENS._drop_terms(base, ["a_wb:t_run"]), {}, terms)
        A = np.column_stack([np.ones(len(base.frame))] + [base.frame[col].to_numpy(float) for col in cols])
        coef, *_ = np.linalg.lstsq(A, base.frame["t_run"].to_numpy(float), rcond=None)
        r2 = _fit_expl(c, md2, "task_E6_time", {t.replace(":", "_x_"): {t: 1.0} for t in terms},
                       export_fixef_vcov=True)
        if r2.get("estimable"):
            r2["nonlinearity_wald"] = _nonlinearity_wald(r2, terms, coef[1:])
        r2.pop("_vcov", None)
        r2.pop("_coef_names", None)
        return {"soa_x_a_wb": r1, "nonlinear_time_x_a_wb": r2,
                "note": "soa centred at the M1 knot-table median 2.28125 s; time: a_wb:t_run replaced by a_wb x the "
                        "ns(t_run, 3) basis (M1 knots) and its 2-df linearity test"}

    def e7():
        res = {}
        try:
            mu = c.provider.build("M1", variant="e7_mu")
            res["mu_roi"] = _fit_expl(c, mu, "task_E7_mu", {"E7_mu": {"a_wb": 1.0}})
        except Exception as e:  # noqa: BLE001
            res["mu_roi"] = {"estimable": False, "error": f"{type(e).__name__}: {e}"[:300]}
        f = base.frame
        ok = (pd.array(f["valid_e7_mu_left"], dtype="boolean").fillna(False).to_numpy(bool)
              & pd.array(f["valid_e7_mu_right"], dtype="boolean").fillna(False).to_numpy(bool))
        md = SENS._filter(base, ok, "E7: both mu halves valid")
        g = md.frame
        lat = g["alpha_e7_mu_left"].to_numpy(float) - g["alpha_e7_mu_right"].to_numpy(float)
        lat = lat - pd.Series(lat).groupby(g["pid_run_block"].to_numpy()).transform("mean").to_numpy()
        sd = pd.Series(lat).groupby(g["pid"].to_numpy()).transform(lambda s: s.std(ddof=1)).to_numpy()
        m = SENS._add(md, {"a_lat": lat / sd}, ["a_lat"])
        res["lateralisation"] = _fit_expl(c, m, "task_E7_lat", {"E7_lat": {"a_lat": 1.0}})
        res["note"] = ("mu ROI: the e7_mu predictor variant in place of a_wb; lateralisation: (left - right mu) "
                       "block-centred, per child SD, added to M1")
        return res

    def e8():
        res = {}
        for model in ("M3", "M4"):
            md = c.provider.miss_model(model)
            f = md.frame.copy()
            if "responded" in f.columns:
                f["miss"] = (~f["responded"].astype(bool)).astype(float)
            mdef = replace(md.model_def, family="glmm", outcome="miss", random=RandomDef(corr=("1", "a_wb")),
                           targets={"E8": {"a_wb": 1.0}}, grp_targets={}, sesoi={})
            m = replace(md, frame=f, model_def=mdef, outcome_column="miss")
            res[model] = _fit_expl(c, m, f"task_E8_{model}", {"E8": {"a_wb": 1.0}})
        res["note"] = "FAST / IC miss models, predictor-set rows, glmmTMB; pooled fixed a_wb effect only"
        return res

    def e9():
        res = {}
        for model, tgt in (("M1", "P1"), ("M3", "S2"), ("M4", "S3")):
            try:
                md = c.provider.build(model, variant="e9")
                res[model] = _fit_expl(c, md, f"task_E9_{model}", {f"E9_{tgt}": {"a_wb": 1.0}})
            except Exception as e:  # noqa: BLE001
                res[model] = {"estimable": False, "error": f"{type(e).__name__}: {e}"[:300]}
        return res

    def e10():
        m1 = c.provider.model("M1").frame
        m2 = c.provider.model("M2").frame
        rt = m1.groupby(["pid", "pid_run_block"]).agg(y=("lrt", "mean"), s=("sample", "min"), r=("run", "min"))
        al = m2.assign(alpha_std=m2["a_wb"] + m2["a_slow"]).groupby(["pid", "pid_run_block"]).agg(
            x=("alpha_std", "mean"))
        j = rt.join(al, how="inner").reset_index().sort_values(["pid", "r", "s"])
        xs, ys, cl = [], [], []
        for pid, g in j.groupby("pid"):
            if len(g) < 3:
                continue
            o = np.arange(len(g), dtype=float)
            A = np.column_stack([np.ones(len(g)), o])
            for v, acc in ((g["x"].to_numpy(float), xs), (g["y"].to_numpy(float), ys)):
                b, *_ = np.linalg.lstsq(A, v, rcond=None)
                acc.append(v - A @ b)
            cl.append(np.full(len(g), pid))
        if not xs:
            return {"estimable": False, "error": "no child with >= 3 blocks"}
        x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(cl)
        a3 = X.a3_cr2(x, y, g)
        return {"estimate": {k: a3.get(k) for k in ("estimate", "se", "df", "p", "ci95", "n_clusters")},
                "definition": "per child, block means of log RT (RT rows) and of within-child standardised alpha "
                              "(predictor set) residualised on a linear block-order term; pooled slope, CR2, "
                              "df = children - 1",
                "note": "cannot separate a slow alpha state from fatigue co-drift"}

    def descriptives():
        res: dict = {}
        for model in ("M1", "M3", "M4"):
            md = c.provider.model(model)
            f = md.frame
            R = X.residualise_per_child(f, ["lrt", "a_wb"], X.nuisance_terms(md.fixed_terms))
            xr, yr = R["a_wb"].to_numpy(), R["lrt"].to_numpy()
            s1, s2 = [], []
            for idx in X.child_series(f):
                n = len(idx)
                if n < 20:
                    continue
                h = n // 2
                for part, acc in ((idx[:h], s1), (idx[h:], s2)):
                    xx = xr[part] - xr[part].mean()
                    acc.append(float(xx @ (yr[part] - yr[part].mean()) / (xx @ xx)))
            r = float(np.corrcoef(s1, s2)[0, 1]) if len(s1) > 2 else float("nan")
            rng = np.random.default_rng(20261020)
            a, b = np.array(s1), np.array(s2)
            bs = []
            for _ in range(2000):
                ii = rng.integers(0, len(a), len(a))
                rr = np.corrcoef(a[ii], b[ii])[0, 1]
                bs.append(2 * rr / (1 + rr))
            res[f"slope_reliability_{md.task}"] = {
                "n_children": len(s1), "r_halves": r, "spearman_brown": 2 * r / (1 + r) if np.isfinite(r) else None,
                "ci95_child_bootstrap": [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))],
                "definition": "first vs second half (time order) of the per-child slopes of residualised lrt on "
                              "residualised a_wb (A2 residualisation); Spearman-Brown"}
        m1, m2 = c.provider.model("M1").frame, c.provider.model("M2").frame
        for name, frame, col in (("mean_log_rt", m1, "lrt"), ("miss_rate", m2, "miss")):
            h1, h2 = [], []
            for _, g in frame.sort_values(["pid", "run", "sample"]).groupby("pid"):
                n = len(g)
                h1.append(float(g[col].iloc[: n // 2].mean()))
                h2.append(float(g[col].iloc[n // 2:].mean()))
            r = float(np.corrcoef(h1, h2)[0, 1])
            res[f"behavioural_reliability_{name}"] = {"n_children": len(h1), "r_halves": r,
                                                      "spearman_brown": 2 * r / (1 + r),
                                                      "definition": "temporal split-half (first vs second half of "
                                                                    "the child's rows), pooled"}
        return res

    for key, fn in (("E1", e1), ("E2", e2), ("E2b", e2b), ("E3", e3), ("E5", e5), ("E6", e6), ("E7", e7),
                    ("E8", e8), ("E9", e9), ("E10", e10), ("descriptives", descriptives)):
        try:
            out[key] = c.sub("EXPLORATORY", key, fn)
        except (EngineError, KeyError, ValueError) as e:
            out[key] = {"estimable": False, "error": f"{type(e).__name__}: {e}"[:300]}
    out["E4"] = {"status": "not run", "reason": "the trial-level specparam exponent (AVSRT 1000-ms windows) is not "
                                               "computed by the feature pipeline"}
    return out


# ex-Gaussian RT model (M6) --------------------------------------------------------------------------------------------
def ex_gaussian_model(c: Ctx) -> dict:
    md = c.provider.model("M1")
    f = md.frame.copy()
    order = f.sort_values(["pid", "run", "sample"]).groupby("pid")["pid_run_block"].transform(
        lambda s: pd.factorize(s)[0] + 1)
    f["block_order"] = order.reindex(f.index).to_numpy(float)
    m = SENS._replace(md, frame=f)
    out: dict = {"label": "exploratory: M6 ex-Gaussian model of AVSRT RT (brms), fixed priors and sampler"}
    for mode in ("full", "split"):
        def run(mode=mode):
            r = c.engine.m6(m, f"task_M6_{mode}", mode=mode, settings=c.settings)
            keep = ("mode", "converged", "attempts", "quantities", "split_correlation", "nobs", "n_children",
                    "priors", "note", "interpretable")
            return {k: r.get(k) for k in keep if k in r}
        try:
            out[mode] = c.sub("H3", mode, run)
        except EngineError as e:
            out[mode] = {"status": "not estimable", "reason": str(e)[:400]}
    full = out.get("full") or {}
    out["interpretation"] = ("M6 converged" if full.get("converged") else
                             "M6 not converged after the 4 x 8,000 refit: reported as 'not converged'")
    return out
