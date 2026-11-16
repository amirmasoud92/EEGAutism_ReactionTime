"""Registry of the sensitivity analyses of the task models (Additional file 1, Table S2) as declarative specs.

Every entry of :data:`REGISTRY` is a :class:`SensSpec`: the base model(s), how the analysis data differ from the
main model (predictor variant, analysis set, extra trial-table row flags, extra columns, a design transform, an
outcome transform), how it is fitted (``path``: the same simplification path as the main model; ``fixed_structure``: the
final random structure of the main model, as for the group models; ``cens``: brms censored model; ``srel``: the
reliability-calibration bootstrap; ``match``: within-child subsampling; ``het``: glmmTMB with group-specific residual
variance) and whether it is a robustness check of the coupling estimate (``qualifier``) or descriptive.

Use::

    md = build("S-W500", "M1", "conf", mode="real")
    md = attach_outcome(md, "S-W500")
    res = fit(md, "S-W500", out_dir, "s_w500_m1")
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modeling import knots as K
from modeling.model_specs import SESOI, RandomDef, get_model
from modeling.splines import ns_basis, ns_column_names

__all__ = ["SensSpec", "REGISTRY", "SPEC_IDS", "get", "build", "attach_outcome", "fit", "refinalise",
           "medication_class", "MEDICATION_CODING", "NOMINAL_TRIALS", "summary_row", "InputMissingError"]

#: Knot table of the models (repository ``configs/model_knots.yaml``).
KNOT_TABLE_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_knots.yaml"


class InputMissingError(RuntimeError):
    """A sensitivity analysis needs an input that is not available (a child covariate or a data-owner file)."""


@dataclass(frozen=True)
class SensSpec:
    id: str
    ref: str
    models: tuple[str, ...]
    change: str
    stage: str = ""                             # "external": needs a data-owner input that is not available
    variant: str = ""
    analysis_set: Mapping[str, str] = field(default_factory=dict)
    extra_row_flags: Mapping[str, Mapping] = field(default_factory=dict)
    extra_predictor_columns: tuple[str, ...] = ()
    extra_trial_columns: tuple[str, ...] | Mapping[str, tuple[str, ...]] = ()
    real_outcome_columns: tuple[str, ...] = ()
    rt_column: Mapping[str, str] = field(default_factory=dict)
    min_rows: int | None = None
    design: Callable | None = None              # (md, ctx) -> md, before the outcome
    outcome: Callable | None = None             # (md, ctx) -> md, after the outcome is attached
    fit_kind: str = "path"
    qualifier: bool = False
    descriptive: bool = False
    inputs: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    base_model: Mapping[str, str] = field(default_factory=dict)   # model -> the model whose data are built
    note: str = ""


# --------------------------------------------------------------------------- generic helpers
def _replace(md, frame=None, fixed=None, splines=None, mdef=None, outcome_column=None, meta=None, log=None):
    return replace(md, frame=md.frame if frame is None else frame,
                   fixed_terms=list(md.fixed_terms) if fixed is None else list(fixed),
                   splines=list(md.splines) if splines is None else list(splines),
                   model_def=md.model_def if mdef is None else mdef,
                   outcome_column=md.outcome_column if outcome_column is None else outcome_column,
                   meta={**md.meta, **(meta or {})}, log={**md.log, **(log or {})})


def _zwithin(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    s = pd.Series(v)
    mu = s.groupby(g).transform("mean")
    sd = s.groupby(g).transform(lambda x: x.std(ddof=1))
    return ((s - mu) / sd).to_numpy(float)


def refinalise(md, *, min_rows: int = 30, reason: str = ""):
    """After a row filter: re-apply the minimum-rows child rule, re-z-score ``t_run`` / ``tib`` within child over
    the new rows, rebuild their spline bases with the knot table and drop optional indicator terms that became
    constant."""
    f = md.frame
    n = f.groupby("pid")["pid"].transform("size")
    dropped = sorted(set(f.loc[n < min_rows, "pid"]))
    f = f.loc[n >= min_rows].reset_index(drop=True).copy()
    g = f["pid"].to_numpy()
    raw_t = f["t_run_raw"].to_numpy(float) if "t_run_raw" in f else f["trial_index_in_run"].to_numpy(float)
    f["t_run"] = _zwithin(raw_t, g)
    if "tib_raw" in f:
        f["tib"] = _zwithin(f["tib_raw"].to_numpy(float), g)
    for c in ("t_run", "tib"):
        if not np.isfinite(f[c].to_numpy(float)).all():
            raise ValueError(f"{c} undefined after the row filter (a child with one position)")
    for s in md.splines:
        if s["variable"] in ("t_run", "tib"):
            B = ns_basis(f[s["variable"]].to_numpy(float), s["interior_knots"], s["boundary_knots"])
            for j, c in enumerate(s["columns"]):
                f[c] = B[:, j]
    opt = set(md.model_def.optional)
    fixed = [t for t in md.fixed_terms
             if not (t in opt and np.nanstd(np.prod([f[p].to_numpy(float) for p in t.split(":")], axis=0)) == 0)]
    log = {"sensitivity_filter": {"reason": reason, "rows": int(len(f)), "children": int(f["pid"].nunique()),
                                  "children_dropped_lt_min_rows": len(dropped)}}
    return _replace(md, frame=f, fixed=fixed, log=log)


def _filter(md, mask: np.ndarray, reason: str, min_rows: int = 30):
    md2 = _replace(md, frame=md.frame.loc[np.asarray(mask, bool)].reset_index(drop=True))
    return refinalise(md2, min_rows=min_rows, reason=reason)


def _add(md, cols: Mapping[str, np.ndarray], terms: Sequence[str], *, splines: Sequence[Mapping] = ()):
    f = md.frame.copy()
    for k, v in cols.items():
        f[k] = v
    fixed = list(md.fixed_terms) + [t for t in terms if t not in md.fixed_terms]
    return _replace(md, frame=f, fixed=fixed, splines=list(md.splines) + list(splines))


def _drop_terms(md, terms: Sequence[str]):
    return _replace(md, fixed=[t for t in md.fixed_terms if t not in set(terms)])


def _missing_rule(f: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Missing-covariate rule: missing -> 0 plus an indicator (used for the S-OCU indices)."""
    v = pd.to_numeric(f[col], errors="coerce").to_numpy(float)
    miss = ~np.isfinite(v)
    return np.where(miss, 0.0, v), miss.astype(float)


def _bool(f: pd.DataFrame, col: str, fill: bool) -> np.ndarray:
    return pd.array(f[col], dtype="boolean").fillna(fill).to_numpy(bool)


def _knots(task_key: str, term: str) -> dict:
    with open(KNOT_TABLE_PATH, encoding="utf-8") as fh:
        kt = yaml.safe_load(fh)
    t = kt["models"][task_key]["terms"][term]
    return t


# --------------------------------------------------------------------------- design transforms
def _t_soa(md, ctx):
    m = md.frame["soa"].to_numpy(float) >= 2.2
    md = _filter(md, m, "S-SOA: SOA >= 2.2 s")
    c = md.frame["soa"].to_numpy(float) - 2.28125           # M1 knot-table median of soa
    return _add(md, {"soa_c": c}, ["a_wb:soa_c"])


def _t_lag(md, ctx):
    key = {"FAST": "S-LAG_FAST", "IC": "S-LAG_IC"}[md.task]
    t = _knots(key, "ns(win_start_lag, 3)")
    x = md.frame["soa"].to_numpy(float) + K.WINDOW_START_S[md.task]
    B = ns_basis(x, t["interior_knots"], t["boundary_knots"])
    cols = ns_column_names("win_start_lag", 3)
    sp = {"variable": "win_start_lag", "df": 3, "interior_knots": [float(k) for k in t["interior_knots"]],
          "boundary_knots": [float(k) for k in t["boundary_knots"]], "columns": cols, "knots_key": key}
    return _add(md, {"win_start_lag": x, **{c: B[:, j] for j, c in enumerate(cols)}}, cols, splines=[sp])


def _t_rap(md, ctx):
    f = md.frame
    g = f["pid"].to_numpy()
    lo = _zwithin(f["ap_2p5_5"].to_numpy(float), g)
    hi = _zwithin(f["ap_15_25"].to_numpy(float), g)
    ok = np.isfinite(lo) & np.isfinite(hi)
    md = _add(md, {"ap_lo_z": lo, "ap_hi_z": hi}, ["ap_lo_z", "ap_hi_z"])
    if not ok.all():
        md = _filter(md, ok, "R-AP: finite aperiodic-band power")
    return md


def _t_blink(md, ctx):
    f = md.frame
    m = ~_bool(f, "blink_in_window", True) & _bool(f, "valid_blink", False)
    return _filter(md, m, "S-BLINK: blink_in_window false and valid_blink true")


def _t_ocu(md, ctx, h2: bool = False):
    f = md.frame
    cz, cm = _missing_rule(f, "closure_z")
    gz, gm = _missing_rule(f, "gaze_dev_z")
    fb = _bool(f, "gaze_dev_fallback_child", False).astype(float)
    cols = {"closure_z": cz, "gaze_dev_z": gz, "gaze_dev_fallback_child": fb, "closure_missing": cm,
            "gaze_dev_missing": gm}
    terms = ["closure_z", "gaze_dev_z"]
    for ind in ("gaze_dev_fallback_child", "closure_missing", "gaze_dev_missing"):
        if np.nanstd(cols[ind]) > 0:
            terms.append(ind)
    if h2:
        terms += ["a_wb:closure_z", "a_wb:gaze_dev_z", "a_wb:closure_z:grp", "a_wb:gaze_dev_z:grp"]
    else:
        terms += ["a_wb:closure_z", "a_wb:heog_z", "a_wb:gaze_dev_z"]
    return _add(md, cols, terms)


#: The ocular covariates of the models and their missingness indicators (removed by S-NOOCU).
NOOCU_TERMS = ("heog_z", "vfp_z", "heog_missing", "vfp_missing")


def _t_noocu(md, ctx):
    """S-NOOCU (descriptive): M1 without any ocular covariate."""
    return _drop_terms(md, NOOCU_TERMS)


def _t_pc3_marginal(md, ctx):
    """PC3-marginal (descriptive): M1 without the ``ns(log_tsp, 4)`` spline and its join indicator."""
    drop = [c for s in md.splines if s.get("variable") == "log_tsp" for c in s["columns"]] + ["tsp_join"]
    return _replace(md, fixed=[t for t in md.fixed_terms if t not in set(drop)],
                    splines=[s for s in md.splines if s.get("variable") != "log_tsp"])


def _t_s1a(md, ctx):
    f = md.frame
    m = np.isclose(f["cVA"].to_numpy(float), -0.5) & np.isclose(f["cAV"].to_numpy(float), -1.0 / 3.0)
    md = _filter(md, m, "S1-A: auditory trials only")
    md = _drop_terms(md, ["cVA", "cAV", "a_wb:cVA", "a_wb:cAV"])
    return md


def _t_s1u(md, ctx):
    r = md.model_def.random
    nr = RandomDef(corr=r.corr, droppable=r.droppable, uncorrelated=r.uncorrelated, block=r.block, group=r.group,
                   block_factor=r.block_factor, start_corr=False)
    return _replace(md, mdef=replace(md.model_def, random=nr))


def _t_harm(md, ctx):
    f = md.frame.copy()
    h = pd.to_numeric(f["a_harm"], errors="coerce").to_numpy(float)
    f["a_wb_block_centred"] = f["a_wb"].to_numpy(float)
    f["a_wb"] = h
    md = _replace(md, frame=f, meta={"s_harm": "a_wb replaced by a_harm"})
    ok = np.isfinite(h)
    return md if ok.all() else _filter(md, ok, "S-HARM: finite a_harm")


def _t_ret(md, ctx):
    rf = ctx.get("retfrac")
    if rf is None:
        raise InputMissingError("S-RET needs the per-child retained fraction (sensitivity.retained_fraction)")
    v = md.frame["pid"].map(rf).astype(float).to_numpy()
    if not np.isfinite(v).all():
        raise ValueError("retained fraction missing for some children")
    kids = md.frame.groupby("pid")["pid"].first().index
    centre = float(np.mean([rf[k] for k in kids]))
    return _add(md, {"retfrac_c": v - centre}, ["a_wb:retfrac_c"])


#: S-ALT: rejection step that contributes most to a group difference in retained trials -> its alternative.
ALT_STEPS: dict[str, str] = {
    "edge_flag": "none (edge validity is a filter-validity requirement; no alternative)",
    "segment": "none (lookback and join rules define the trial; no alternative)",
    "bad_annotation": "BAD_icafit spans not applied (other BAD* annotations kept)",
    "n_roi": "bad ROI channels interpolated (MNE interpolate_bads) instead of dropped, so the n_roi < 5 rule "
             "cannot fire",
    "artefact": "ROI peak-to-peak threshold 300 uV instead of 200 uV",
    "child_min_rows": ">= 20 instead of >= 30 analysis rows per child x task",
    "nonperformance": "FAST / IC behavioural non-performance rule not applied",
}


def _t_alt(md, ctx):
    step = ctx.get("alt_step")
    if step is None:
        raise InputMissingError("S-ALT needs the rejection step chosen by the retention decomposition (alt_step)")
    if step not in ALT_STEPS:
        raise ValueError(f"unknown S-ALT rejection step {step!r}; one of {sorted(ALT_STEPS)}")
    if step in ("child_min_rows", "nonperformance"):
        return md                                    # handled at build time (min_rows = 20; no non-performance rule)
    if step in ("edge_flag", "segment"):
        raise InputMissingError(f"S-ALT for {step}: there is no alternative; not run")
    raise InputMissingError(f"S-ALT for {step} needs the feature pipeline re-run with its alternative setting "
                            f"(bad_annotation: BAD_icafit not applied; n_roi: interpolated ROI channels; artefact: "
                            f"300 uV); supply the variant table")


def _child_values(md, ctx, key: str) -> np.ndarray:
    tab = ctx.get(key)
    if tab is None:
        raise InputMissingError(f"needs the child covariate '{key}' (pid -> value)")
    return md.frame["pid"].map(tab).to_numpy()


def _t_fsiq(md, ctx):
    v = pd.to_numeric(pd.Series(_child_values(md, ctx, "fsiq")), errors="coerce").to_numpy(float)
    md = _filter(md, np.isfinite(v), "S-FSIQ: complete cases")
    v = pd.to_numeric(md.frame["pid"].map(ctx["fsiq"]), errors="coerce").to_numpy(float)
    kids = md.frame.groupby("pid")["pid"].first().index
    centre = float(np.mean([float(ctx["fsiq"][k]) for k in kids]))
    return _add(md, {"fsiq_c": v - centre}, ["a_wb:fsiq_c"])


def _t_fsiq_excl(md, ctx):
    v = pd.to_numeric(pd.Series(_child_values(md, ctx, "fsiq")), errors="coerce").to_numpy(float)
    return _filter(md, ~(np.isfinite(v) & (v < 80)), "S-FSIQ: exclude FSIQ < 80")


def _t_med(md, ctx, strict: bool = False):
    med = ctx.get("medication")
    if med is None:
        raise InputMissingError("S-MED needs the medication strings (pid -> (medication, medication_dup))")
    cls = {k: medication_class(*(v if isinstance(v, (tuple, list)) else (v, None))) for k, v in med.items()}
    c = md.frame["pid"].map(cls).to_numpy()
    bad = np.isin(c, list(CNS_ACTIVE)) | (strict & (c == "unknown"))
    return _filter(md, ~bad, "S-MED2" if strict else "S-MED")


def _t_nom(md, ctx):
    f = md.frame
    if md.task == "IC":
        if "beyond_nominal_contours_child" not in f:
            raise KeyError("S-NOM (IC) needs beyond_nominal_contours_child")
        m = ~_bool(f, "beyond_nominal_contours_child", True)
    else:
        m = pd.to_numeric(f["trial_index_in_child"], errors="coerce").to_numpy(float) <= NOMINAL_TRIALS[md.task]
    return _filter(md, m, f"S-NOM: first {NOMINAL_TRIALS[md.task]} nominal trials")


def _t_min(md, ctx):
    return refinalise(md, min_rows={"AVSRT": 100, "FAST": 60, "IC": 60}[md.task], reason="S-MIN floor")


# --------------------------------------------------------------------------- outcome transforms (after y exists)
def _o_inv(md, ctx):
    f = md.frame.copy()
    f["irt"] = -np.exp(-f["lrt"].to_numpy(float))          # -1000 / RT(ms) = -1 / RT(s)
    return _replace(md, frame=f, outcome_column="irt", mdef=replace(md.model_def, sesoi={}))


def _o_prt(md, ctx):
    """``prev_lrt``: child-centred log RT of the previous stimulus of the run (0 + ``prev_lrt_undef`` after a miss,
    at the first stimulus or when the previous stimulus has no RT). Real mode: the trial-table ``prev_rt_ms``;
    synthetic mode: the synthetic outcome of the immediately preceding row when it is the previous trial."""
    f = md.frame.copy()
    if md.outcome_kind == "real":
        prt = pd.to_numeric(f["real_prev_rt_ms"], errors="coerce").to_numpy(float)
        ok = np.isfinite(prt) & (prt > 0) & (f["prev_outcome"].astype("string") == "hit").fillna(False).to_numpy()
        pl = np.where(ok, np.log(np.where(ok, prt, 1.0) / 1000.0), np.nan)
    else:
        o = f.sort_values(["pid", "run", "trial_index_in_run"]).index
        g = f.loc[o]
        prev_ok = ((g["pid"].to_numpy()[1:] == g["pid"].to_numpy()[:-1])
                   & (g["run"].to_numpy()[1:] == g["run"].to_numpy()[:-1])
                   & (g["trial_index_in_run"].to_numpy()[1:] - g["trial_index_in_run"].to_numpy()[:-1] == 1))
        vals = np.full(len(g), np.nan)
        vals[1:] = np.where(prev_ok, g[md.outcome_column].to_numpy(float)[:-1], np.nan)
        pl = np.empty(len(f))
        pl[np.asarray(o)] = vals
    undef = ~np.isfinite(pl)
    cm = pd.Series(np.where(undef, np.nan, pl)).groupby(f["pid"].to_numpy()).transform("mean").to_numpy()
    f["prev_lrt"] = np.where(undef, 0.0, pl - cm)
    f["prev_lrt_undef"] = undef.astype(float)
    fixed = list(md.fixed_terms) + ["prev_lrt"] + (["prev_lrt_undef"] if 0 < undef.mean() < 1 else [])
    return _replace(md, frame=f, fixed=fixed)


def _o_cens(md, ctx):
    """S-CENS: right-censoring at the response-window end ``min(1.5 s, time to the next stimulus)``. Real mode: hits
    with RT >= 150 ms are observed, valid-window misses censored, hits with RT < 150 ms dropped; synthetic mode: the
    latent synthetic log RT is censored where it exceeds the window end."""
    f = md.frame.copy()
    # no next stimulus (last stimulus of a run): the window is bounded by 1.5 s only (np.fmin ignores NaN)
    wend = np.fmin(1.5, pd.to_numeric(f["t_to_bound_ms"], errors="coerce").to_numpy(float) / 1000.0)
    cap = np.log(wend)
    if md.outcome_kind == "real":
        rt = f["rt_ms"].to_numpy(float)
        hit = f["responded"].astype(bool).to_numpy()
        keep = ~(hit & (rt < 150))
        f = f.loc[keep].reset_index(drop=True)
        cap, hit, rt = cap[keep], hit[keep], rt[keep]
        f["cens"] = (~hit).astype(float)
        f["lrt"] = np.where(hit, np.log(np.where(hit, rt, 1.0) / 1000.0), cap)
    else:
        y = f["lrt"].to_numpy(float)
        f["cens"] = (y > cap).astype(float)
        f["lrt"] = np.minimum(y, cap)
    if not np.isfinite(f["lrt"]).all():
        raise ValueError("S-CENS: non-finite censoring times")
    return _replace(md, frame=f, outcome_column="lrt", meta={"s_cens": {"n_censored": int(f["cens"].sum())}})


def _o_omit(md, ctx):
    if md.outcome_kind == "real":
        f = md.frame.copy()
        f["miss"] = (pd.to_numeric(f["real_n_press_until_bound"], errors="coerce").to_numpy(float) == 0).astype(float)
        return _replace(md, frame=f)
    return md                                    # synthetic: the simulated miss stands for "no press at all"


def lapse_l1(miss: np.ndarray, rt_ms: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """S-L1 lapse: miss, or RT > child mean + 2 SD (mean, SD ddof 1 over the child's hits on the model rows)."""
    hit = ~np.asarray(miss, bool)
    r = pd.Series(np.where(hit, rt_ms, np.nan))
    mu = r.groupby(pid).transform("mean").to_numpy()
    sd = r.groupby(pid).transform(lambda x: x.std(ddof=1)).to_numpy()
    return (~hit | (hit & (np.asarray(rt_ms, float) > mu + 2 * sd))).astype(float)


def _o_l1(md, ctx):
    if md.outcome_kind == "real":
        f = md.frame.copy()
        f["miss"] = lapse_l1(~f["responded"].astype(bool).to_numpy(), f["rt_ms"].to_numpy(float),
                             f["pid"].to_numpy())
        return _replace(md, frame=f)
    return md


# --------------------------------------------------------------------------- medication coding
MEDICATION_CODING: dict[str, str] = {
    "Methylphenidate": "stimulant", "Concerta 75mg": "stimulant", "Focalin 5 mg": "stimulant", "Focalin": "stimulant",
    "ADHD Meds": "stimulant", "Sertraline 2ml": "ssri", "Stimulant + SSRI": "stimulant_ssri",
    "Zoloft, Adderall": "stimulant_ssri", "Vyvanse, Sertraline": "stimulant_ssri",
    "Albuterol for asthma, Diphenhydramine for eczema": "sedating_antihistamine",
    "Albuterol 2.5 mg/3 ml sol,ventolyn 90mcg, claritin 10mg": "non_psychotropic",
    "Asthma inhaler": "non_psychotropic", "Meds for Allergy": "non_psychotropic",
    "Oxybutynin 10mg 1x a day": "non_psychotropic", "Polyethelene Laxative": "non_psychotropic",
    "No": "none", "n/a": "missing", "Unknown": "unknown",
}
CNS_ACTIVE = ("stimulant", "ssri", "stimulant_ssri", "sedating_antihistamine")


def medication_class(original, dup=None) -> str:
    """Coding of the 18 medication strings of participants.tsv; ``_dup`` only where the original is missing (n/a,
    Unknown, empty). An unlisted string raises."""
    def norm(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return str(v).strip()

    o, d = norm(original), norm(dup)
    if o in (None, "", "n/a", "Unknown") and d not in (None, "", "n/a"):
        o = d
    if o in (None, ""):
        return "missing"
    if o not in MEDICATION_CODING:
        raise KeyError(f"medication string {o!r} is not one of the 18 coded strings")
    return MEDICATION_CODING[o]


NOMINAL_TRIALS = {"AVSRT": 400, "FAST": 720, "IC": 540}

_ALL_RT = ("M1", "M3", "M4")
_AV = {"AVSRT": "s_w500_rt"}


def _rt_flags(flag: str) -> Mapping:
    return {"require_true": [flag]}


# --------------------------------------------------------------------------- the registry
REGISTRY: dict[str, SensSpec] = {}


def _reg(s: SensSpec) -> None:
    if s.id in REGISTRY:
        raise ValueError(f"duplicate sensitivity id {s.id}")
    REGISTRY[s.id] = s


_reg(SensSpec("S-W500", "Additional file 1, Table S2", ("M1",), "AVSRT window 500 ms [-550, -50), L 1000", variant="s_w500",
              analysis_set={"AVSRT": "s_w500_rt"}, qualifier=True))
_reg(SensSpec("S-W1000", "Additional file 1, Table S2", ("M1",), "AVSRT window 1000 ms [-1050, -50), L 1200",
              variant="s_w1000", analysis_set={"AVSRT": "s_w1000_rt"}))
_reg(SensSpec("S-W400", "Additional file 1, Table S2", ("M3", "M4"), "FAST and IC window 400 ms [-450, -50)",
              variant="s_w400", analysis_set={"FAST": "s_w400_rt", "IC": "s_w400_rt"}))
_reg(SensSpec("S-SOA", "Additional file 1, Table S2", ("M1",), "AVSRT trials with SOA >= 2.2 s, plus soa x a_wb",
              design=_t_soa, qualifier=True, note="soa centred at the M1 knot-table median 2.28125 s"))
_reg(SensSpec("S-LAG", "Additional file 1, Table S2", ("M3", "M4"),
              "FAST / IC: + ns(win_start_lag, 3) (time from the previous onset to the window start)",
              design=_t_lag, note="knots S-LAG_FAST / S-LAG_IC of configs/model_knots.yaml; not in AVSRT"))
_reg(SensSpec("S-PRESS", "Additional file 1, Table S2", ("M1",), "AVSRT windows >= 1 s after the previous press",
              analysis_set={"AVSRT": "s_press_rt"}, qualifier=True))
_reg(SensSpec("S-ERP", "Additional file 1, Table S2", ("M3", "M4"), "FAST / IC alpha after subtracting the previous-stimulus "
              "evoked response", variant="s_erp"))
_reg(SensSpec("S-HH", "Additional file 1, Table S2", _ALL_RT, "history-homogeneous trials only",
              analysis_set={"AVSRT": "s_hh_rt", "FAST": "s_hh_rt", "IC": "s_hh_rt"}, qualifier=True))
_reg(SensSpec("S-PRT", "Additional file 1, Table S2", _ALL_RT, "+ prev_lrt (child-centred previous log RT; 0 + indicator)",
              real_outcome_columns=("prev_rt_ms",), outcome=_o_prt,
              note="reads the trial-table prev_rt_ms"))
_reg(SensSpec("S-EST1", "Additional file 1, Table S2", _ALL_RT, "Welch estimator (Hann, 256 samples, 50% overlap)",
              variant="s_est1", qualifier=True))
_reg(SensSpec("S-EST2", "Additional file 1, Table S2", _ALL_RT, "4-Hz bandwidth multitaper", variant="s_est2"))
_reg(SensSpec("S-BAND", "Additional file 1, Table S2", _ALL_RT, "fixed 8-12 Hz band", variant="s_band"))
_reg(SensSpec("R-AP", "Additional file 1, Table S2", _ALL_RT, "+ log power 2.5-5 Hz and 15-25 Hz",
              extra_predictor_columns=("ap_2p5_5", "ap_15_25"), design=_t_rap, qualifier=True,
              note="both z-scored within child over the model rows (ap_lo_z, ap_hi_z)"))
_reg(SensSpec("S-ICA", "Additional file 1, Table S2", _ALL_RT, "alpha without ICA ocular correction (same window rules)",
              variant="s_ica", qualifier=True))
_reg(SensSpec("S-ICA-H2", "Additional file 1, Table S2", ("M1-H2",), "M1-H2 with the S-ICA predictor",
              variant="s_ica", fit_kind="fixed_structure"))
_reg(SensSpec("S-BLINK", "Additional file 1, Table S2", ("M1",), "exclude windows with a blink or without a valid blink "
              "check", extra_predictor_columns=("blink_in_window", "valid_blink"), design=_t_blink, qualifier=True))
_reg(SensSpec("S-BLINK-H2", "Additional file 1, Table S2", ("M1-H2",), "M1-H2 on the S-BLINK trial set",
              extra_predictor_columns=("blink_in_window", "valid_blink"), design=_t_blink,
              fit_kind="fixed_structure"))
_reg(SensSpec("S-OCU", "Additional file 1, Table S2", ("M1",),
              "+ closure_z + gaze_dev_z + gaze_dev_fallback_child + a_wb:(closure_z + heog_z + gaze_dev_z)",
              extra_predictor_columns=("closure_z", "gaze_dev_z", "gaze_dev_fallback_child"), design=_t_ocu,
              descriptive=True, note="missing closure_z / gaze_dev_z: 0 + indicator"))
_reg(SensSpec("S-OCU-H2", "Additional file 1, Table S2", ("M1-H2",),
              "M1-H2 + the S-OCU indices + a_wb:(closure_z + gaze_dev_z) and their :grp terms",
              extra_predictor_columns=("closure_z", "gaze_dev_z", "gaze_dev_fallback_child"),
              design=lambda md, ctx: _t_ocu(md, ctx, h2=True), descriptive=True, fit_kind="fixed_structure"))
_reg(SensSpec("S-NOOCU", "Additional file 1, Table S2", ("M1",),
              "M1 without the ocular covariates heog_z, vfp_z (and their missingness indicators)",
              design=_t_noocu, descriptive=True, note="descriptive"))
_reg(SensSpec("PC3-MARGINAL", "", ("M1",),
              "M1 without ns(log_tsp, 4) (and tsp_join); target: the SOA contrast of the behavioural model check",
              design=_t_pc3_marginal, descriptive=True, note="descriptive"))
_reg(SensSpec("S-GAZE", "", ("M1", "M3"), "exclude owner-flagged gaze-rejection trials",
              stage="external", inputs=("gaze_flags",),
              note="only if the data owners provide per-trial gaze-rejection flags; not available"))
_reg(SensSpec("S1-A", "", ("M2",), "M2 on auditory trials only, without the modality terms",
              design=_t_s1a))
_reg(SensSpec("S1-U", "Additional file 1, Table S2", ("M2",), "M2 with uncorrelated (1 | pid) + (0 + a_wb | pid)",
              design=_t_s1u, note="also called S1-RE; the path starts from the || form, so the decorrelation "
                                     "step cannot fire"))
_reg(SensSpec("S-MIS", "Additional file 1, Table S2", ("M1",), "exclude (prev_trial_no_press and RT < 200 ms) and presses "
              "after an anticipation", analysis_set={"AVSRT": "sens_misattribution"}))
for _id, _m, _flag, _rtc, _txt in (
        ("S-RT-100-1000", ("M1",), "responded_sens_100_1000", "rt_ms_sens_100_1000", "AVSRT RT 100-1000 ms"),
        ("S-RT-100-2000", ("M1",), "responded_sens_100_2000", "rt_ms_sens_100_2000", "AVSRT RT 100-2000 ms"),
        ("S-RT-OWN-100-1500", ("M3", "M4"), "responded_sens_100_1500", "rt_ms_sens_100_1500",
         "FAST / IC owners' 100-1500 ms"),
        ("S-RT-FAST-150-1200", ("M3",), "responded_sens_150_1200", "rt_ms_sens_150_1200", "FAST 150-1200 ms"),
        ("S-RT-IC-200-1000", ("M4",), "responded_sens_200_1000", "rt_ms_sens_200_1000", "IC 200-1000 ms")):
    _reg(SensSpec(_id, "Additional file 1, Table S2", _m, _txt,
                  analysis_set={t: "predictor_set" for t in ("AVSRT", "FAST", "IC")},
                  extra_row_flags={t: _rt_flags(_flag) for t in ("AVSRT", "FAST", "IC")},
                  rt_column={t: _rtc for t in ("AVSRT", "FAST", "IC")},
                  note="rows: predictor set and a hit in the window (no 150-ms floor unless the window has it)"))
_reg(SensSpec("S-RT-INV", "Additional file 1, Table S2", _ALL_RT, "inverse-RT outcome -1000 / RT",
              outcome=_o_inv, note="no SESOI on this scale; sign and CI reported"))
_reg(SensSpec("S-CENS", "Additional file 1, Table S2", ("M1",),
              "valid-window misses right-censored at the window end; brms cens(); M1 fixed part, (1 + a_wb | pid) "
              "+ child time slopes", analysis_set={"AVSRT": "predictor_set"}, extra_trial_columns=("t_to_bound_ms",),
              outcome=_o_cens, fit_kind="cens", qualifier=True,
              note="window end = min(1.5 s, next stimulus); 4 chains x 4,000 iterations, warmup 2,000; "
                   "flat priors on b"))
_reg(SensSpec("S-OMIT", "Additional file 1, Table S2", ("M2",), "miss = no press at all before the next stimulus",
              real_outcome_columns=("n_press_until_bound",), outcome=_o_omit))
_reg(SensSpec("S-L1", "Additional file 1, Table S2", ("M2",), "lapse rule: miss or RT > child mean + 2 SD",
              outcome=_o_l1, note="mean and SD over the child's hits on the model rows, in ms"))
_reg(SensSpec("S-REL", "Additional file 1, Table S2", ("M1", "M3", "M4"), "regression calibration with child "
              "reliability; child bootstrap B = 499, seed 20261020, final random structure of the main model",
              fit_kind="srel"))
_reg(SensSpec("S-REL-H2", "Additional file 1, Table S2", ("M1-H2",), "S-REL on M1-H2", fit_kind="srel"))
_reg(SensSpec("S-DTOST", "", ("M1",), "TOST of the S-REL slope against +-0.0102 (true-score "
              "scale), child-bootstrap 90% CI", fit_kind="srel", note="computed from the S-REL bootstrap"))
_reg(SensSpec("S-HARM", "Additional file 1, Table S2", ("M1", "M3", "M4", "M5h"), "a_harm (+-60-s running mean "
              "within the run) in place of a_wb", extra_predictor_columns=("a_harm",), design=_t_harm,
              qualifier=True, note="a_slow kept as the block-mean covariate"))
_reg(SensSpec("S-HET", "Additional file 1, Table S2", ("M1-H2",), "glmmTMB, dispformula ~ grp, group-specific a_wb "
              "slope variances; CR2 of the group difference", fit_kind="het"))
_reg(SensSpec("S-MATCH", "Additional file 1, Table S2", ("M1", "M3", "M4"), "within-child random "
              "subsampling to the pooled per-child target count (SMATCH_TARGET_COUNT; children with fewer rows "
              "keep all); 20 repetitions; seed 20261020; median estimate and 2.5 / 97.5 percentiles over the "
              "repetitions; targets beta(a_wb) and beta(a_wb:grp)",
              inputs=("target_counts",), fit_kind="match",
              note="target_counts = smatch_target_counts(md) (pooled behaviour-only counts per task)"))
_reg(SensSpec("S-RET", "Additional file 1, Table S2", _ALL_RT, "+ a_wb:retfrac_c (child retained-trial fraction)",
              design=_t_ret, inputs=("retfrac",),
              note="retained fraction = valid windows / predictor-set stimuli of the child x task, centred at the "
                   "mean over the model's children"))
_reg(SensSpec("S-ALT", "Additional file 1, Table S2", _ALL_RT, "the alternative of the rejection step that contributes most "
              "to a group difference in retained trials", inputs=("alt_step",),
              design=_t_alt, note="alternatives: ALT_STEPS; 'child_min_rows' "
                   "(>= 20 rows) and 'nonperformance' run directly; bad_annotation / n_roi / artefact need a "
                   "feature re-run"))
_reg(SensSpec("S-FSIQ", "Additional file 1, Table S2", ("M1",), "+ a_wb:fsiq_c (complete cases)", inputs=("fsiq",),
              design=_t_fsiq))
_reg(SensSpec("S-FSIQ-EXCL", "Additional file 1, Table S2", ("M1",), "exclude FSIQ < 80", inputs=("fsiq",),
              design=_t_fsiq_excl))
_reg(SensSpec("S-MED", "Additional file 1, Table S2", ("M1",), "exclude CNS-active medication",
              inputs=("medication",), design=_t_med))
_reg(SensSpec("S-MED2", "Additional file 1, Table S2", ("M1",), "S-MED and exclude 'Unknown'",
              inputs=("medication",), design=lambda md, ctx: _t_med(md, ctx, strict=True)))
_reg(SensSpec("S-NOM", "Additional file 1, Table S2", _ALL_RT, "cap at the nominal trial count (AVSRT 400, FAST 720, IC 540)",
              extra_trial_columns={"AVSRT": ("trial_index_in_child",), "FAST": ("trial_index_in_child",),
                                   "IC": ("beyond_nominal_contours_child",)}, design=_t_nom,
              note="IC: the 540-contour cap applies to the main model already; S-NOM uses "
                   "beyond_nominal_contours_child"))
_reg(SensSpec("S-MIN", "Additional file 1, Table S2", _ALL_RT, "floor of >= 100 (AVSRT) / >= 60 (FAST, IC) analysis trials",
              design=_t_min))
_reg(SensSpec("S-ROI", "Additional file 1, Table S2", _ALL_RT, "owners' P-THETA 7-channel ROI", variant="s_roi"))
_reg(SensSpec("S-IAF", "Additional file 1, Table S2", _ALL_RT, "IAF from main-day rest only", variant="s_iaf"))

#: Sensitivity-analysis labels (each maps to one or more registry entries).
SPEC_IDS: dict[str, tuple[str, ...]] = {
    "S-W500": ("S-W500",), "S-W1000": ("S-W1000",), "S-W400": ("S-W400",), "S-SOA": ("S-SOA",), "S-LAG": ("S-LAG",),
    "S-PRESS": ("S-PRESS",), "S-ERP": ("S-ERP",), "S-HH": ("S-HH",), "S-PRT": ("S-PRT",), "S-EST1": ("S-EST1",),
    "S-EST2": ("S-EST2",), "S-BAND": ("S-BAND",), "R-AP": ("R-AP",), "S-ICA": ("S-ICA",), "S-ICA-H2": ("S-ICA-H2",),
    "S-BLINK": ("S-BLINK",), "S-BLINK-H2": ("S-BLINK-H2",), "S-OCU": ("S-OCU",), "S-OCU-H2": ("S-OCU-H2",),
    "S-NOOCU": ("S-NOOCU",), "S-GAZE": ("S-GAZE",), "S1-A": ("S1-A",), "S1-U": ("S1-U",), "S-MIS": ("S-MIS",),
    "S-RT": ("S-RT-100-1000", "S-RT-100-2000", "S-RT-OWN-100-1500", "S-RT-FAST-150-1200", "S-RT-IC-200-1000",
             "S-RT-INV"),
    "S-CENS": ("S-CENS",), "S-OMIT": ("S-OMIT",), "S-L1": ("S-L1",), "S-REL": ("S-REL", "S-REL-H2"),
    "S-DTOST": ("S-DTOST",), "S-HARM": ("S-HARM",), "S-HET": ("S-HET",), "S-MATCH": ("S-MATCH",),
    "S-RET": ("S-RET",), "S-ALT": ("S-ALT",), "S-FSIQ": ("S-FSIQ", "S-FSIQ-EXCL"), "S-MED": ("S-MED",),
    "S-MED2": ("S-MED2",), "S-NOM": ("S-NOM",), "S-MIN": ("S-MIN",), "S-ROI": ("S-ROI",), "S-IAF": ("S-IAF",),
}
ALIASES = {"S1-RE": "S1-U"}


def get(spec_id: str) -> SensSpec:
    sid = ALIASES.get(spec_id, spec_id)
    try:
        return REGISTRY[sid]
    except KeyError:
        raise KeyError(f"unknown sensitivity analysis {spec_id!r}") from None


# --------------------------------------------------------------------------- build / outcome / fit
def retained_fraction(task: str, cohort: str, *, cfg=None, pseudonymizer) -> dict[str, float]:
    """S-RET input: per child, valid windows / predictor-set stimuli (primary window; predictor-only)."""
    if cfg is None:
        from utils.config import load_project_config

        cfg = load_project_config()
    coh = str(cohort).strip().lower()
    if coh not in ("dev", "conf"):
        raise ValueError(f"cohort must be 'dev' or 'conf', not {cohort!r}")
    path = cfg.paths.derivatives / "features" / task / f"{task}_{coh}_features_predictor.parquet"
    p = pd.read_parquet(path, columns=["subject", "in_predictor_set", "valid"])
    ps = _bool(p, "in_predictor_set", False)
    v = _bool(p, "valid", False)
    q = pd.DataFrame({"subject": p["subject"], "ps": ps, "ok": ps & v})
    r = q.groupby("subject").apply(lambda g: g["ok"].sum() / max(1, g["ps"].sum()), include_groups=False)
    return {pseudonymizer(s): float(x) for s, x in r.items()}


__all__.append("retained_fraction")


def build(spec_id: str, model: str, cohort: str, *, cfg=None, ctx: Mapping | None = None, mode: str = "design",
          **kw):
    """Analysis data of sensitivity ``spec_id`` for ``model`` (design only unless ``mode="real"``)."""
    from modeling.model_data import build_model_data

    spec = get(spec_id)
    if model not in spec.models:
        raise ValueError(f"{spec.id} applies to {spec.models}, not {model}")
    if model == "M5h":
        raise ValueError("M5h is combined from the M1 / M3 / M4 sensitivity data (model_data.combine_m5h)")
    if spec.stage == "external":
        raise InputMissingError(f"{spec.id}: {spec.note}")
    ctx = dict(ctx or {})
    for need in spec.inputs:
        if need not in ctx:
            raise InputMissingError(f"{spec.id} needs the input {need!r}")
    mdef = get_model(model)
    task = mdef.task
    min_rows = spec.min_rows or (20 if (spec.id == "S-ALT" and ctx.get("alt_step") == "child_min_rows") else 30)
    etc = spec.extra_trial_columns
    etc = tuple(etc.get(task, ())) if isinstance(etc, Mapping) else tuple(etc)
    if mode == "real" and spec.fit_kind == "cens":
        kw.setdefault("allow_missing_rt", True)
    if spec.id == "S-ALT" and ctx.get("alt_step") == "nonperformance":
        kw.setdefault("apply_nonperformance", False)
    md = build_model_data(
        model, cohort, analysis_set=spec.analysis_set.get(task), variant=spec.variant, cfg=cfg,
        outcome="real" if mode == "real" else "none", extra_predictor_columns=spec.extra_predictor_columns,
        extra_row_flags=spec.extra_row_flags.get(task), extra_trial_columns=etc,
        real_outcome_columns=spec.real_outcome_columns if mode == "real" else (), min_rows=min_rows,
        rt_column=spec.rt_column.get(task, "rt_ms"), **kw)
    md = _replace(md, meta={"sensitivity": {"id": spec.id, "ref": spec.ref, "change": spec.change}})
    if spec.design is not None:
        md = spec.design(md, ctx)
    return md


def attach_outcome(md, spec_id: str, *, ctx: Mapping | None = None):
    """Apply the outcome transform of ``spec_id`` to data built with ``mode="real"``."""
    spec = get(spec_id)
    if md.outcome_kind != "real":
        raise ValueError("no outcome: build with mode='real'")
    if spec.outcome is not None:
        md = spec.outcome(md, dict(ctx or {}))
    return md


def fit(md, spec_id: str, out_dir, label: str, *, cfg=None, primary_state: Mapping | None = None,
        ctx: Mapping | None = None, workers: int = 8, **kw) -> dict:
    """Fit sensitivity ``spec_id`` on ``md`` (outcome attached); returns the engine (or extras) result with a
    ``sensitivity`` block."""
    from modeling import extras_bridge, r_bridge

    spec = get(spec_id)
    ctx = dict(ctx or {})
    kind = spec.fit_kind
    if kind == "path":
        res = r_bridge.fit(md, out_dir, label, second_impl=False, **kw)
    elif kind == "fixed_structure":
        if primary_state is None:
            raise ValueError(f"{spec.id} refits the final structure of the main model: pass primary_state")
        res = r_bridge.fit(md, out_dir, label, reduction=False, fixed_state=primary_state,
                           second_impl=False, **kw)
    elif kind == "cens":
        res = extras_bridge.s_cens(md, out_dir, label, random_string="(1 + a_wb | pid) + (0 + t_run | pid) + "
                                   "(0 + tib | pid)", **kw)
    elif kind == "het":
        res = _fit_het(md, out_dir, label, primary_state)
    elif kind == "srel":
        res = _fit_srel(md, out_dir, label, cfg=cfg, primary_state=primary_state, ctx=ctx, workers=workers, **kw)
    elif kind == "match":
        res = _fit_match(md, out_dir, label, cfg=cfg, ctx=ctx, workers=workers)
    else:
        raise ValueError(kind)
    res["sensitivity"] = {"id": spec.id, "ref": spec.ref, "change": spec.change, "stage": spec.stage,
                          "qualifier": spec.qualifier, "descriptive": spec.descriptive, "fit_kind": kind,
                          "note": spec.note}
    return res


def _fit_het(md, out_dir, label, primary_state):
    from modeling import extras_bridge

    if "grp" not in md.frame.columns:
        raise InputMissingError("S-HET needs grp (conf cohort)")
    f = md.frame.copy()
    g = f["grp"].to_numpy(float)
    f["a_wb_gtd"] = f["a_wb"].to_numpy(float) * (g < 0)
    f["a_wb_gasd"] = f["a_wb"].to_numpy(float) * (g > 0)
    st = primary_state or {"block": True}
    parts = ["(1 | pid)", "(0 + a_wb_gtd | pid)", "(0 + a_wb_gasd | pid)", "(0 + t_run | pid)", "(0 + tib | pid)"]
    if st.get("block", True):
        parts.append("(1 | pid_run_block)")
    tg = dict(md.model_def.grp_targets) or {"H2": {"a_wb:grp": 1.0}}
    return extras_bridge.s_het(_replace(md, frame=f), out_dir, label, random_string=" + ".join(parts), targets=tg)


def _fit_srel(md, out_dir, label, *, cfg, primary_state, ctx, workers, B=None, seed=None):
    from pathlib import Path

    from modeling import r_bridge, srel

    if primary_state is None:
        raise ValueError("S-REL refits the final random structure of the main model: pass primary_state")
    windows = ctx.get("rel_windows")
    if windows is None:
        windows = srel.load_rel_windows(md.task, md.cohort, cfg=cfg, pseudonymizer=ctx["pseudonymizer"])
    tgt = next(iter(md.model_def.targets)) if md.model_def.targets else "H2"
    out = Path(out_dir)

    def _fit(m, lab):
        return r_bridge.fit(m, out / label, lab, reduction=False, fixed_state=primary_state, cr2=False,
                            second_impl=False)

    res = srel.srel_bootstrap(md, windows, fit=_fit, target=tgt, B=B or srel.B_DEFAULT, seed=seed or srel.SEED,
                              workers=workers)
    res["dtost"] = srel.dtost(res, SESOI.get(md.task, 0.0102))
    res["targets"] = {tgt: {"estimate": res["estimate"], "se": res["bootstrap_sd"], "df": None,
                            "ci95": res["ci95_percentile"], "ci90": res["ci90_percentile"]}}
    return res


#: S-MATCH target count per task: the median over all children (pooled over cohorts and groups) of the per-child
#: number of RT analysis-set trials, from behaviour-only data (the ``primary_rt`` subset of the trial tables, children
#: with >= 30 such trials, FAST / IC after the non-performance rule; the EEG window rules not applied), rounded down.
SMATCH_TARGET_COUNT: dict[str, int] = {"AVSRT": 313, "FAST": 134, "IC": 95}
SMATCH_REPS = 20
SMATCH_SEED = 20261020


def smatch_target_counts(md) -> dict[str, int]:
    """S-MATCH input: every child of ``md`` -> the target count of its task (:data:`SMATCH_TARGET_COUNT`)."""
    n = SMATCH_TARGET_COUNT[md.task]
    return {k: n for k in sorted(md.frame["pid"].unique())}


__all__ += ["SMATCH_TARGET_COUNT", "smatch_target_counts"]


def _fit_match(md, out_dir, label, *, cfg, ctx, workers, reps: int = SMATCH_REPS, seed: int = SMATCH_SEED):
    """S-MATCH: ``reps`` within-child random subsamples (a child with more rows than its target is
    subsampled to the target without replacement; a child with fewer keeps all rows), each refitted with the full
    reduction path; per target the median estimate and the 2.5 / 97.5 percentiles over the repetitions. Targets: the
    model's first target (P1 / S2 / S3 = beta(a_wb)) and, when the model has ``a_wb:grp``, its group-difference target."""
    from concurrent.futures import ThreadPoolExecutor

    from modeling import r_bridge

    target = ctx["target_counts"]                       # pid -> target number of rows
    rng = np.random.default_rng(seed)
    f = md.frame
    by = {k: np.flatnonzero(f["pid"].to_numpy() == k) for k in f["pid"].unique()}
    draws = []
    for _ in range(reps):
        keep = []
        for k, idx in by.items():
            n = int(min(len(idx), target.get(k, len(idx))))
            keep.append(np.sort(rng.choice(idx, n, replace=False)))
        draws.append(np.concatenate(keep))
    n_sub = int(sum(len(idx) > target.get(k, len(idx)) for k, idx in by.items()))
    tgts = [next(iter(md.model_def.targets))]
    if "a_wb:grp" in md.fixed_terms:
        tgts += [t for t in md.model_def.grp_targets if t not in tgts]

    def one(r):
        m = refinalise(_replace(md, frame=f.iloc[draws[r]].reset_index(drop=True)), reason="S-MATCH subsample")
        res = r_bridge.fit(m, out_dir, f"{label}_rep{r + 1:02d}", second_impl=False, cr2=False)
        return {t: (res["targets"][t]["estimate"], res["targets"][t]["se"]) for t in tgts}

    with ThreadPoolExecutor(max(1, workers)) as ex:
        est = list(ex.map(one, range(reps)))
    out_t, all_e = {}, {}
    for t in tgts:
        e = np.array([x[t][0] for x in est], float)
        all_e[t] = e.tolist()
        out_t[t] = {"estimate": float(np.median(e)), "se": float(np.median([x[t][1] for x in est])), "df": None,
                    "rep_percentiles_2.5_97.5": [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]}
    return {"targets": out_t, "estimates": all_e[tgts[0]], "estimates_by_target": all_e, "reps": reps,
            "seed": seed, "n_children_subsampled": n_sub, "n_children": len(by),
            "note": "median over the repetitions; the 2.5 / 97.5 percentiles describe subsampling variability"}


def summary_row(spec_id: str, model: str, res: Mapping) -> dict:
    spec = get(spec_id)
    out = {"id": spec.id, "ref": spec.ref, "model": model, "fit_kind": spec.fit_kind, "qualifier": spec.qualifier,
           "descriptive": spec.descriptive}
    for k, t in (res.get("targets") or {}).items():
        out[k] = {"estimate": t.get("estimate"), "se": t.get("se"), "df": t.get("df"), "ci95": t.get("ci95")}
    for k in ("flag", "nobs", "max_rhat", "min_bulk_ess", "n_censored", "converged"):
        if k in res:
            out[k] = res[k]
    if "final_state" in res and isinstance(res["final_state"], Mapping):
        out["final_structure"] = res["final_state"].get("re")
    return out
