"""Model registry: fixed-effects formulas, starting random structures (protected and droppable terms), target
coefficients and contrasts, and the JSON model spec handed to the R engine.

Every formula is written over numeric columns of the model-input table built by :mod:`modeling.model_data` (factors
are expanded there to sum-contrast columns; spline terms are the ``ns_<x>_<df>_<k>`` basis columns, rebuilt and
checked in R from the knot table). The same fixed-effects string is therefore read by ``lme4`` / ``glmmTMB`` and by
the statsmodels second implementation, and the coefficient names agree.

Token rules in :attr:`ModelDef.fixed`:

- ``NS(x,df)`` expands to the basis columns of ``ns(x, df)``;
- a term in :attr:`ModelDef.optional` enters only when the model data provide the column (the ocular missingness
  indicators and the run-join indicators, both only when non-constant in the model rows);
- ``grp`` terms (TD -0.5, ASD +0.5) are added only for the cohort of autistic and typically developing children
  (``conf``); models of the unaffected siblings (``dev``) have no group term.

Random structure (:class:`RandomDef`): one correlated child block ``(1 + a_wb + ... | pid)`` (or its ``||`` form
after the decorrelation step of the simplification path), the protected uncorrelated child time slopes
``(0 + t_run | pid) + (0 + tib | pid)`` (and ``(0 + a_hist5 | pid)`` in M1-hist), the droppable covariate slopes in
their fixed order (M1: ``cAV`` then ``cVA``) and the block intercept ``(1 | pid_run_block)``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from modeling.splines import ns_column_names

__all__ = ["RandomDef", "ModelDef", "MODELS", "SESOI", "S_TASK", "ALLFIT_OPTIMIZERS", "OPTIMUM_CHECK_TOL",
           "expand_fixed", "fixed_formula", "random_string", "build_spec", "spec_json", "get_model"]

#: Smallest effect sizes of interest (log RT per SD of alpha; misses in log-odds; ``r_units`` for M5h).
SESOI: dict[str, float] = {"AVSRT": 0.0102, "FAST": 0.0066, "IC": 0.0084, "a_hist5": 0.0102, "miss": 0.077,
                           "r_units": 0.03}
#: Scale constants of M5h (log RT SD per task) that standardise each task's outcome to r units.
S_TASK: dict[str, float] = {"AVSRT": 0.3395, "FAST": 0.2200, "IC": 0.2812}
#: The five allFit optimisers used on non-convergence (lme4 1.1.37 ``meth.tab`` rows, optimizer / method).
ALLFIT_OPTIMIZERS: tuple[tuple[str, str], ...] = (
    ("bobyqa", ""), ("Nelder_Mead", ""), ("nlminbwrap", ""), ("nloptwrap", "NLOPT_LN_NELDERMEAD"),
    ("nloptwrap", "NLOPT_LN_BOBYQA"))
#: Log-likelihood margin of the final-fit optimum check (every LMM).
OPTIMUM_CHECK_TOL: float = 0.01


@dataclass(frozen=True)
class RandomDef:
    """Starting random structure of a model and its simplification metadata."""

    corr: tuple[str, ...]                      # terms of the correlated child block; "1" = random intercept
    droppable: tuple[str, ...] = ()            # droppable covariate slopes, in the fixed drop order
    uncorrelated: tuple[str, ...] = ("t_run", "tib")   # protected (0 + x | pid) terms
    block: bool = True                         # (1 | pid_run_block) in the starting structure (droppable)
    group: str = "pid"
    block_factor: str = "pid_run_block"
    start_corr: bool = True                    # False: the child block starts in its || form (S1-U)

    @property
    def protected(self) -> tuple[str, ...]:
        return tuple(t for t in self.corr if t not in self.droppable) + self.uncorrelated


@dataclass(frozen=True)
class ModelDef:
    name: str
    task: str
    family: str                                # "lmm" (lmer) or "glmm" (glmmTMB binomial)
    outcome: str                               # "lrt" or "miss" ("y" for M5h)
    default_set: str                           # configs/events.yaml analysis subset of the model rows
    knots_key: str                             # configs/model_knots.yaml models.<key>
    splines: tuple[tuple[str, int], ...]
    fixed: tuple[str, ...]
    random: RandomDef
    targets: Mapping[str, Mapping[str, float]]
    optional: tuple[str, ...] = ()
    grp_terms: tuple[str, ...] = ("grp", "a_wb:grp")
    grp_targets: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    sesoi: Mapping[str, float] = field(default_factory=dict)
    requires: tuple[str, ...] = ()            # extra row requirement (finite column), e.g. a_hist5 in M1-hist
    h2: bool = False                          # group-model moderator terms; needs grp (conf cohort)
    note: str = ""


_OCULAR_OPT = ("heog_missing", "vfp_missing")
_JOIN_OPT = ("tsp_join", "soa_join", "tsfl_join", "contour_join")
_CHILD = ("age_c", "sex_c", "task_start_h_c")

_M1_FIXED = (
    "a_wb", "a_slow",
    "cVA", "cAV", "pcA", "pcV", "cVA:pcA", "cVA:pcV", "cAV:pcA", "cAV:pcV",   # (cVA + cAV) * prev_cond, sum contrasts
    "a_wb:cVA", "a_wb:cAV",
    "prev_miss", "NS(log_tsp,4)", "tsp_join", "NS(soa,4)", "soa_join",
    "NS(t_run,3)", "NS(tib,3)", "a_wb:t_run",
    "heog_z", "vfp_z", "heog_missing", "vfp_missing", *_CHILD)
_M1_TARGETS = {"P1": {"a_wb": 1.0},
               "H1A": {"a_wb": 1.0, "a_wb:cVA": -0.5, "a_wb:cAV": -1.0 / 3.0},
               "S5": {"a_wb:cVA": 1.0}}
_M1_RANDOM = RandomDef(corr=("1", "a_wb", "cVA", "cAV"), droppable=("cAV", "cVA"))
_H2_TERMS = ("a_wb:age_c", "a_wb:sex_c", "a_wb:relq_c", "relq_missing", "a_wb:relq_missing")
# relq missing-indicator terms enter only when some child has no split-channel reliability
_H2_OPT = ("relq_missing", "a_wb:relq_missing")

_M3_FIXED = (
    "a_wb", "a_slow", "cat", *[f"nnt{k}" for k in range(1, 7)], "soa_prev_c", "soa_join", "prev_miss",
    "NS(log_tsp,3)", "tsp_join", "NS(t_run,3)", "NS(tib,3)", "half_status", "a_wb:t_run",
    "heog_z", "vfp_z", "heog_missing", "vfp_missing", *_CHILD)
_M4_FIXED = (
    "a_wb", "a_slow", "NS(log_tsfl,3)", "tsfl_join", "NS(t_since_prev_contour,3)", "contour_join",
    "prev_contour_type", "prev_miss", "NS(log_tsp,3)", "tsp_join", "seq180_c", "NS(tib,3)", "NS(t_run,3)",
    "a_wb:t_run", "heog_z", "vfp_z", "heog_missing", "vfp_missing", *_CHILD)
_M2_FIXED = (
    "a_wb", "a_slow", "cVA", "cAV", "a_wb:cVA", "a_wb:cAV", "prev_miss", "NS(log_tsp,3)", "tsp_join",
    "NS(soa,3)", "soa_join", "NS(t_run,3)", "NS(tib,3)", "heog_z", "vfp_z", "heog_missing", "vfp_missing",
    *_CHILD)

MODELS: dict[str, ModelDef] = {
    "M1": ModelDef("M1", "AVSRT", "lmm", "lrt", "primary_rt", "M1",
                   (("log_tsp", 4), ("soa", 4), ("t_run", 3), ("tib", 3)), _M1_FIXED, _M1_RANDOM, _M1_TARGETS,
                   optional=_OCULAR_OPT + _JOIN_OPT, grp_targets={"H2_unadjusted": {"a_wb:grp": 1.0}},
                   sesoi={"P1": SESOI["AVSRT"], "H1A": SESOI["AVSRT"], "S5": SESOI["AVSRT"],
                          "H2_unadjusted": SESOI["AVSRT"]},
                   note="M1: AVSRT RT model (P1, H1-A, S5; group difference without moderator terms)"),
    "M1-H2": ModelDef("M1-H2", "AVSRT", "lmm", "lrt", "primary_rt", "M1",
                      (("log_tsp", 4), ("soa", 4), ("t_run", 3), ("tib", 3)), _M1_FIXED + _H2_TERMS, _M1_RANDOM,
                      {}, optional=_OCULAR_OPT + _JOIN_OPT + _H2_OPT, grp_targets={"H2": {"a_wb:grp": 1.0}},
                      sesoi={"H2": SESOI["AVSRT"]}, h2=True,
                      note="M1-H2 = M1 + grp + a_wb:grp + a_wb:age_c + a_wb:sex_c + a_wb:relq_c; fitted with the "
                           "final M1 random structure"),
    "M1-hist": ModelDef("M1-hist", "AVSRT", "lmm", "lrt", "primary_rt", "M1",
                        (("log_tsp", 4), ("soa", 4), ("t_run", 3), ("tib", 3)), _M1_FIXED + ("a_hist5",),
                        RandomDef(corr=("1", "a_wb", "cVA", "cAV"), droppable=("cAV", "cVA"),
                                  uncorrelated=("t_run", "tib", "a_hist5")),
                        {"S6": {"a_hist5": 1.0}}, optional=_OCULAR_OPT + _JOIN_OPT, requires=("a_hist5",),
                        sesoi={"S6": SESOI["a_hist5"]},
                        note="M1-hist = M1 + a_hist5 + (0 + a_hist5 | pid) on rows with a_hist5 defined"),
    "M2": ModelDef("M2", "AVSRT", "glmm", "miss", "predictor_set", "M2",
                   (("log_tsp", 3), ("soa", 3), ("t_run", 3), ("tib", 3)), _M2_FIXED,
                   RandomDef(corr=("1", "a_wb")), {"S1": {"a_wb": 1.0}}, optional=_OCULAR_OPT + _JOIN_OPT,
                   sesoi={"S1": SESOI["miss"]}, note="M2: AVSRT miss model (S1), glmmTMB Laplace"),
    "M3": ModelDef("M3", "FAST", "lmm", "lrt", "primary_rt", "M3",
                   (("log_tsp", 3), ("t_run", 3), ("tib", 3)), _M3_FIXED, RandomDef(corr=("1", "a_wb")),
                   {"S2": {"a_wb": 1.0}}, optional=_OCULAR_OPT + _JOIN_OPT, sesoi={"S2": SESOI["FAST"]},
                   note="M3: FAST RT model (S2)"),
    "M4": ModelDef("M4", "IC", "lmm", "lrt", "primary_rt", "M4",
                   (("log_tsfl", 3), ("t_since_prev_contour", 3), ("log_tsp", 3), ("tib", 3), ("t_run", 3)),
                   _M4_FIXED, RandomDef(corr=("1", "a_wb")), {"S3": {"a_wb": 1.0}},
                   optional=_OCULAR_OPT + _JOIN_OPT, sesoi={"S3": SESOI["IC"]}, note="M4: IC RT model (S3)"),
}
for _n, _b in (("M3-H2", "M3"), ("M4-H2", "M4")):
    _m = MODELS[_b]
    MODELS[_n] = ModelDef(_n, _m.task, "lmm", "lrt", _m.default_set, _m.knots_key, _m.splines, _m.fixed + _H2_TERMS,
                          _m.random, {}, optional=_m.optional + _H2_OPT, grp_targets={"H2": {"a_wb:grp": 1.0}},
                          sesoi={"H2": SESOI[_m.task]}, h2=True, note=f"{_n} = {_b} + the group and moderator terms")


def get_model(name: str) -> ModelDef:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}") from None


_NS_RE = re.compile(r"^NS\(([A-Za-z_][A-Za-z0-9_]*),(\d+)\)$")


def expand_fixed(terms: Sequence[str], available: Sequence[str], optional: Sequence[str] = ()) -> list[str]:
    """Expand ``NS(x,df)`` tokens and drop optional terms whose column is not in ``available``.

    A non-optional term whose columns are missing raises ``KeyError`` (no silent formula change).
    """
    avail = set(available)
    out: list[str] = []
    for t in terms:
        m = _NS_RE.match(t)
        if m:
            cols = ns_column_names(m.group(1), int(m.group(2)))
            missing = [c for c in cols if c not in avail]
            if missing:
                raise KeyError(f"spline columns {missing} missing from the model data")
            out.extend(cols)
            continue
        parts = t.split(":")
        if t in optional and not all(p in avail for p in parts):
            continue
        missing = [p for p in parts if p not in avail]
        if missing:
            raise KeyError(f"term {t!r} needs columns {missing} that are not in the model data")
        out.append(t)
    return out


def fixed_formula(outcome: str, terms: Sequence[str], intercept: bool = True) -> str:
    """``outcome ~ t1 + t2 + ...`` (``0 +`` without an intercept)."""
    rhs = " + ".join(terms)
    return f"{outcome} ~ {'' if intercept else '0 + '}{rhs}"


def random_string(state: Mapping) -> str:
    """lme4 random-effects string of a reduction-path state (same function as R ``re_string``)."""
    s: list[str] = []
    corr_terms = list(state["corr_terms"])
    has_int = "1" in corr_terms
    slopes = [t for t in corr_terms if t != "1"]
    g = state.get("group", "pid")
    if state["corr"]:
        if corr_terms:
            s.append(f"({' + '.join((['1'] if has_int else ['0']) + slopes)} | {g})")
    else:
        if has_int:
            s.append(f"(1 | {g})")
        s.extend(f"(0 + {t} | {g})" for t in slopes)
    s.extend(f"(0 + {t} | {g})" for t in state["uncorrelated"])
    if state["block"]:
        s.append(f"(1 | {state.get('block_factor', 'pid_run_block')})")
    return " + ".join(s)


def build_spec(md, *, label: str, input_tsv: str, output_json: str,
               reduction: bool = True, fixed_state: Mapping | None = None, second_impl: bool = True,
               cr2: bool = True, extra_targets: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """JSON model spec for the R engine from a :class:`~modeling.model_data.ModelData`.

    Parameters
    ----------
    reduction
        Run the simplification path (default). ``False`` fits ``fixed_state`` (or the starting structure) as is,
        e.g. M1-H2 on the final M1 structure, or the S-REL bootstrap refits.
    fixed_state
        A reduction-path state (``corr``, ``corr_terms``, ``uncorrelated``, ``block``) to fit without the path.
    """
    mdef: ModelDef = md.model_def
    terms = list(md.fixed_terms)
    rnd = mdef.random
    start = {"corr": bool(rnd.start_corr), "corr_terms": list(rnd.corr), "uncorrelated": list(rnd.uncorrelated),
             "block": bool(rnd.block), "group": rnd.group, "block_factor": rnd.block_factor}
    targets = {k: dict(v) for k, v in mdef.targets.items()}
    if md.has_grp:
        targets.update({k: dict(v) for k, v in mdef.grp_targets.items()})
    if extra_targets:
        targets.update({k: dict(v) for k, v in extra_targets.items()})
    spec = {
        "schema": "alpha_state.model_spec/1",
        "label": label,
        "model": mdef.name,
        "task": md.task,
        "cohort": md.cohort,
        "family": mdef.family,
        "outcome": md.outcome_column,
        "outcome_kind": md.outcome_kind,
        "fixed_terms": terms,
        "intercept": md.intercept,
        "fixed_formula": fixed_formula(md.outcome_column, terms, md.intercept),
        "random": {"start": start, "droppable": list(rnd.droppable), "protected": list(rnd.protected)},
        "reduction_path": bool(reduction),
        "fixed_state": dict(fixed_state) if fixed_state is not None else None,
        "targets": targets,
        "sesoi": dict(mdef.sesoi),
        "splines": [dict(s) for s in md.splines],
        "factors": ["pid", rnd.block_factor],
        "cluster": "pid",
        "engine": {
            "lmm": {"REML": True, "optimizer": "bobyqa", "ddf": "Satterthwaite",
                    "allfit": [list(o) for o in ALLFIT_OPTIMIZERS],
                    # final-fit optimum check: allFit on the selected fit, switch to a converged refit whose
                    # REML log-likelihood is higher by more than 0.01
                    "optimum_check": True, "optimum_check_tol": OPTIMUM_CHECK_TOL},
            "glmm": {"primary": "glmmTMB", "laplace": True, "optimizer": "nlminb",
                     "refits_on_nonconvergence": ["optim_BFGS", "nlminb_from_glmer"],
                     "second_impl": "glmer_bobyqa"},
            "singular_tol": 1e-4, "block_boundary_rel": 1e-4, "lrt_drop_p": 0.2,
            "consensus_fe_rel_se": 0.01, "consensus_loglik": 0.01,
            "glmm_second_impl_fe_rel_se": 0.05, "glmm_second_impl_loglik": 0.05,
        },
        "cr2": bool(cr2 and mdef.family == "lmm"),
        "second_impl": bool(second_impl),
        "input_tsv": str(input_tsv),
        "output_json": str(output_json),
        "ns_check_tol": 1e-8,
    }
    return spec


def spec_json(spec: Mapping) -> str:
    return json.dumps(spec, indent=1, sort_keys=False, default=str)


def model_def_dict(mdef: ModelDef) -> dict:
    return asdict(mdef)
