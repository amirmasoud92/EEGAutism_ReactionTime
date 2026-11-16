"""Python side of ``R/fit_extras.R`` (A1, A3-L, the clubSandwich CR2 cross-check, S-CENS, S-HET, E3c) and the
A-set runner for one model's data.

Every call writes a model-input TSV (the same format as :func:`modeling.model_data.write_model_input`) and a JSON
spec, runs ``Rscript fit_extras.R`` and returns its JSON result.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import extras as X
from modeling.model_specs import fixed_formula
from modeling.r_bridge import RSCRIPT, R_DIR, EngineError

__all__ = ["EXTRAS_SCRIPT", "run_extra", "a1", "a3l", "a2l", "cr2_check", "s_cens", "s_het", "inference_set",
           "E3C_B", "E3C_SEED", "e3c_denominator", "e3c"]

EXTRAS_SCRIPT = R_DIR / "fit_extras.R"


def _write_tsv(frame: pd.DataFrame, path: Path) -> Path:
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(float)
    out.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g", lineterminator="\n")
    return path


def run_extra(frame: pd.DataFrame, out_dir: str | os.PathLike[str], label: str, extra: str, *, outcome: str,
              fixed_terms: Sequence[str], splines: Sequence[Mapping] = (), timeout: float = 6 * 3600,
              **fields) -> dict:
    """Write ``<label>_input.tsv`` + ``<label>_spec.json`` and run ``fit_extras.R`` (``extra`` = A1 / A3L / CR2CHECK /
    CENS / HET / E3C)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / f"{label}_input.tsv"
    _write_tsv(frame, tsv)
    spec = {"schema": "alpha_state.extras_spec/1", "extra": extra, "label": label, "model": fields.pop("model", ""),
            "outcome": outcome, "fixed_terms": list(fixed_terms), "splines": [dict(s) for s in splines],
            "factors": ["pid"], "input_tsv": tsv.as_posix(),
            "output_json": (out / f"{label}_result.json").as_posix(), "ns_check_tol": 1e-8, **fields}
    sp = out / f"{label}_spec.json"
    sp.write_text(json.dumps(spec, indent=1, default=str), encoding="utf-8", newline="\n")
    res_path = Path(spec["output_json"])
    if res_path.exists():
        res_path.unlink()
    t0 = time.time()
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    r = subprocess.run([RSCRIPT, "--vanilla", str(EXTRAS_SCRIPT), str(sp)], capture_output=True, text=True,
                       timeout=timeout, env=env)
    if not res_path.is_file():
        raise EngineError(f"fit_extras.R ({extra}) exited {r.returncode} without a result:\n{r.stderr[-3000:]}")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    res["wall_seconds"] = time.time() - t0
    if "error" in res:
        raise EngineError(f"{label} ({extra}): {res['error']}")
    return res


def _model_cols(md, extra: Sequence[str] = ()) -> list[str]:
    cols = md.columns_needed()
    return [*cols, *[c for c in extra if c not in cols and c in md.frame.columns]]


def a1(md, out_dir, label: str, *, targets: Mapping | None = None, random_formula: str = "~ 1 + a_wb | pid",
       t_seq: str = "trial_index_in_run", ar_group: str = "pid/run") -> dict:
    """A1: nlme AR(1) model with the model's fixed part; targets default to the model's targets."""
    tg = {k: dict(v) for k, v in (targets or md.model_def.targets).items()}
    return run_extra(md.frame[_model_cols(md, [t_seq])], out_dir, label, "A1", outcome=md.outcome_column,
                     fixed_terms=md.fixed_terms, splines=md.splines, model=md.model_def.name,
                     fixed_formula=fixed_formula(md.outcome_column, md.fixed_terms, md.intercept), targets=tg,
                     random_formula=random_formula, t_seq=t_seq, ar_group=ar_group)


def _within_constant(frame: pd.DataFrame, term: str) -> bool:
    v = np.prod([frame[p].to_numpy(float) for p in term.split(":")], axis=0)
    return bool((pd.Series(v).groupby(frame["pid"].to_numpy()).transform("std").fillna(0) == 0).all())


def a3l(md, out_dir, label: str) -> dict:
    """A3-L (S1): logistic regression with child fixed intercepts, the M2 fixed covariates without alpha
    terms (child-constant terms are absorbed by the intercepts), child-specific linear time slopes and ``xr`` =
    ``a_wb`` detrended per child on linear ``t_run`` and ``tib``; CR2 (clubSandwich) with Satterthwaite df. Returns the
    result with ``p0`` (row order of ``md.frame``) for A2-L."""
    f = md.frame
    R = X.residualise_per_child(f, ["a_wb"], [], block_col=None)
    nuis = [t for t in X.nuisance_terms(md.fixed_terms) if not _within_constant(f, t)]
    d = f[_model_cols(md)].copy()
    d["xr"] = R["a_wb"].to_numpy()
    res = run_extra(d, out_dir, label, "A3L", outcome="miss", fixed_terms=[*nuis, "xr", "t_run", "tib"],
                    splines=md.splines, model=md.model_def.name, nuisance_terms=nuis)
    p0 = pd.read_csv(res["p0_file"], sep="\t")["p0"].to_numpy(float)
    res["p0"] = p0
    miss_by = f.groupby("pid")["miss"].sum()
    res["children_kept"] = sorted(miss_by.index[miss_by > 0])
    return res


def a2l(md, a3l_res: Mapping, *, seed: int, n_surrogates: int = X.N_SURROGATES) -> dict:
    return X.a2l_surrogate(md, a3l_res["p0"], seed=seed, n_surrogates=n_surrogates,
                           children=a3l_res["children_kept"])


def cr2_check(xr: np.ndarray, yr: np.ndarray, pid: np.ndarray, out_dir, label: str, *,
              columns: Mapping[str, np.ndarray] | None = None, weights: Sequence[float] | None = None) -> dict:
    """clubSandwich CR2 of ``lm(yr ~ 0 + x...)`` clustered by child (cross-check of :func:`extras.a3_cr2`)."""
    d = pd.DataFrame({"pid": pid, "yr": yr})
    if columns:
        for k, v in columns.items():
            d[k] = v
        xs = list(columns)
    else:
        d["xr"] = xr
        xs = ["xr"]
    return run_extra(d, out_dir, label, "CR2CHECK", outcome="yr", fixed_terms=xs, x_columns=xs,
                     contrast_weights=list(weights) if weights is not None else None)


def s_cens(md, out_dir, label: str, *, random_string: str, chains: int = 4, iter: int = 4000,
           warmup: int | None = None, seed: int = 20261020, cores: int = 4, survreg_check: bool = False,
           targets: Mapping | None = None) -> dict:
    """S-CENS: ``md.frame`` must carry ``lrt`` and ``cens`` (1 = right-censored at the value of ``lrt``).

    Sampler: 4 chains x 4,000 iterations, warmup 2,000. ``warmup`` defaults to ``iter // 2``."""
    if warmup is None:
        warmup = iter // 2
    if "cens" not in md.frame.columns:
        raise ValueError("S-CENS needs a cens column (1 = right-censored)")
    tg = {k: dict(v) for k, v in (targets or md.model_def.targets).items()}
    rhs = " + ".join(md.fixed_terms)
    return run_extra(md.frame[_model_cols(md, ["cens"])], out_dir, label, "CENS", outcome="lrt",
                     fixed_terms=md.fixed_terms, splines=md.splines, model=md.model_def.name,
                     fixed_formula_rhs=rhs, random_string=random_string, targets=tg, chains=chains, iter=iter,
                     warmup=warmup, seed=seed, cores=cores, survreg_check=survreg_check)


def s_het(md, out_dir, label: str, *, random_string: str, targets: Mapping | None = None) -> dict:
    """S-HET: glmmTMB with ``dispformula = ~ grp`` and ``random_string`` (group-specific slopes)."""
    tg = {k: dict(v) for k, v in (targets or md.model_def.grp_targets).items()}
    extra_cols = [c for c in md.frame.columns if c.startswith("a_wb_g")] + ["grp"]
    return run_extra(md.frame[_model_cols(md, extra_cols)], out_dir, label, "HET", outcome=md.outcome_column,
                     fixed_terms=md.fixed_terms, splines=md.splines, model=md.model_def.name,
                     fixed_formula=fixed_formula(md.outcome_column, md.fixed_terms, md.intercept),
                     random_string=random_string, targets=tg)


def inference_set(md, out_dir, label: str, *, seed: int, targets: Sequence[str] | None = None,
                  n_surrogates: int = X.N_SURROGATES, run_a1: bool = True, md_hist=None) -> dict:
    """A1 / A2 / A3 (LMMs) or A2-L / A3-L (M2) for the targets of one model's data; A3-NL for S6 with ``md_hist``."""
    out: dict = {"model": md.model_def.name, "seed": seed}
    if md.model_def.family == "glmm":
        r = a3l(md, out_dir, f"{label}_a3l")
        out["A3L"] = {k: v for k, v in r.items() if k not in ("p0",)}
        out["A2L"] = a2l(md, r, seed=seed, n_surrogates=n_surrogates)
        return out
    tg = list(targets or md.model_def.targets)
    out["A2_A3"] = {t: X.a2_a3_model(md, target=t, seed=seed + i, n_surrogates=n_surrogates)
                    for i, t in enumerate(tg)}
    if run_a1:
        out["A1"] = a1(md, out_dir, f"{label}_a1", targets={k: md.model_def.targets[k] for k in tg})
    if md_hist is not None:
        out["A3NL"] = X.a3nl(md_hist)
    return out


E3C_B = 1000
E3C_SEED = 20261020


def e3c_denominator(md, outcome: str = "lrt") -> float:
    """E3c denominator: the median over children of the within-child variance (ddof 1) of log RT on the model rows
    of ``md`` (the pooled M1 analysis set)."""
    v = md.frame.groupby("pid")[outcome].var(ddof=1).dropna()
    if v.empty:
        raise ValueError("no child with >= 2 rows")
    return float(np.median(v.to_numpy(float)))


def e3c(md, primary: Mapping, out_dir, label: str, *, B: int = E3C_B, seed: int = E3C_SEED, target: str = "P1",
        slope_term: str = "a_wb", ncpus: int = 1) -> dict:
    """E3c: behavioural variance attributable to alpha.

    ``primary`` is the engine result of the final M1 fit on ``md`` (``r_bridge.fit``); its ``full_formula`` (fixed
    part + final random structure) is refitted in R, and ``lme4::bootMer`` (parametric, ``use.u = FALSE``,
    B = 1,000, seed 20261020) refits that structure without the reduction path on B outcomes simulated from it.
    Estimand ``beta^2 + tau^2`` (beta = the ``target`` contrast, P1 = beta(a_wb); tau^2 = the a_wb random-slope
    variance, 0 if absent) and ``(beta^2 + tau^2) / denominator`` (:func:`e3c_denominator`, held fixed); percentile
    95% CIs. The point estimate is the primary fit's (its beta and a_wb slope SD); the R refit's values are reported
    next to it (``refit_agreement``)."""
    if "full_formula" not in primary:
        raise ValueError("primary must be the engine result of the final M1 fit (full_formula missing)")
    denom = e3c_denominator(md, md.outcome_column)
    w = {k: float(v) for k, v in md.model_def.targets[target].items()}
    bf = getattr(md.model_def.random, "block_factor", None)
    res = run_extra(md.frame[_model_cols(md)], out_dir, label, "E3C", outcome=md.outcome_column,
                    fixed_terms=md.fixed_terms, splines=md.splines, model=md.model_def.name,
                    factors=["pid", bf] if bf else ["pid"], full_formula=primary["full_formula"],
                    target_weights=w, slope_term=slope_term, B=int(B), seed=int(seed), ncpus=int(ncpus),
                    denominator=denom)
    beta = float(primary["targets"][target]["estimate"])
    sd = primary.get("a_wb_slope_sd")
    tau2 = float(sd) ** 2 if sd is not None and np.isfinite(float(sd)) else 0.0
    est = beta ** 2 + tau2
    res["estimate"] = {"beta": beta, "tau2": tau2, "e3c": est, "fraction": est / denom}
    pt = res.get("point", {})
    res["refit_agreement"] = {k: float(pt[k]) - res["estimate"][k] for k in ("beta", "tau2", "e3c") if k in pt}
    res["denominator_definition"] = "median over children of the within-child variance of lrt on the model rows"
    return res
