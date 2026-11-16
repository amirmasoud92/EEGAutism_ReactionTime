"""Second implementation of the LMMs and its term-by-term comparison against lme4.

statsmodels ``MixedLM`` (REML) on the same model-input TSV and the final lme4 random structure of the R result:

- the correlated child block -> ``re_formula`` (free covariance); after the decorrelation step (``||``) its slopes
  become independent variance components;
- the protected child time slopes ``(0 + t_run | pid)``, ``(0 + tib | pid)`` (and ``a_hist5``) -> variance components
  ``vc_formula = {"t_run": "0 + t_run", ...}``;
- the block intercept ``(1 | pid_run_block)`` -> the variance component ``"0 + C(pid_run_block)"`` within child.

This represents the lme4 structures exactly (nested within ``pid``). Several optimiser starts (at least 4) are run and
the highest REML log-likelihood is kept; a single successful start is never accepted (:class:`SecondImplError`); the
best start is then polished by one L-BFGS pass from its own solution (kept only if the log-likelihood improves).
Tolerance: every fixed effect within 1% of its lme4 SE and the log-likelihood within 0.01.

The GLMM cross-check (glmer bobyqa against glmmTMB, 5% of SE and 0.05 log-likelihood) runs inside the R engine
(``R/fit_glmm.R``); :func:`glmm_check` reads it. statsmodels ``BinomialBayesMixedGLM`` is not used for inference.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["SecondImplError", "STARTS", "statsmodels_fit", "compare_with_lme4", "glmm_check",
           "lmm_cr2_python"]

#: Optimiser starts (statsmodels ``method`` sequences); >= 4 are always tried.
STARTS: tuple[tuple[str, ...], ...] = (("lbfgs",), ("bfgs",), ("powell", "lbfgs"), ("nm", "lbfgs"), ("cg",))


class SecondImplError(RuntimeError):
    """The statsmodels fit could not be accepted (fewer than two successful starts)."""


def _structure(state: Mapping) -> tuple[str, dict]:
    corr_terms = list(state["corr_terms"])
    has_int = "1" in corr_terms
    slopes = [t for t in corr_terms if t != "1"]
    vc: dict[str, str] = {}
    if state["corr"]:
        re = ("1" if has_int else "0") + "".join(f" + {s}" for s in slopes)
    else:
        re = "1" if has_int else "0"
        for s in slopes:
            vc[s] = f"0 + {s}"
    for u in state["uncorrelated"]:
        vc[u] = f"0 + {u}"
    if state["block"]:
        vc["block"] = f"0 + C({state.get('block_factor', 'pid_run_block')})"
    return re, vc


def statsmodels_fit(data: pd.DataFrame, fixed_formula: str, state: Mapping, *,
                    starts: Sequence[Sequence[str]] = STARTS, min_starts: int = 4, maxiter: int = 2000) -> dict:
    """REML fit with every start in ``starts`` (at least ``min_starts``); returns the best by log-likelihood."""
    import statsmodels.formula.api as smf

    if len(starts) < min_starts:
        raise ValueError(f"the second implementation needs >= {min_starts} optimiser starts")
    re, vc = _structure(state)
    group = state.get("group", "pid")
    d = data.copy()
    md = smf.mixedlm(fixed_formula, d, groups=d[group], re_formula=re, vc_formula=vc or None)
    fits, info = [], []
    for meth in starts:
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = md.fit(reml=True, method=list(meth), maxiter=maxiter)
            ok = np.isfinite(f.llf) and np.all(np.isfinite(f.fe_params.to_numpy()))
            info.append({"method": "+".join(meth), "llf": float(f.llf), "converged": bool(f.converged),
                         "seconds": time.time() - t0, "ok": bool(ok)})
            if ok:
                fits.append(f)
        except Exception as e:  # noqa: BLE001 - every failure is recorded
            info.append({"method": "+".join(meth), "error": f"{type(e).__name__}: {e}"[:300],
                         "seconds": time.time() - t0, "ok": False})
    if len(fits) < 2:
        raise SecondImplError(f"only {len(fits)} successful statsmodels start(s); a single start is never accepted")
    best = max(fits, key=lambda z: z.llf)
    llfs = np.array([z.llf for z in fits])
    # polish: one more L-BFGS pass started from the best statsmodels solution (its own parameters, not lme4's);
    # kept only if the REML log-likelihood improves. On a flat REML surface the best start can otherwise stop a few
    # hundredths of an SE away from the optimum.
    polished = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = md.fit(reml=True, start_params=best.params_object, method=["lbfgs"], maxiter=maxiter)
        if np.isfinite(fp.llf) and fp.llf > best.llf and np.all(np.isfinite(fp.fe_params.to_numpy())):
            best, polished = fp, True
        info.append({"method": "polish_lbfgs_from_best", "llf": float(fp.llf), "kept": polished})
    except Exception as e:  # noqa: BLE001
        info.append({"method": "polish_lbfgs_from_best", "error": f"{type(e).__name__}: {e}"[:300]})
    fe = best.fe_params
    names = [("(Intercept)" if n == "Intercept" else n) for n in fe.index]
    return {"fe": dict(zip(names, fe.to_numpy(float))),
            "se": dict(zip(names, best.bse_fe.to_numpy(float))),
            "llf": float(best.llf), "starts": info, "n_ok": len(fits), "polished": polished,
            "llf_spread": float(llfs.max() - llfs.min()), "re_formula": re, "vc_formula": vc,
            "cov_re": best.cov_re.to_numpy().tolist(), "vcomp": list(map(float, np.atleast_1d(best.vcomp))),
            "scale": float(best.scale)}


def compare_with_lme4(sm: Mapping, r_result: Mapping, *, fe_rel_se: float = 0.01, loglik_tol: float = 0.01) -> dict:
    """Term-by-term comparison: max |fe_sm - fe_lme4| / se_lme4 and |llf_sm - loglik_lme4|."""
    rfe = {r["term"]: r for r in r_result["fixed"]}
    rows = []
    for k, v in sm["fe"].items():
        if k not in rfe:
            continue
        se = rfe[k]["se"]
        rows.append({"term": k, "lme4": rfe[k]["estimate"], "statsmodels": v, "diff_rel_se": abs(v - rfe[k]["estimate"]) / se,
                     "se_ratio": sm["se"][k] / se})
    missing = sorted(set(rfe) - set(sm["fe"]))
    worst = max(r["diff_rel_se"] for r in rows)
    lld = sm["llf"] - r_result["loglik"]
    return {"max_fe_diff_rel_se": worst, "loglik_diff": lld, "terms_compared": len(rows),
            "terms_missing": missing, "pass": (worst <= fe_rel_se) and (abs(lld) <= loglik_tol) and not missing,
            "tolerance": {"fe_rel_se": fe_rel_se, "loglik": loglik_tol}, "rows": rows,
            "se_ratio_range": [min(r["se_ratio"] for r in rows), max(r["se_ratio"] for r in rows)]}


def glmm_check(r_result: Mapping) -> dict:
    """The glmer-vs-glmmTMB second implementation recorded by ``R/fit_glmm.R``."""
    s = r_result.get("second_impl")
    if not s:
        raise SecondImplError("no second-implementation block in the GLMM result")
    return s


# --------------------------------------------------------------------------- CR2 cross-check
def _z_blocks(d: pd.DataFrame, state: Mapping) -> list[tuple[list[str], str]]:
    """Random-effect blocks of the final state: (columns, kind) with kind 'pid' or 'block'."""
    corr_terms = list(state["corr_terms"])
    cols = [("1" if t == "1" else t) for t in corr_terms]
    blocks: list[tuple[list[str], str]] = []
    if state["corr"]:
        blocks.append((cols, "pid"))
    else:
        blocks.extend(([c], "pid") for c in cols)
    blocks.extend(([u], "pid") for u in state["uncorrelated"])
    if state["block"]:
        blocks.append((["1"], "block"))
    return blocks


def _match_cov(r_blocks: Sequence[Mapping], cols: Sequence[str], kind: str) -> np.ndarray:
    names = ["(Intercept)" if c == "1" else c for c in cols]
    want_grp = "pid_run_block" if kind == "block" else "pid"
    for b in r_blocks:
        terms = [b["terms"]] if isinstance(b["terms"], str) else list(b["terms"])
        if b["group"] == want_grp and terms == names:
            return np.asarray(b["cov"], float).reshape(len(names), len(names))
    raise KeyError(f"no random-effect block {want_grp} {names} in the R result")


def lmm_cr2_python(data: pd.DataFrame, r_result: Mapping, fixed_formula: str, *, targets: Mapping | None = None
                   ) -> dict:
    """Independent CR2 (Bell-McCaffrey with the working model as target; Pustejovsky & Tipton 2018) for an lmer fit,
    from the exported variance components, to cross-check clubSandwich ``vcovCR(type = "CR2")``.

    V_i / sigma^2 = Theta_i = Z_i G Z_i' / sigma^2 + I; W_i = Theta_i^-1; M = (sum X_i' W_i X_i)^-1;
    A_i = U_i' (U_i (I - X_i M X_i' W_i) Theta_i U_i')^-1/2 U_i with U_i = chol(Theta_i) (upper);
    V_CR2 = M (sum X_i' W_i A_i e_i e_i' A_i' W_i X_i) M, e = y - X beta (marginal residuals).
    Also returns the GLS beta from the same variance components (must equal lme4's beta).
    """
    import patsy

    y, X = patsy.dmatrices(fixed_formula, data, return_type="dataframe")
    names = [("(Intercept)" if c == "Intercept" else c) for c in X.columns]
    X = X.to_numpy(float)
    y = y.to_numpy(float).ravel()
    sigma2 = float(r_result["sigma"]) ** 2
    state = r_result["final_state"]
    blocks = _z_blocks(data, state)
    pid = data["pid"].to_numpy()
    blk = data[state.get("block_factor", "pid_run_block")].astype(str).to_numpy()
    groups = pd.unique(pid)
    p = X.shape[1]
    XWX = np.zeros((p, p))
    XWy = np.zeros(p)
    cache = []
    for g in groups:
        idx = np.flatnonzero(pid == g)
        n = len(idx)
        V = np.eye(n) * sigma2
        for cols, kind in blocks:
            G = _match_cov(r_result["random_blocks"], cols, kind)
            Z = np.column_stack([np.ones(n) if c == "1" else data[c].to_numpy(float)[idx] for c in cols])
            if kind == "pid":
                V += Z @ G @ Z.T
            else:
                b = blk[idx]
                same = (b[:, None] == b[None, :]).astype(float)
                V += G[0, 0] * same
        Theta = V / sigma2
        W = np.linalg.inv(Theta)
        Xi = X[idx]
        XWX += Xi.T @ W @ Xi
        XWy += Xi.T @ W @ y[idx]
        cache.append((idx, Theta, W))
    M = np.linalg.inv(XWX)
    beta = M @ XWy
    beta_r = np.array([next(r["estimate"] for r in r_result["fixed"] if r["term"] == nm) for nm in names])
    e = y - X @ beta_r
    meat = np.zeros((p, p))
    for idx, Theta, W in cache:
        Xi = X[idx]
        U = np.linalg.cholesky(Theta).T                    # upper: U'U = Theta
        IH = np.eye(len(idx)) - Xi @ M @ Xi.T @ W
        G = U @ IH @ Theta @ U.T
        ev, evec = np.linalg.eigh((G + G.T) / 2)
        ev_p = np.where(ev > 1e-12, ev ** -0.5, 0.0)
        A = U.T @ (evec * ev_p) @ evec.T @ U
        c = Xi.T @ W @ A @ e[idx]
        meat += np.outer(c, c)
    Vcr2 = M @ meat @ M
    out = {"coef_names": names, "beta_gls": beta.tolist(), "beta_lme4": beta_r.tolist(),
           "max_abs_beta_diff": float(np.max(np.abs(beta - beta_r))), "vcov_cr2": Vcr2.tolist(),
           "se_cr2": np.sqrt(np.diag(Vcr2)).tolist(), "se_model": np.sqrt(np.diag(M * sigma2)).tolist()}
    if targets:
        out["targets"] = {}
        for k, w in targets.items():
            L = np.array([float(w.get(nm, 0.0)) for nm in names])
            out["targets"][k] = {"estimate": float(L @ beta_r), "se_cr2": float(np.sqrt(L @ Vcr2 @ L))}
    return out
