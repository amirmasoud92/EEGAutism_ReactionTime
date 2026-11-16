"""One interface to the R model engine (:mod:`modeling.r_bridge` with ``R/fit_lmm.R`` / ``R/fit_glmm.R``), its
extras (:mod:`modeling.extras_bridge` / ``R/fit_extras.R``), the statsmodels second implementation and the M6
ex-Gaussian model (``R/fit_m6.R``). Model-input TSVs, specs and results are written under ``work``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import extras_bridge as XB
from modeling import r_bridge
from modeling.r_bridge import RSCRIPT, EngineError

__all__ = ["REngine", "M6_SCRIPT", "run_m6"]

M6_SCRIPT = Path(__file__).resolve().parent / "R" / "fit_m6.R"


# wall-clock budgets of the two M6 attempts (h); an attempt that does not finish counts as not converged
M6_FIRST_BUDGET_H = 24.0
M6_REFIT_BUDGET_H = 48.0


class REngine:
    """The R engine (see the module docstring)."""

    def __init__(self, work: str | os.PathLike[str], *, cfg=None, log: Callable[[str], None] | None = None):
        self.work = Path(work)
        self.cfg = cfg
        self.log = log or (lambda m: None)

    # ------------------------------------------------------------------ engine fits (lmm / glmm)
    def fit(self, md, label: str, *, reduction: bool = True, fixed_state: Mapping | None = None, cr2: bool = True,
            second_impl: bool | None = None, extra_targets: Mapping | None = None, export_fixef_vcov: bool = False,
            subdir: str = "fits") -> dict:
        """``r_bridge`` fit. ``second_impl`` defaults to the GLMM glmer check (the LMM second implementation is
        :meth:`sm_check`)."""
        out = self.work / subdir
        if second_impl is None:
            second_impl = md.model_def.family == "glmm"
        kw = {"reduction": reduction, "fixed_state": dict(fixed_state) if fixed_state else None,
              "cr2": cr2 and md.model_def.family == "lmm", "second_impl": bool(second_impl),
              "extra_targets": dict(extra_targets) if extra_targets else None}
        if export_fixef_vcov:
            kw["export_fixef_vcov"] = True
        spec_path, spec = r_bridge.prepare(md, out, label, **kw)
        res = r_bridge.run_spec(spec_path)
        if "error" in res:
            raise EngineError(f"{label}: {res['error']}")
        return res

    def sm_check(self, label: str, subdir: str = "fits") -> dict:
        """Second implementation (statsmodels MixedLM, >= 4 starts) on a finished LMM fit, compared term by term."""
        from modeling import second_impl

        out = self.work / subdir
        res = json.loads((out / f"{label}_result.json").read_text(encoding="utf-8"))
        spec = json.loads((out / f"{label}_spec.json").read_text(encoding="utf-8"))
        d = pd.read_csv(out / f"{label}_input.tsv", sep="\t")
        t0 = time.time()
        try:
            sm = second_impl.statsmodels_fit(d, spec["fixed_formula"], res["final_state"])
            cmp = second_impl.compare_with_lme4(sm, res)
            awb = next((r for r in cmp["rows"] if r["term"] == "a_wb"), None)
            r = {"label": label, "pass": bool(cmp["pass"]), "max_fe_diff_rel_se": cmp["max_fe_diff_rel_se"],
                 "loglik_diff_statsmodels_minus_lme4": cmp["loglik_diff"],
                 "se_ratio_a_wb": awb["se_ratio"] if awb else None, "se_ratio_range": cmp["se_ratio_range"],
                 "n_ok_starts": sm["n_ok"], "terms_compared": cmp["terms_compared"],
                 "terms_missing": cmp["terms_missing"], "seconds": time.time() - t0,
                 "tolerance": cmp["tolerance"],
                 "note_if_better": ("statsmodels log-likelihood exceeds lme4 by > 0.01: reported beside the lme4 "
                                    "fit as a labelled check") if cmp["loglik_diff"] > 0.01 else None}
        except Exception as e:  # noqa: BLE001 - recorded, never silently passed
            r = {"label": label, "pass": False, "error": f"{type(e).__name__}: {e}"[:400],
                 "seconds": time.time() - t0}
        return r

    # ------------------------------------------------------------------ extras
    def a1(self, md, label: str, *, targets: Mapping) -> dict:
        return XB.a1(md, self.work / "extras", label, targets=targets)

    def a3l(self, md, label: str) -> dict:
        return XB.a3l(md, self.work / "extras", label)

    def e3c(self, md, primary: Mapping, label: str, *, B: int, seed: int, ncpus: int = 1) -> dict:
        return XB.e3c(md, primary, self.work / "extras", label, B=B, seed=seed, ncpus=ncpus)

    def s_cens(self, md, label: str, *, random_string: str, iter: int, chains: int = 4) -> dict:
        return XB.s_cens(md, self.work / "extras", label, random_string=random_string, iter=iter, chains=chains,
                         cores=chains)

    def s_het(self, md, label: str, *, primary_state: Mapping | None) -> dict:
        from modeling import sensitivity as SENS

        return SENS._fit_het(md, self.work / "extras", label, primary_state)

    # ------------------------------------------------------------------ M6 (ex-Gaussian)
    def m6(self, md, label: str, *, mode: str, settings) -> dict:
        """M6 (``mode="full"``) or its odd/even-block split (``mode="split"``) with the fixed priors and sampler; the
        R script applies the convergence rule and the one refit itself."""
        return run_m6(md, self.work / "m6", label, mode=mode, settings=settings)


def run_m6(md, out_dir, label: str, *, mode: str, settings, first_budget_h: float = M6_FIRST_BUDGET_H,
           refit_budget_h: float = M6_REFIT_BUDGET_H) -> dict:
    """Write the TSV + spec and run ``fit_m6.R`` (same hand-over as ``extras_bridge.run_extra``).

    Wall-clock budgets: attempt 1 (``iter``, ``seed``) gets ``first_budget_h``; if it does not finish it is killed and
    counted as not converged, and the refit (``refit_iter``, ``refit_seed``) runs with ``refit_budget_h``. A refit
    that does not finish raises :class:`EngineError` (not estimable)."""
    from modeling.model_specs import fixed_formula

    f = md.frame
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = md.columns_needed()
    for c in ("block_order",):
        if c in f.columns and c not in cols:
            cols.append(c)
    d = f[cols].copy()
    d["rt_ms"] = np.exp(d["lrt"].to_numpy(float)) * 1000.0
    tsv = out / f"{label}_input.tsv"
    XB._write_tsv(d, tsv)
    terms = list(md.fixed_terms)
    spec = {"schema": "alpha_state.m6_spec/1", "label": label, "mode": mode, "fixed_terms": terms,
            "covariate_rhs": " + ".join(t for t in terms if t not in ("a_wb", "grp", "a_wb:grp")),
            "has_grp": "grp" in terms, "splines": [dict(s) for s in md.splines], "factors": ["pid"],
            "input_tsv": tsv.as_posix(), "output_json": (out / f"{label}_result.json").as_posix(),
            "ns_check_tol": 1e-8, "iter": int(settings.m6_iter), "warmup": int(settings.m6_iter) // 2,
            "refit_iter": int(settings.m6_refit_iter), "refit_warmup": int(settings.m6_refit_iter) // 2,
            "chains": int(settings.m6_chains), "seed": int(settings.m6_seed),
            "refit_seed": int(settings.m6_refit_seed), "adapt_delta": 0.95, "max_treedepth": 12,
            "fixed_formula_m1": fixed_formula("lrt", terms)}
    sp = out / f"{label}_spec.json"
    rp = Path(spec["output_json"])
    t0 = time.time()
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}

    def call(budget_h: float):
        sp.write_text(json.dumps(spec, indent=1, default=str), encoding="utf-8", newline="\n")
        if rp.exists():
            rp.unlink()
        # the whole process tree (rstan chain workers included) is killed on timeout
        p = subprocess.Popen([RSCRIPT, "--vanilla", str(M6_SCRIPT), str(sp)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
        try:
            _, err = p.communicate(timeout=budget_h * 3600)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            p.kill()
            p.communicate()
            return None
        return subprocess.CompletedProcess(p.args, p.returncode, "", err)

    r = call(first_budget_h)
    if r is None:
        spec["skip_first"] = (f"attempt 1 (iter {spec['iter']}, seed {spec['seed']}) did not finish within "
                              f"{first_budget_h:g} h; killed and counted as not converged")
        r = call(refit_budget_h)
        if r is None:
            raise EngineError(f"{label} (M6): not converged -- attempt 1 over its {first_budget_h:g} h budget and the "
                              f"refit did not finish within {refit_budget_h:g} h")
    if not rp.is_file():
        raise EngineError(f"fit_m6.R exited {r.returncode} without a result:\n{r.stderr[-3000:]}")
    res = json.loads(rp.read_text(encoding="utf-8"))
    res["wall_seconds"] = time.time() - t0
    if "error" in res:
        raise EngineError(f"{label} (M6): {res['error']}")
    return res
