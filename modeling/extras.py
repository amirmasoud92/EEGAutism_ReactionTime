"""Inference checks of the coupling estimates that need no mixed model: A2 (circular-shift surrogate), A3
(detrended CR2), A2-L (statistic), A3-NL, and the per-child residualisation they share. A1 (nlme AR(1)), A3-L
(logistic + clubSandwich CR2) and the R cross-checks are in ``R/fit_extras.R`` (:mod:`modeling.extras_bridge`).
Additional file 1, Table S1a (notes).

- **Per-child residualisation (A2, A3).** Within each child, ``lrt`` and ``a_wb`` are residualised by OLS on the
  model's fixed covariates without the alpha terms (every term containing ``a_wb``, ``a_slow``, ``a_hist5``,
  ``a_wb_raw``, ``a_harm``; constant child-level columns are absorbed), plus child-specific block dummies (the
  decomposition block, ``pid_run_block``) and linear ``t_run`` and ``tib``. Rank deficiency is handled by the
  minimum-norm least-squares solution (the residual is the orthogonal projection either way).
- **A2** (within-child circular-shift surrogate null). Statistic = the unweighted mean of the per-child OLS slopes
  of the residualised ``lrt`` on the residualised ``a_wb`` (with intercept, i.e. centred within child; the residuals
  have mean 0 already). Each of ``n_surrogates`` (10,000) surrogates applies one random non-zero circular shift per
  child to the child's time-ordered alpha-residual series (rows ordered by run and sample, runs concatenated), and
  ``p = (1 + #{|null| >= |observed|}) / (1 + n_surrogates)`` (two-sided). For a child with n rows all n - 1 shifted
  slopes are computed at once by FFT circular cross-correlation, with row masks for the subset statistics (H1-A:
  auditory rows; S5: V minus A rows); :func:`shift_slopes_explicit` recomputes them by explicit ``np.roll`` and every
  call verifies a random sample of surrogates against it.
- **A3** CR2 cluster-robust SE of the pooled OLS within-child slope after the same residualisation, closed form
  (single regressor: ``V = sum_g (x_g' e_g)^2 / (1 - h_g) / S^2``, ``h_g = x_g' x_g / S``, ``S = sum x^2``;
  Bell-McCaffrey with the OLS working model), df = number of children - 1. The general CR2 of :func:`cr2_ols` (any
  small number of regressors) gives the contrasts (S5 V - A; S4 FAST/IC mean) and reproduces the closed form exactly
  for one regressor and clubSandwich ``vcovCR(type = "CR2")``.
- **A3-NL** (S6): CR2 of the per-child slope of log RT on ``a_hist5`` after per-child residualisation of both on
  block dummies, ``ns(t_run, 3)``, ``ns(tib, 3)`` (the M1 knots) and ``a_wb``; df = children - 1.
- **A2-L** statistic (S1): the mean over children of ``sum x_c (miss - p0) / sum x_c^2``, ``x`` = ``a_wb`` detrended per
  child on linear ``t_run`` and ``tib`` (``x_c`` centred within child), ``p0`` from the A3-L nuisance model without
  alpha (``R/fit_extras.R``); the same circular-shift surrogate of ``x`` against ``miss - p0``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["ALPHA_TOKENS", "nuisance_terms", "residualise_per_child", "shift_slopes_fft", "shift_slopes_explicit",
           "a2_surrogate", "cr2_ols", "cr2_single", "a3_cr2", "a3nl", "a2_a3_s4", "a2l_stat", "a2l_surrogate",
           "child_series", "a2_a3_model", "AUDITORY", "VISUAL"]

ALPHA_TOKENS = ("a_wb", "a_slow", "a_hist5", "a_wb_raw", "a_harm", "a_cal")
N_SURROGATES = 10_000


def _uses_alpha(term: str) -> bool:
    return any(p in ALPHA_TOKENS or p.startswith(("a_wb", "a_cal")) for p in term.split(":"))


def nuisance_terms(fixed_terms: Sequence[str]) -> list[str]:
    """The model's fixed covariates without the alpha terms (A2)."""
    return [t for t in fixed_terms if not _uses_alpha(t)]


def _term_column(frame: pd.DataFrame, term: str) -> np.ndarray:
    return np.prod([frame[p].to_numpy(float) for p in term.split(":")], axis=0)


def residualise_per_child(frame: pd.DataFrame, cols: Sequence[str], nuisance: Sequence[str], *,
                          block_col: str | None = "pid_run_block", linear: Sequence[str] = ("t_run", "tib"),
                          child_col: str = "pid", extra_design: Mapping[str, np.ndarray] | None = None
                          ) -> pd.DataFrame:
    """Per-child OLS residuals of ``cols`` on ``nuisance`` terms + block dummies + ``linear`` (+ intercept).

    Returns a frame with the same index and the residual columns (named as ``cols``). ``extra_design`` adds
    precomputed design columns (e.g. spline bases) by name.
    """
    out = pd.DataFrame(index=frame.index, columns=list(cols), dtype=float)
    nz = {t: _term_column(frame, t) for t in nuisance}
    lin = {c: frame[c].to_numpy(float) for c in linear}
    ext = {k: np.asarray(v, float) for k, v in (extra_design or {}).items()}
    Y = frame[list(cols)].to_numpy(float)
    kids = frame[child_col].to_numpy()
    blocks = frame[block_col].astype(str).to_numpy() if block_col else None
    for g in pd.unique(kids):
        idx = np.flatnonzero(kids == g)
        parts = []
        if blocks is not None:
            b = blocks[idx]
            ub = pd.unique(b)
            parts.append((b[:, None] == ub[None, :]).astype(float))
        else:
            parts.append(np.ones((len(idx), 1)))
        for dct in (lin, nz, ext):
            if dct:
                parts.append(np.column_stack([v[idx] for v in dct.values()]))
        Z = np.column_stack(parts)
        coef, *_ = np.linalg.lstsq(Z, Y[idx], rcond=None)
        out.iloc[idx, :] = Y[idx] - Z @ coef
    return out.astype(float)


# --------------------------------------------------------------------------- circular-shift slopes
def _ccorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c[k] = sum_t a[(t - k) mod n] b[t] = sum_u a[u] b[(u + k) mod n] (FFT)."""
    return np.real(np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b)))


def shift_slopes_fft(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """OLS slope (with intercept) of ``y`` on ``np.roll(x, k)`` over the rows in ``mask``, for every k = 0..n-1.

    k = 0 is the observed slope. With ``mask`` all True and ``x`` centred this is ``cc[k + 1] / sum(x^2)`` of the
    circular cross-correlation ``cc``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
    nm = m.sum()
    yc = np.where(m, y - y[m].mean(), 0.0)
    num = _ccorr(x, yc)
    s1 = _ccorr(x, m.astype(float))
    s2 = _ccorr(x * x, m.astype(float))
    den = s2 - s1 * s1 / nm
    return num / den


def shift_slopes_explicit(x: np.ndarray, y: np.ndarray, shifts: Sequence[int], mask: np.ndarray | None = None
                          ) -> np.ndarray:
    """The same slopes by explicit circular shifts (``np.roll``) and ``np.polyfit``-free OLS (verification)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
    out = []
    for k in shifts:
        xs = np.roll(x, int(k))[m]
        ys = y[m]
        xc = xs - xs.mean()
        out.append(float(np.sum(xc * (ys - ys.mean())) / np.sum(xc * xc)))
    return np.asarray(out)


def child_series(frame: pd.DataFrame, child_col: str = "pid") -> list[np.ndarray]:
    """Row indices of each child in time order (run, sample), children in sorted order."""
    order = np.lexsort((frame["sample"].to_numpy(), frame["run"].to_numpy(), frame[child_col].to_numpy()))
    kids = frame[child_col].to_numpy()[order]
    out = []
    for g in pd.unique(kids):
        out.append(order[kids == g])
    return out


def a2_surrogate(per_child: Sequence[Mapping], *, n_surrogates: int = N_SURROGATES, seed: int,
                 combine=None, n_verify: int = 20) -> dict:
    """Circular-shift surrogate test of an average of per-child slopes.

    Parameters
    ----------
    per_child
        One mapping per child with ``x`` (alpha residual, time order), ``y`` and optionally ``masks`` (a dict name ->
        boolean mask, default ``{"all": all rows}``) and ``weights`` (name -> weight; the child's statistic is
        ``sum_k w_k slope_k``; default 1 for ``"all"``). Children without a usable statistic are skipped.
    combine
        Optional ``f(stat_by_child: ndarray (n_children,), group: ndarray) -> float``; default the unweighted mean.
        Each child may carry ``group`` (e.g. the task for S4).
    n_verify
        Number of surrogate draws whose per-child slopes are recomputed by explicit ``np.roll`` (must agree to 1e-9).
    """
    rng = np.random.default_rng(seed)
    obs_c, null_c, groups, verify_max = [], [], [], 0.0
    vrng = np.random.default_rng(seed + 1)
    for c in per_child:
        x, y = np.asarray(c["x"], float), np.asarray(c["y"], float)
        n = len(x)
        masks = c.get("masks") or {"all": np.ones(n, bool)}
        w = c.get("weights") or {k: 1.0 for k in masks}
        if n < 3 or any(np.asarray(masks[k], bool).sum() < 3 for k in w):
            continue
        sl = {k: shift_slopes_fft(x, y, masks[k]) for k in w}
        tot = sum(w[k] * sl[k] for k in w)                     # (n,) statistic for every shift
        if not np.isfinite(tot).all():
            continue
        shifts = rng.integers(1, n, n_surrogates)              # one random non-zero shift per surrogate
        obs_c.append(tot[0])
        null_c.append(tot[shifts])
        groups.append(c.get("group", 0))
        for k in vrng.integers(1, n, min(n_verify, n - 1)):
            ex = sum(w[j] * shift_slopes_explicit(x, y, [k], masks[j])[0] for j in w)
            verify_max = max(verify_max, abs(ex - tot[k]))
    obs_c = np.asarray(obs_c)
    null_m = np.vstack(null_c)                                  # children x surrogates
    groups = np.asarray(groups)
    if combine is None:
        obs = float(obs_c.mean())
        null = null_m.mean(axis=0)
    else:
        obs = float(combine(obs_c, groups))
        null = np.array([combine(null_m[:, s], groups) for s in range(null_m.shape[1])])
    p = (1 + int(np.sum(np.abs(null) >= abs(obs)))) / (1 + len(null))
    if verify_max > 1e-9:
        raise AssertionError(f"FFT shift slopes differ from explicit shifts by {verify_max:g}")
    return {"statistic": obs, "p": float(p), "n_surrogates": int(len(null)), "n_children": int(len(obs_c)),
            "null_sd": float(np.std(null, ddof=1)), "null_q025_q975": [float(np.quantile(null, 0.025)),
                                                                       float(np.quantile(null, 0.975))],
            "seed": int(seed), "fft_vs_explicit_max_abs_diff": float(verify_max),
            "method": "within-child circular-shift surrogate (FFT), p = (1 + #{|null| >= |obs|}) / (1 + B)"}


# --------------------------------------------------------------------------- CR2
def cr2_ols(X: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> dict:
    """OLS with the CR2 cluster-robust covariance (Bell-McCaffrey, working model OLS; = clubSandwich CR2 on lm).

    ``A_g = (I - X_g M X_g')^{-1/2}``, ``M = (X'X)^-1``, ``V = M (sum X_g' A_g e_g e_g' A_g X_g) M``.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    M = np.linalg.inv(X.T @ X)
    beta = M @ X.T @ y
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = 0
    for g in pd.unique(cluster):
        idx = np.flatnonzero(cluster == g)
        Xg = X[idx]
        H = Xg @ M @ Xg.T
        ev, evec = np.linalg.eigh(np.eye(len(idx)) - (H + H.T) / 2)
        ev_p = np.where(ev > 1e-12, ev ** -0.5, 0.0)
        A = (evec * ev_p) @ evec.T
        u = Xg.T @ (A @ e[idx])
        meat += np.outer(u, u)
        G += 1
    V = M @ meat @ M
    return {"beta": beta, "vcov": V, "n_clusters": G}


def cr2_single(x: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> dict:
    """Closed-form CR2 for a single regressor without intercept (A3)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    S = float(np.sum(x * x))
    b = float(np.sum(x * y) / S)
    e = y - b * x
    v = 0.0
    G = 0
    for g in pd.unique(cluster):
        m = cluster == g
        xe = float(np.sum(x[m] * e[m]))
        h = float(np.sum(x[m] * x[m])) / S
        v += xe * xe / (1.0 - h)
        G += 1
    return {"estimate": b, "se": math.sqrt(v) / S, "n_clusters": G}


def _t_row(est: float, se: float, df: float) -> dict:
    t = est / se
    q95, q90 = stats.t.ppf(0.975, df), stats.t.ppf(0.95, df)
    return {"estimate": float(est), "se": float(se), "df": float(df), "stat": float(t),
            "p": float(2 * stats.t.sf(abs(t), df)), "ci95": [est - q95 * se, est + q95 * se],
            "ci90": [est - q90 * se, est + q90 * se]}


def a3_cr2(xr: np.ndarray, yr: np.ndarray, cluster: np.ndarray, *, mask: np.ndarray | None = None,
           contrast: Mapping[str, Sequence] | None = None) -> dict:
    """A3: CR2 of the pooled within-child slope (closed form), df = children - 1.

    ``contrast`` = ``{"columns": [mask1, mask2, ...], "weights": [w1, w2, ...]}`` fits ``y ~ 0 + x*mask_k`` and
    returns the CR2 contrast (general :func:`cr2_ols`); ``mask`` restricts the single-regressor version.
    """
    xr, yr, cluster = np.asarray(xr, float), np.asarray(yr, float), np.asarray(cluster)
    if contrast is None:
        m = np.ones(len(xr), bool) if mask is None else np.asarray(mask, bool)
        x, y, cl = xr[m], yr[m], cluster[m]
        if mask is not None:                       # re-centre within child on the subset
            x = x - pd.Series(x).groupby(cl).transform("mean").to_numpy()
            y = y - pd.Series(y).groupby(cl).transform("mean").to_numpy()
        r = cr2_single(x, y, cl)
        out = _t_row(r["estimate"], r["se"], r["n_clusters"] - 1)
        out["n_clusters"] = r["n_clusters"]
        out["method"] = "CR2 closed form (single regressor), df = children - 1"
        return out
    cols = []
    for mk in contrast["columns"]:
        mk = np.asarray(mk, bool)
        xc = np.where(mk, xr, 0.0)
        cols.append(xc)
    X = np.column_stack(cols)
    keep = X.any(axis=1) | np.zeros(len(xr), bool)
    r = cr2_ols(X[keep], yr[keep], cluster[keep])
    w = np.asarray(contrast["weights"], float)
    est = float(w @ r["beta"])
    se = float(np.sqrt(w @ r["vcov"] @ w))
    out = _t_row(est, se, r["n_clusters"] - 1)
    out["n_clusters"] = r["n_clusters"]
    out["coefficients"] = [float(v) for v in r["beta"]]
    out["method"] = "CR2 (general, OLS working model), contrast, df = children - 1"
    return out


# --------------------------------------------------------------------------- model-level wrappers
AUDITORY = {"cVA": -0.5, "cAV": -1.0 / 3.0}
VISUAL = {"cVA": 0.5, "cAV": -1.0 / 3.0}


def _is(frame: pd.DataFrame, codes: Mapping[str, float]) -> np.ndarray:
    m = np.ones(len(frame), bool)
    for c, v in codes.items():
        m &= np.isclose(frame[c].to_numpy(float), v)
    return m


def a2_a3_model(md, *, target: str = "P1", x_col: str = "a_wb", y_col: str | None = None, seed: int,
                n_surrogates: int = N_SURROGATES, extra_nuisance: Sequence[str] = ()) -> dict:
    """A2 and A3 for one target of a model's data.

    ``target``: ``"P1"`` / ``"S2"`` / ``"S3"`` (all rows), ``"H1A"`` (auditory rows), ``"S5"`` (V minus A slopes),
    ``"S6"`` (``x_col = "a_hist5"`` with ``a_wb`` added to the nuisance).
    """
    f = md.frame
    y_col = y_col or md.outcome_column
    nuis = nuisance_terms(md.fixed_terms) + list(extra_nuisance)
    if target == "S6" and "a_wb" not in nuis:
        nuis.append("a_wb")
    R = residualise_per_child(f, [y_col, x_col], nuis)
    xr, yr = R[x_col].to_numpy(), R[y_col].to_numpy()
    pid = f["pid"].to_numpy()
    per = []
    masks_all = None
    if target == "H1A":
        masks_all = {"A": _is(f, AUDITORY)}
    elif target == "S5":
        masks_all = {"V": _is(f, VISUAL), "A": _is(f, AUDITORY)}
    for idx in child_series(f):
        rec = {"x": xr[idx], "y": yr[idx]}
        if masks_all is not None:
            rec["masks"] = {k: v[idx] for k, v in masks_all.items()}
            rec["weights"] = {"A": 1.0} if target == "H1A" else {"V": 1.0, "A": -1.0}
        per.append(rec)
    a2 = a2_surrogate(per, n_surrogates=n_surrogates, seed=seed)
    if target == "H1A":
        a3 = a3_cr2(xr, yr, pid, mask=masks_all["A"])
    elif target == "S5":
        a3 = a3_cr2(xr, yr, pid, contrast={"columns": [masks_all["V"], masks_all["A"]], "weights": [1.0, -1.0]})
    else:
        a3 = a3_cr2(xr, yr, pid)
    return {"target": target, "x": x_col, "y": y_col, "nuisance_terms": nuis, "A2": a2, "A3": a3,
            "residualisation": "per child OLS: nuisance terms + block dummies + linear t_run, tib (+ intercept)"}


def a3nl(md, *, x_col: str = "a_hist5", y_col: str | None = None) -> dict:
    """A3-NL (S6): CR2 of the slope of log RT on ``a_hist5`` after per-child residualisation on block dummies,
    ``ns(t_run, 3)``, ``ns(tib, 3)`` (the M1 knots: the ``ns_t_run_3_*`` / ``ns_tib_3_*`` columns of the model data)
    and ``a_wb``; df = children - 1."""
    f = md.frame
    y_col = y_col or md.outcome_column
    cols = [c for c in f.columns if c.startswith(("ns_t_run_3_", "ns_tib_3_"))]
    if len(cols) != 6:
        raise KeyError("A3-NL needs the ns(t_run, 3) and ns(tib, 3) basis columns of the model data")
    ext = {c: f[c].to_numpy(float) for c in cols}
    ext["a_wb"] = f["a_wb"].to_numpy(float)
    R = residualise_per_child(f, [y_col, x_col], [], linear=(), extra_design=ext)
    out = a3_cr2(R[x_col].to_numpy(), R[y_col].to_numpy(), f["pid"].to_numpy())
    out["method"] = "A3-NL: CR2 after per-child residualisation on block dummies, ns(t_run, 3), ns(tib, 3), a_wb"
    return out


def a2_a3_s4(parts: Mapping[str, object], *, seed: int, n_surrogates: int = N_SURROGATES,
             s_task: Mapping[str, float] | None = None) -> dict:
    """A2 / A3 for S4 = (beta_F + beta_I) / 2 in r units: per task, the A2 residualisation of ``lrt`` and
    ``a_wb`` on that task's model covariates, ``lrt`` divided by the fixed ``s_task``; A2 statistic = half the sum of
    the two task means of the per-child slopes (one independent random shift per child x task); A3 = CR2 of the
    contrast in the stacked regression ``y ~ 0 + x_F + x_I`` clustered by child (children in both tasks are one
    cluster), df = children - 1."""
    from modeling.model_specs import S_TASK

    st = dict(s_task or S_TASK)
    per, xs, ys, pids, task_of = [], [], [], [], []
    for gi, task in enumerate(("FAST", "IC")):
        md = parts[task]
        f = md.frame
        R = residualise_per_child(f, ["lrt", "a_wb"], nuisance_terms(md.fixed_terms))
        x, y = R["a_wb"].to_numpy(), R["lrt"].to_numpy() / st[task]
        for idx in child_series(f):
            per.append({"x": x[idx], "y": y[idx], "group": gi})
        xs.append(x)
        ys.append(y)
        pids.append(f["pid"].to_numpy())
        task_of.append(np.full(len(f), gi))

    def comb(v, g):
        return 0.5 * (v[g == 0].mean() + v[g == 1].mean())

    a2 = a2_surrogate(per, n_surrogates=n_surrogates, seed=seed, combine=comb)
    x, y, pid, tk = map(np.concatenate, (xs, ys, pids, task_of))
    a3 = a3_cr2(x, y, pid, contrast={"columns": [tk == 0, tk == 1], "weights": [0.5, 0.5]})
    return {"target": "S4", "A2": a2, "A3": a3}


def a2l_stat(x: np.ndarray, r: np.ndarray) -> float:
    xc = x - x.mean()
    return float(np.sum(xc * r) / np.sum(xc * xc))


def a2l_surrogate(md, p0: np.ndarray, *, seed: int, n_surrogates: int = N_SURROGATES,
                  children: Sequence[str] | None = None) -> dict:
    """A2-L (S1): ``x`` = ``a_wb`` detrended per child on linear ``t_run`` and ``tib``; statistic = mean over
    children of ``sum x_c (miss - p0) / sum x_c^2``; circular-shift surrogate. ``p0`` (row order of ``md.frame``,
    NaN for children without a miss) from the A3-L nuisance model; ``children`` = the children kept by A3-L."""
    f = md.frame
    R = residualise_per_child(f, ["a_wb"], [], block_col=None)
    x = R["a_wb"].to_numpy()
    resid = f["miss"].to_numpy(float) - np.asarray(p0, float)
    keep = set(children) if children is not None else None
    per = []
    for idx in child_series(f):
        if keep is not None and f["pid"].iloc[idx[0]] not in keep:
            continue
        if not np.isfinite(resid[idx]).all():
            continue
        xc = x[idx] - x[idx].mean()
        # the masked-slope machinery with y = miss - p0 (centred within child is harmless: sum x_c = 0)
        per.append({"x": xc, "y": resid[idx]})
    out = a2_surrogate(per, n_surrogates=n_surrogates, seed=seed)
    out["method"] = "A2-L: circular-shift surrogate of x against miss - p0 (FFT)"
    return out
