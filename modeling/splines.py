"""Natural cubic spline basis identical to R's ``splines::ns`` (knots from ``configs/model_knots.yaml``).

The models call ``ns(x, knots = <interior>, Boundary.knots = <boundary>)`` with the knot table of
:mod:`modeling.knots`. The R engine builds that basis itself and uses it in the fit; this module builds the same matrix
in Python for the statsmodels second implementation and for the model-input TSV, and the R engine compares both bases
column by column on every fit (a fit is refused if they differ by more than 1e-8).

Algorithm of ``splines::ns`` (R 4.4, ``intercept = FALSE``):

1. ``Aknots`` = the boundary knots repeated 4 times plus the interior knots (cubic, order 4);
2. ``basis`` = the B-spline basis (``splineDesign``) at ``x`` inside the boundary knots; outside, the linear
   extrapolation ``B(b) + (x - b) B'(b)`` from the nearer boundary knot ``b``;
3. ``const`` = second derivatives of the basis at both boundary knots (2 x n_basis);
4. drop the first basis column (no intercept) from ``basis`` and ``const``;
5. ``basis`` <- ``t(qr.qty(qr(t(const)), t(basis)))[, -(1:2)]``, i.e. project onto the null space of the
   boundary second-derivative constraints with the Householder ``Q`` of LINPACK ``dqrdc2`` (R's default ``qr``).

The Householder reflections are computed exactly as LINPACK ``dqrdc2`` does (sign convention included), so the
result is the same matrix, not merely the same column space.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import BSpline

__all__ = ["ns_basis", "ns_column_names", "linpack_householder_qty"]


def _bspline_design(aknots: np.ndarray, x: np.ndarray, deriv: int = 0) -> np.ndarray:
    """B-spline basis (order 4) or its derivative at ``x`` for the full knot vector ``aknots`` (splineDesign)."""
    n_basis = len(aknots) - 4
    spl = BSpline(aknots, np.eye(n_basis), 3, extrapolate=True)
    if deriv:
        spl = spl.derivative(deriv)
    out = np.asarray(spl(np.asarray(x, float)), float)
    return out.reshape(len(np.atleast_1d(x)), n_basis)


def linpack_householder_qty(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``qr.qty(qr(a), y)`` with LINPACK ``dqrdc2`` Householder reflections (no pivoting needed for full rank).

    Parameters
    ----------
    a
        n x p matrix (full column rank).
    y
        n x m matrix.

    Returns
    -------
    Q' y (n x m).
    """
    a = np.array(a, float, copy=True)
    y = np.array(y, float, copy=True)
    n, p = a.shape
    for j in range(min(p, n - 1)):
        x = a[j:, j]
        nrm = np.linalg.norm(x)
        if nrm == 0.0:
            continue
        if x[0] != 0.0:
            nrm = np.copysign(nrm, x[0])
        v = x / nrm
        v[0] += 1.0
        # reflector H = I - v v' / v[0] (dqrdc2: qraux = v[0]); apply to the remaining columns of a and to y
        for k in range(j + 1, p):
            t = -np.dot(v, a[j:, k]) / v[0]
            a[j:, k] += t * v
        t = -(v @ y[j:, :]) / v[0]
        y[j:, :] += np.outer(v, t)
        a[j:, j] = v
    return y


def ns_basis(x: Sequence[float] | np.ndarray, interior_knots: Sequence[float],
             boundary_knots: Sequence[float]) -> np.ndarray:
    """``splines::ns(x, knots = interior_knots, Boundary.knots = boundary_knots)`` (no intercept).

    Non-finite ``x`` gives a row of NaN (R returns NA rows).
    """
    x = np.asarray(x, float)
    kn = np.sort(np.asarray(interior_knots, float))
    b = np.asarray(boundary_knots, float)
    if b.shape != (2,) or not b[0] < b[1]:
        raise ValueError("boundary_knots must be two increasing values")
    aknots = np.concatenate([np.repeat(b[0], 4), kn, np.repeat(b[1], 4)])
    n_basis = len(aknots) - 4
    out = np.full((len(x), n_basis), np.nan)
    fin = np.isfinite(x)
    lo = fin & (x < b[0])
    hi = fin & (x > b[1])
    inside = fin & ~lo & ~hi
    if inside.any():
        out[inside] = _bspline_design(aknots, x[inside])
    for mask, piv in ((lo, b[0]), (hi, b[1])):
        if mask.any():
            v0 = _bspline_design(aknots, np.array([piv]))[0]
            d1 = _bspline_design(aknots, np.array([piv]), deriv=1)[0]
            out[mask] = v0[None, :] + (x[mask] - piv)[:, None] * d1[None, :]
    const = _bspline_design(aknots, b, deriv=2)          # 2 x n_basis
    out = out[:, 1:]
    const = const[:, 1:]
    res = np.full((len(x), out.shape[1] - 2), np.nan)
    ok = np.isfinite(out).all(axis=1)
    if ok.any():
        qty = linpack_householder_qty(const.T, out[ok].T)   # (n_basis-1) x n_ok
        res[ok] = qty[2:, :].T
    return res


def ns_column_names(variable: str, df: int) -> list[str]:
    """Column names of the basis of ``ns(variable, df)`` in the model-input TSV: ``ns_<variable>_<df>_<k>``."""
    return [f"ns_{variable}_{df}_{k}" for k in range(1, df + 1)]
