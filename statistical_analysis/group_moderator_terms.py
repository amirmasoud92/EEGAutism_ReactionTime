"""FAST models with group terms for curvature, time on task, alpha lateralisation / central mu alpha and alpha-burst
occupancy (Additional file 1, Supplementary Methods S4, X1-X4).

Each model is the FAST group model (M3-H2, final M3 random-effects structure) with the added term and its ``:grp``
interaction, so that the TD and ASD values and their difference are estimated:

- ``X1_curvature``: + a_wb_sq + a_wb_sq:grp (a_wb squared, centred);
- ``X2_time``: + a_wb:t_run:grp (M3-H2 already has a_wb:t_run);
- ``X3_lateralisation``: + a_lat + a_lat:grp (left - right centro-parietal mu alpha, block-centred, per child SD);
- ``X3_mu_roi``: + a_mu + a_mu x grp (within-block centro-parietal mu alpha) next to posterior a_wb;
- ``X4_burst``: + burst_c, a_wb x burst_c and their products with grp (burst occupancy, child-centred).

Interactions with grp are explicit product columns (``<term>_x_grp``) so their names do not depend on R's
term ordering. Group-specific values: TD = term - 0.5 term:grp, ASD = term + 0.5 term:grp. Unadjusted for
multiplicity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling import sensitivity as SENS
from modeling.r_bridge import EngineError
from statistical_analysis import coupling_analyses as ST
from statistical_analysis.group_difference_checks import Run, _counts, _t

EXPL_COLS = ("burst_occ", "burst_valid", "alpha_e7_mu_left", "alpha_e7_mu_right", "valid_e7_mu_left",
             "valid_e7_mu_right")


def _by_group(term: str) -> dict:
    """Targets for ``term`` and its explicit product column ``term + '_x_grp'``."""
    gx = f"{term}_x_grp"
    return {f"{term}|TD": {term: 1.0, gx: -0.5}, f"{term}|ASD": {term: 1.0, gx: 0.5}, f"{term}|ASD-TD": {gx: 1.0}}


def _with_grp(md, cols: dict) -> tuple[dict, list]:
    """Each new column plus its product with grp as a plain column (R would reorder an ``x:grp`` label)."""
    g = md.frame["grp"].to_numpy(float)
    out, terms = {}, []
    for k, v in cols.items():
        out[k], out[f"{k}_x_grp"] = v, v * g
        terms += [k, f"{k}_x_grp"]
    return out, terms


def _fit(r: Run, md, label: str, targets: dict, state: dict) -> dict:
    tg = {"H2": {"a_wb:grp": 1.0}, "a_wb|TD": {"a_wb": 1.0, "a_wb:grp": -0.5},
          "a_wb|ASD": {"a_wb": 1.0, "a_wb:grp": 0.5}, **targets}
    try:
        res = r.fit(md, label, reduction=False, fixed_state=state, cr2=True, extra_targets=tg, subdir="expl")
    except EngineError as e:
        return {"estimable": False, "error": str(e)[:300]}
    return {"estimable": True, **_counts(md), "targets": {k: _t(res, k) for k in tg}}


def fast_models_with_group_terms(r: Run) -> dict:
    st = r.state("S2")
    base = r.provider.build("M3-H2", extra_predictor_columns=EXPL_COLS)
    out: dict = {}

    # X1 curvature
    v = base.frame["a_wb"].to_numpy(float) ** 2
    md = SENS._add(base, *_with_grp(base, {"a_wb_sq": v - v.mean()}))
    out["X1_curvature"] = _fit(r, md, "fast_curvature_M3_H2", _by_group("a_wb_sq"), st)

    # X2 time on task (linear a_wb x t_run by group)
    md = SENS._add(base, {}, ["a_wb:t_run:grp"])
    out["X2_time"] = _fit(r, md, "fast_time_M3_H2", {
        "a_wb:t_run|TD": {"a_wb:t_run": 1.0, "a_wb:t_run:grp": -0.5},
        "a_wb:t_run|ASD": {"a_wb:t_run": 1.0, "a_wb:t_run:grp": 0.5},
        "a_wb:t_run|ASD-TD": {"a_wb:t_run:grp": 1.0}}, st)

    # X3 lateralisation and the mu ROI
    f = base.frame
    if all(c in f for c in EXPL_COLS[2:]):
        ok = (pd.array(f["valid_e7_mu_left"], dtype="boolean").fillna(False).to_numpy(bool)
              & pd.array(f["valid_e7_mu_right"], dtype="boolean").fillna(False).to_numpy(bool))
        md = SENS._filter(base, ok, "X3: both mu halves valid")
        g = md.frame
        lat = g["alpha_e7_mu_left"].to_numpy(float) - g["alpha_e7_mu_right"].to_numpy(float)
        lat = lat - pd.Series(lat).groupby(g["pid_run_block"].to_numpy()).transform("mean").to_numpy()
        sd = pd.Series(lat).groupby(g["pid"].to_numpy()).transform(lambda s: s.std(ddof=1)).to_numpy()
        md = SENS._add(md, *_with_grp(md, {"a_lat": lat / sd}))
        out["X3_lateralisation"] = _fit(r, md, "fast_lateralisation_M3_H2", _by_group("a_lat"), st)
    else:
        out["X3_lateralisation"] = {"estimable": False, "error": "no E7 mu columns in the FAST predictor table"}
    # X3 mu: within-block centro-parietal mu alpha (mean of the E7 halves) next to posterior a_wb, by group
    if all(c in f for c in EXPL_COLS[2:]):
        ok = (pd.array(f["valid_e7_mu_left"], dtype="boolean").fillna(False).to_numpy(bool)
              & pd.array(f["valid_e7_mu_right"], dtype="boolean").fillna(False).to_numpy(bool))
        md = SENS._filter(base, ok, "X3: both mu halves valid")
        g = md.frame
        mu = 0.5 * (g["alpha_e7_mu_left"].to_numpy(float) + g["alpha_e7_mu_right"].to_numpy(float))
        mu = mu - pd.Series(mu).groupby(g["pid_run_block"].to_numpy()).transform("mean").to_numpy()
        sd = pd.Series(mu).groupby(g["pid"].to_numpy()).transform(lambda s: s.std(ddof=1)).to_numpy()
        md = SENS._add(md, *_with_grp(md, {"a_mu": mu / sd}))
        out["X3_mu_roi"] = _fit(r, md, "fast_mu_M3_H2", _by_group("a_mu"), st)
        out["X3_mu_roi"]["note"] = ("a_mu: mean of left / right centro-parietal mu alpha (E7 columns), block-centred, "
                                    "per child SD; fitted together with posterior a_wb (H2 = posterior a_wb:grp "
                                    "adjusted for mu)")
    else:
        out["X3_mu_roi"] = {"estimable": False, "error": "no E7 mu columns in the FAST predictor table"}

    # X4 burst occupancy as predictor and moderator, by group
    if "burst_occ" in f and "burst_valid" in f:
        ok = pd.array(f["burst_valid"], dtype="boolean").fillna(False).to_numpy(bool) \
            & np.isfinite(pd.to_numeric(f["burst_occ"], errors="coerce").to_numpy(float))
        md = SENS._filter(base, ok, "X4: burst_valid")
        b = ST._child_centre(md, "burst_occ")
        md = SENS._add(md, *_with_grp(md, {"burst_c": b, "awb_x_burst": md.frame["a_wb"].to_numpy(float) * b}))
        out["X4_burst"] = _fit(r, md, "fast_burst_M3_H2", {**_by_group("burst_c"), **_by_group("awb_x_burst")}, st)
    else:
        out["X4_burst"] = {"estimable": False, "error": "no burst columns in the FAST predictor table"}
    out["definition"] = ("FAST versions of E1, E6 (linear), E7 and E3 with :grp terms on M3-H2, final M3 "
                         "random-effects structure; group values TD = term - 0.5 term:grp, ASD = term + 0.5 "
                         "term:grp; unadjusted")
    return out
