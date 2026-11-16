"""Sensitivity power and the effect-size interval of the FAST group difference (descriptive; no data read).

Inputs: the FAST group-model difference (M3-H2 target ``a_wb:grp``: estimate, se, ci95, p) and the two-stage estimate
(hedges_g, n_TD, n_ASD) of :func:`statistical_analysis.follow_up_analyses.two_stage_permutation_leave_one_out`.
- Minimum detectable difference at 80% and 90% power, two-sided alpha 0.05 and 0.005, from the model SE (normal
  approximation): MDE = (z_{1-alpha/2} + z_{power}) x SE.
- 95% CI of Hedges g with the large-sample SE sqrt((n1 + n2)/(n1 n2) + g^2 / (2 (n1 + n2))).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats


def sensitivity_power(h2: Mapping, r1: Mapping) -> dict:
    """``h2``: the M3-H2 group-difference target row; ``r1``: the ``two_stage`` block of the two-stage analysis."""
    se = float(h2["se"])
    mde = {f"alpha{a}_power{p}": float((stats.norm.ppf(1 - a / 2) + stats.norm.ppf(p)) * se)
           for a in (0.05, 0.005) for p in (0.8, 0.9)}
    g, n1, n2 = float(r1["hedges_g"]), int(r1["n_TD"]), int(r1["n_ASD"])
    se_g = float(np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2))))
    out = {"definition": __doc__.strip().splitlines()[0],
           "fast_group_difference": {"estimate": float(h2["estimate"]), "se": se, "ci95": h2["ci95"], "p": h2["p"]},
           "minimum_detectable_difference_log_rt": mde,
           "hedges_g_two_stage": {"g": g, "se": se_g, "ci95": [g - 1.96 * se_g, g + 1.96 * se_g], "n_TD": n1,
                                  "n_ASD": n2}}
    return out
