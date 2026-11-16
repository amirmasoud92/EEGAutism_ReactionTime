"""Inference rules on model estimates: equivalence tests (TOST) against the SESOI, Benjamini-Hochberg and Holm
adjustment, true-score bounds, estimate categories, robustness labels, group-difference categories and the
cross-task contrasts.

Pure functions on estimates (no data, no model fitting). An :class:`Estimate` is a point estimate with its SE and
df (``None`` / ``inf`` = Wald z, the GLMM), as returned by the R engine for every target (``targets[name]``).

- **Single estimates**: categories *positive* (95% CI excludes 0, beta > 0; "at least the SESOI" if the one-sided
  minimum-effect test at 0.025 is significant, i.e. the 95% CI lower bound exceeds +SESOI), *practically null* (90% CI
  inside +-SESOI, i.e. TOST at 0.05; reported with the true-score bound), *opposite* (95% CI excludes 0, beta < 0),
  *inconclusive* (none). H1-A (auditory trials) is interpreted only if P1 is positive.
- **Family of six** S1-S6 (m = 6 always; a member that cannot be estimated enters with p = 1 and TOST p = 1):
  Benjamini-Hochberg at q = 0.05 over the two-sided p-values for every directional claim; BH over the TOST p-values
  (the larger one-sided p) for "practically null"; Benjamini-Yekutieli FCR intervals at level 1 - 0.05 R / m for the
  selected tests; the S6 co-drift conjunction (BH-significant M1-hist test and A3-NL positive at p < 0.05).
- **True-score bound**: SESOI / sqrt(rho_bar), rho_bar = 0.642 (pooled AVSRT split-time reliability).
- **Robustness labels**: "sensitive to X" (|X - estimate| >= SESOI / 2, or X's 95% CI excludes the estimate) for X in
  a fixed list; "inference-sensitive to X" for A1 / A2 / A3 if X's p-value or CI would change the category.
- **Group difference** Delta = beta_ASD - beta_TD: "CI excludes 0" / "CI inside +-SESOI" / "inconclusive".
- **Cross-task contrasts**: Holm over C1 and C2, equivalence margin +-0.03 r (90% CIs).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["Estimate", "RHO_BAR", "ROBUSTNESS_X", "INFERENCE_X", "tost_p", "min_effect_p", "true_score_bound",
           "primary_category", "primary_family", "bh_adjust", "holm_adjust", "secondary_family",
           "secondary_inference_sensitive", "QUALIFIER_APPLICABILITY", "qualifier_applies",
           "robustness_qualifiers", "h2_category", "cross_task"]

RHO_BAR = 0.642            # pooled AVSRT split-time reliability
ALPHA = 0.05
#: Robustness analyses of the robustness-label rule (S-OCU is descriptive and excluded).
ROBUSTNESS_X = ("S-HH", "S-PRESS", "S-SOA", "S-W500", "R-AP", "S-EST1", "S-BLINK", "S-ICA", "S-GAZE", "S-HARM",
                "S-CENS")
INFERENCE_X = ("A1", "A2", "A3")
#: Which test each robustness analysis applies to. The threshold is half of that test's SESOI; S-CENS uses its 95%
#: equal-tailed posterior interval.
QUALIFIER_APPLICABILITY: dict[str, tuple[str, ...]] = {
    **{x: ("P1", "H1A") for x in ("S-PRESS", "S-SOA", "S-W500", "S-BLINK", "S-CENS")},      # AVSRT-only
    **{x: ("P1", "S2", "S3") for x in ("S-HH", "R-AP", "S-EST1", "S-ICA")},                  # task-general
    "S-HARM": ("P1", "S2", "S3", "S4"),                                                      # S4 via M5h
    "S-GAZE": ("P1", "S2"),
}
assert set(QUALIFIER_APPLICABILITY) == set(ROBUSTNESS_X)


def qualifier_applies(x: str, test: str) -> bool:
    """Whether robustness analysis ``x`` qualifies ``test`` (P1, H1A, S2, S3, S4; :data:`QUALIFIER_APPLICABILITY`)."""
    if x not in QUALIFIER_APPLICABILITY:
        raise ValueError(f"{x} is not in the robustness list")
    return test in QUALIFIER_APPLICABILITY[x]


@dataclass(frozen=True)
class Estimate:
    """Point estimate with SE and df (``None`` or ``inf``: normal reference, Wald z)."""

    estimate: float
    se: float
    df: float | None = None

    @classmethod
    def from_result(cls, row: Mapping) -> Estimate:
        df = row.get("df")
        return cls(float(row["estimate"]), float(row["se"]), None if df is None else float(df))

    @property
    def _dist(self):
        return stats.norm if (self.df is None or not math.isfinite(self.df)) else stats.t(self.df)

    def ci(self, level: float) -> tuple[float, float]:
        q = self._dist.ppf(1 - (1 - level) / 2)
        return self.estimate - q * self.se, self.estimate + q * self.se

    @property
    def ci95(self) -> tuple[float, float]:
        return self.ci(0.95)

    @property
    def ci90(self) -> tuple[float, float]:
        return self.ci(0.90)

    @property
    def p(self) -> float:
        return float(2 * self._dist.sf(abs(self.estimate / self.se)))


def tost_p(e: Estimate, sesoi: float) -> float:
    """TOST p: the larger of the one-sided p-values against -SESOI and +SESOI."""
    d = e._dist
    p_low = float(d.sf((e.estimate + sesoi) / e.se))        # H0: beta <= -SESOI
    p_high = float(d.cdf((e.estimate - sesoi) / e.se))      # H0: beta >= +SESOI
    return max(p_low, p_high)


def min_effect_p(e: Estimate, sesoi: float) -> float:
    """One-sided minimum-effect test of H0: beta <= +SESOI (significant at 0.025 iff the 95% CI lower bound exceeds
    +SESOI)."""
    return float(e._dist.sf((e.estimate - sesoi) / e.se))


def true_score_bound(sesoi: float, rho_bar: float = RHO_BAR) -> float:
    """SESOI / sqrt(rho_bar) on the slope scale."""
    return float(sesoi / math.sqrt(rho_bar))


def _inside(ci: Sequence[float], bound: float) -> bool:
    return (-bound < ci[0]) and (ci[1] < bound)


def primary_category(e: Estimate, sesoi: float, *, rho_bar: float = RHO_BAR, ci95: Sequence[float] | None = None,
                     ci90: Sequence[float] | None = None) -> dict:
    """Category of a single estimate such as P1 / H1-A (unadjusted). ``ci95`` / ``ci90`` override the intervals
    (e.g. the engine's)."""
    c95 = tuple(ci95) if ci95 is not None else e.ci95
    c90 = tuple(ci90) if ci90 is not None else e.ci90
    positive = c95[0] > 0
    opposite = c95[1] < 0
    null = _inside(c90, sesoi)
    at_least = positive and c95[0] > sesoi
    if positive and null:
        cat = "positive"
        reading = "detectable, but below the SESOI"
    elif positive:
        cat, reading = "positive", ("at least the SESOI" if at_least else "")
    elif opposite:
        cat, reading = "opposite", ("detectable (opposite direction), but below the SESOI" if null else "")
    elif null:
        cat, reading = "practically_null", ""
    else:
        cat, reading = "inconclusive", ""
    cats = [c for c, f in (("positive", positive), ("practically_null", null), ("opposite", opposite)) if f]
    out = {"category": cat, "categories": cats or ["inconclusive"], "positive": positive, "practically_null": null, "opposite": opposite,
           "at_least_sesoi": at_least, "reading": reading, "estimate": e.estimate, "se": e.se, "df": e.df,
           "p": e.p, "ci95": list(c95), "ci90": list(c90), "sesoi": sesoi, "tost_p": tost_p(e, sesoi),
           "min_effect_p": min_effect_p(e, sesoi)}
    if null:
        out["true_score_bound"] = true_score_bound(sesoi, rho_bar)
        out["true_score_bound_note"] = f"SESOI {sesoi:g} on the observed scale = +-{out['true_score_bound']:.4f} " \
                                       f"on the true-score scale (rho_bar {rho_bar:g})"
    return out


def primary_family(p1: Estimate, h1a: Estimate, sesoi: float, *, rho_bar: float = RHO_BAR) -> dict:
    """P1 then H1-A (auditory trials), H1-A tested only if P1 is positive.

    The auditory reading ("present for auditory targets") requires P1 positive and H1-A positive. With P1 positive
    and H1-A practically null, inconclusive or opposite the reading is "visual-trial coupling; an ocular account is
    not excluded". Whether H1-A's 95% CI includes the P1 estimate is reported descriptively
    (``h1a_ci_includes_p1``)."""
    c1 = primary_category(p1, sesoi, rho_bar=rho_bar)
    out = {"P1": c1}
    ca = primary_category(h1a, sesoi, rho_bar=rho_bar)
    lo, hi = ca["ci95"]
    # descriptive only: never supports the auditory reading
    ca["h1a_ci_includes_p1"] = bool(lo <= p1.estimate <= hi)
    ca["h1a_ci_includes_p1_note"] = ("the auditory slope is not distinguishable from the overall slope"
                                     if ca["h1a_ci_includes_p1"] else
                                     "the auditory slope's 95% CI excludes the overall slope")
    if c1["category"] == "positive":
        ca["tested"] = True
        if ca["category"] == "positive":
            reading = ("the coupling is present for auditory targets, so loss of visual input (eye closure, gaze off "
                       "the screen) does not explain it")
        else:                           # practically null, inconclusive or opposite
            reading = "visual-trial coupling; an ocular account is not excluded"
        out["H1A_reading"] = reading
    else:
        ca = {**ca, "tested": False, "category": "estimation_only",
              "note": "H1-A is tested only if P1 is positive (fixed sequence); reported as estimation"}
    out["H1A"] = ca
    return out


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up; R ``p.adjust(method = "BH")``)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)[::-1]
    ranked = p[o] * m / np.arange(m, 0, -1)
    adj = np.minimum.accumulate(ranked)
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def holm_adjust(p: Sequence[float]) -> np.ndarray:
    """Holm adjusted p-values (R ``p.adjust(method = "holm")``)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (m - np.arange(m)))
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


SECONDARY = ("S1", "S2", "S3", "S4", "S5", "S6")


def secondary_family(members: Mapping[str, Estimate | None], sesoi: Mapping[str, float], *, q: float = ALPHA,
                     a3nl_s6: Estimate | None = None, rho_bar: float = RHO_BAR,
                     p_override: Mapping[str, float] | None = None) -> dict:
    """Family of six tests S1-S6 with Benjamini-Hochberg correction.

    m = 6 always: a member that is absent from ``members`` or ``None`` (it could not be estimated) enters BH with
    p = 1 and TOST p = 1 and is reported with category ``"not_estimable"``; the adjusted p-values of the other five
    are computed with m = 6.

    Parameters
    ----------
    members
        The six estimates (S1 log-OR from M2; S2 / S3 slopes; S4 r units from M5h; S5 ``a_wb:cVA``; S6 ``a_hist5``).
    sesoi
        SESOI per member.
    a3nl_s6
        A3-NL estimate of ``a_hist5`` (CR2, df = children - 1) for the S6 conjunction rule.
    p_override
        Member -> two-sided p-value used in place of the member's model p in the directional BH step (the
        "inference-sensitive to X" rule for an inference-only X that yields a p-value only, e.g. A2); the other
        members are unchanged (:func:`secondary_inference_sensitive`).
    """
    if set(members) - set(SECONDARY):
        raise ValueError(f"unknown secondary members {sorted(set(members) - set(SECONDARY))}")
    names = list(SECONDARY)
    m = len(names)
    est = {n: members.get(n) for n in names}
    po = dict(p_override or {})
    p2 = np.array([1.0 if est[n] is None else float(po.get(n, est[n].p)) for n in names])
    pt = np.array([1.0 if est[n] is None else tost_p(est[n], sesoi[n]) for n in names])
    adj = bh_adjust(p2)
    adj_t = bh_adjust(pt)
    selected = adj < q
    R = int(selected.sum())
    fcr_level = 1 - q * R / m if R else None
    out: dict = {"m": m, "R_selected": R, "fcr_level": fcr_level, "members": {},
                 "not_estimable": [n for n in names if est[n] is None]}
    for i, n in enumerate(names):
        e = est[n]
        s = sesoi[n]
        if e is None:
            out["members"][n] = {"estimate": None, "se": None, "df": None, "p": 1.0, "p_bh": float(adj[i]),
                                 "tost_p": 1.0, "tost_p_bh": float(adj_t[i]), "sesoi": s, "positive": False,
                                 "opposite": False, "practically_null": False, "category": "not_estimable",
                                 "qualifiers": ["not estimable: entered BH with p = 1 and TOST p = 1 (m = 6)"]}
            continue
        pos = bool(selected[i] and e.estimate > 0)
        opp = bool(selected[i] and e.estimate < 0)
        null = bool(adj_t[i] < q)
        rec = {"estimate": e.estimate, "se": e.se, "df": e.df, "p": float(p2[i]), "p_bh": float(adj[i]),
               "tost_p": float(pt[i]), "tost_p_bh": float(adj_t[i]), "sesoi": s, "positive": pos,
               "opposite": opp, "practically_null": null, "qualifiers": []}
        if n in po:
            rec["p_model"] = float(e.p)
            rec["p_substituted"] = True
        if selected[i]:
            fci = e.ci(fcr_level)
            rec["fcr_interval"] = list(fci)
            rec["at_least_sesoi"] = bool(pos and fci[0] > s)
            rec["beyond_sesoi_opposite"] = bool(opp and fci[1] < -s)
        if n == "S6" and pos:
            ok = a3nl_s6 is not None and a3nl_s6.estimate > 0 and a3nl_s6.p < ALPHA
            rec["s6_conjunction"] = {"a3nl_estimate": None if a3nl_s6 is None else a3nl_s6.estimate,
                                     "a3nl_p": None if a3nl_s6 is None else a3nl_s6.p, "met": bool(ok)}
            if not ok:
                pos = False
                rec["positive"] = False
                rec["qualifiers"].append("S6 co-drift conjunction rule not met (A3-NL)")
        cat = ("positive" if pos else "opposite" if opp else "practically_null" if null else "inconclusive")
        if pos and null:
            rec["reading"] = "detectable, but below the SESOI"
        if null:
            rec["true_score_bound"] = true_score_bound(s, rho_bar) if n != "S4" else None
        if n == "S5":
            rec["qualifiers"].append("ambiguous by design: V > A is predicted by sensory gating and equally by eye "
                                     "closure or gaze loss; the ocular diagnostic is beta_A (H1-A)")
        rec["category"] = cat
        out["members"][n] = rec
    return out


def secondary_inference_sensitive(members: Mapping[str, Estimate | None], sesoi: Mapping[str, float], member: str, *,
                                  estimate: Estimate | None = None, p: float | None = None,
                                  a3nl_s6: Estimate | None = None, q: float = ALPHA) -> dict:
    """"Inference-sensitive to X" for one member of the family of six.

    X's estimate (with its SE / CI: A1, A3) or X's p-value alone (A2) is substituted for ``member``; BH (and, with an
    estimate, TOST-BH and the FCR interval) is recomputed with the other five members unchanged (m = 6); the label
    applies if the member's category changes."""
    if (estimate is None) == (p is None):
        raise ValueError("give exactly one of estimate or p")
    base = secondary_family(members, sesoi, a3nl_s6=a3nl_s6, q=q)
    if estimate is not None:
        alt = secondary_family({**members, member: estimate}, sesoi, a3nl_s6=a3nl_s6, q=q)
    else:
        alt = secondary_family(members, sesoi, a3nl_s6=a3nl_s6, q=q, p_override={member: float(p)})
    b, a = base["members"][member]["category"], alt["members"][member]["category"]
    return {"member": member, "category": b, "category_under_x": a, "changed": a != b,
            "p_bh_under_x": alt["members"][member]["p_bh"]}


def _category_with(e: Estimate | None, p: float | None, base: Mapping, sesoi: float) -> str:
    if e is not None:
        return primary_category(e, sesoi)["category"]
    pos = p < ALPHA and base["estimate"] > 0
    opp = p < ALPHA and base["estimate"] < 0
    if pos:
        return "positive"
    if opp:
        return "opposite"
    return "practically_null" if base["practically_null"] else "inconclusive"


def robustness_qualifiers(primary: Estimate, sesoi: float, *, sensitivity: Mapping[str, Estimate] | None = None,
                          inference: Mapping[str, Mapping] | None = None, test: str | None = None,
                          sensitivity_ci: Mapping[str, Sequence[float]] | None = None) -> dict:
    """Robustness labels attached to the category of an estimate (never replacing it).

    ``sensitivity``: X -> estimate of the re-run (X in :data:`ROBUSTNESS_X`). ``inference``: A1 / A2 / A3 ->
    ``{"estimate": Estimate}`` (A1, A3: CI-bearing) or ``{"p": float}`` (A2: surrogate p only). ``sesoi`` is the SESOI
    of the test qualified (the threshold is half of it). With ``test`` (P1, H1A, S2, S3, S4), an X that does not
    qualify that test (:data:`QUALIFIER_APPLICABILITY`) raises. ``sensitivity_ci``: X -> the interval to use instead
    of the Wald 95% CI; required for S-CENS (its 95% equal-tailed posterior interval). For a BH-adjusted member of
    the family of six use :func:`secondary_inference_sensitive` for the inference-only analyses.
    """
    base = primary_category(primary, sesoi)
    labels: list[str] = []
    detail: dict = {}
    sci = dict(sensitivity_ci or {})
    for x, ex in (sensitivity or {}).items():
        if x == "S-OCU" or x.startswith("S-OCU"):
            raise ValueError("S-OCU is descriptive and never a robustness qualifier")
        if x not in ROBUSTNESS_X:
            raise ValueError(f"{x} is not in the robustness list")
        if test is not None and not qualifier_applies(x, test):
            raise ValueError(f"{x} does not apply to {test}")
        if x == "S-CENS" and x not in sci:
            raise ValueError("S-CENS needs its 95% equal-tailed posterior interval (sensitivity_ci)")
        diff = abs(ex.estimate - primary.estimate)
        lo, hi = tuple(sci[x]) if x in sci else ex.ci95
        sens = diff >= sesoi / 2 or not (lo <= primary.estimate <= hi)
        detail[x] = {"diff": diff, "half_sesoi": sesoi / 2, "ci95_excludes_primary": not (lo <= primary.estimate <= hi),
                     "sensitive": sens}
        if sens:
            labels.append(f"sensitive to {x}")
    for x, spec in (inference or {}).items():
        if x not in INFERENCE_X:
            raise ValueError(f"{x} is not an inference-only analysis (A1, A2, A3)")
        e = spec.get("estimate")
        p = spec.get("p")
        cat = _category_with(e, p if e is None else None, base, sesoi)
        changed = cat != base["category"]
        detail[x] = {"category_under_x": cat, "changed": changed}
        if changed:
            labels.append(f"inference-sensitive to {x}")
        if x == "A2" and base["category"] == "positive":
            pa = p if p is not None else (e.p if e is not None else None)
            if pa is not None and pa >= ALPHA:
                labels.append("not robust to autocorrelation (A2)")
    return {"category": base["category"], "labels": labels, "detail": detail}


def h2_category(diff: Estimate, sesoi: float, *, pooled_beta: float | None = None, s_rel: Estimate | None = None,
                s_blink_h2: float | None = None, s_ica_h2: float | None = None) -> dict:
    """Group-difference categories: Delta = beta_ASD - beta_TD.

    "CI excludes 0" uses the 95% CI; "CI inside +-SESOI" the 90% CI (TOST; labelled practically unreachable).
    With ``pooled_beta``, a CI-excluding-0 difference is read as *steeper* (same sign as the pooled slope) or
    *flatter* ASD coupling; a flatter slope is called *decoupling* only if the S-REL difference keeps its sign with a
    95% CI excluding 0 and the S-BLINK-H2 and S-ICA-H2 differences keep its sign (S-OCU-H2 does not enter).
    """
    lo, hi = diff.ci95
    excl = lo > 0 or hi < 0
    inside = _inside(diff.ci90, sesoi)
    cat = "ci_excludes_0" if excl else ("ci_inside_sesoi" if inside else "inconclusive")
    out = {"category": cat, "estimate": diff.estimate, "se": diff.se, "ci95": [lo, hi], "ci90": list(diff.ci90),
           "sesoi": sesoi}
    if cat == "ci_inside_sesoi":
        out["label"] = "CI inside +-SESOI (practically unreachable: TOST power <= 0.41 AVSRT, 0.00 FAST / IC)"
    if excl and pooled_beta is not None and pooled_beta != 0:
        same = math.copysign(1, diff.estimate) == math.copysign(1, pooled_beta)
        out["direction"] = "steeper_asd" if same else "flatter_asd"
        if not same:
            sgn = math.copysign(1, diff.estimate)
            conds = {
                "s_rel_keeps_sign_ci_excludes_0": bool(s_rel is not None and math.copysign(1, s_rel.estimate) == sgn
                                                       and (s_rel.ci95[0] > 0 or s_rel.ci95[1] < 0)),
                "s_blink_h2_keeps_sign": bool(s_blink_h2 is not None and math.copysign(1, s_blink_h2) == sgn),
                "s_ica_h2_keeps_sign": bool(s_ica_h2 is not None and math.copysign(1, s_ica_h2) == sgn),
            }
            out["decoupling_conditions"] = conds
            out["decoupling"] = all(conds.values())
    return out


def cross_task(c1: Estimate, c2: Estimate, *, p1_category: str, s4_practically_null: bool, s2_positive: bool,
               s3_positive: bool, margin: float = 0.03, alpha: float = ALPHA, power_c003: float | None = None) -> dict:
    """Holm over C1 = beta_A - (beta_F + beta_I) / 2 and C2 = beta_F - beta_I; equivalence +-``margin`` (90% CIs);
    the first matching label L1-L6. ``s4_practically_null``, ``s2_positive``, ``s3_positive`` are the categories of
    the family of six (BH-adjusted)."""
    ph = holm_adjust([c1.p, c2.p])
    sig1, sig2 = ph[0] < alpha, ph[1] < alpha
    eq1, eq2 = _inside(c1.ci90, margin), _inside(c2.ci90, margin)
    if sig1 and c1.estimate > 0 and p1_category == "positive" and s4_practically_null and not sig2 \
            and not (s2_positive or s3_positive):
        lab = ("L1", "AVSRT-specific")
    elif sig1 and c1.estimate > 0:
        lab = ("L2", "Stronger in AVSRT than in FAST/IC")
    elif sig1 and c1.estimate < 0:
        lab = ("L3", "Weaker in AVSRT than in FAST/IC")
    elif sig2:
        lab = ("L4", "FAST and IC differ")
    elif eq1 and eq2:
        lab = ("L5", "Similar across tasks")
    else:
        lab = ("L6", "Task differences not resolved")
    out = {"label_id": lab[0], "label": lab[1], "C1": {"estimate": c1.estimate, "se": c1.se, "p": c1.p,
                                                       "p_holm": float(ph[0]), "ci95": list(c1.ci95),
                                                       "ci90": list(c1.ci90), "equivalent": eq1},
           "C2": {"estimate": c2.estimate, "se": c2.se, "p": c2.p, "p_holm": float(ph[1]), "ci95": list(c2.ci95),
                  "ci90": list(c2.ci90), "equivalent": eq2}, "margin": margin,
           "note": "task is confounded with time in session and a_wb is task-specific; the label "
                   "'consistent across tasks' is never used"}
    if lab[0] == "L6" and power_c003 is not None:
        out["power_contrast_0.03"] = power_c003
    return out
