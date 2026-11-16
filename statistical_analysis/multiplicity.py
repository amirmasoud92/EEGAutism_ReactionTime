"""Holm correction over the 18 group-difference tests of the follow-up analyses (Additional file 1, Table S5).

The family: face vs object blocks and first vs second half (split x group x alpha), siblings minus TD and minus
autistic children, age x coupling, the three temporal-expectation differences, the alpha effect at five RT quantiles
and on misses, and the four evoked-response paths (alpha -> P1 / N1 window, P1 / N1 window -> RT; -100 to 0 ms
baseline). Each argument is the result dict of the corresponding function of :mod:`statistical_analysis.follow_up_analyses` or
:mod:`statistical_analysis.evoked_response_pathway`.
"""

from __future__ import annotations


def holm(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r.get("p") is not None]
    order = sorted(range(len(ok)), key=lambda i: ok[i]["p"])
    m, run = len(ok), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ok[i]["p"]))
        ok[i]["p_holm"] = run
    return ok


def tgt(block: dict | None, key: str) -> dict | None:
    return ((block or {}).get("targets") or {}).get(key)


def family(splits: dict | None, siblings: dict | None, age: dict | None, temporal_expectation: dict | None,
           rt_quantiles: dict | None, evoked_response: dict | None) -> list[dict]:
    """Rows of the follow-up family with their Holm-adjusted p (``p_holm``)."""
    rows = []

    def add(name, block, key, src):
        t = tgt(block, key)
        if t:
            rows.append({"test": name, "source": src, "estimate": t["estimate"], "ci95": t["ci95"], "p": t["p"],
                         "cr2_p": t.get("cr2_p")})

    r2, r3, r5, r6, r7 = splits, siblings, temporal_expectation, rt_quantiles, evoked_response
    if r2:
        add("R2 FAST group difference: face vs object blocks", r2.get("category"), "H2_difference_between_splits", "R2")
        add("R2 FAST group difference: first vs second half", r2.get("session_half"),
            "H2_difference_between_splits", "R2")
    if r3:
        add("R3 FAST SIB - TD", r3.get("FAST"), "SIB-TD", "R3")
        add("R3 FAST SIB - ASD", r3.get("FAST"), "SIB-ASD", "R3")
    r4 = age
    if r4:
        add("R4 FAST age x coupling, ASD - TD", r4.get("FAST"), "age_slope_of_coupling|ASD-TD", "R4")
    if r5:
        add("R5(i) FAST RT hazard, ASD - TD", r5.get("i_rt_hazard"), "nnt_c|ASD-TD", "R5")
        add("R5(ii) FAST alpha ~ hazard, ASD - TD", r5.get("ii_alpha_x_hazard"), "nnt_c|ASD-TD", "R5")
        add("R5(iii) FAST coupling x hazard, ASD - TD", r5.get("iii_coupling_x_hazard"), "awb_x_nnt|ASD-TD", "R5")
    if r6:
        for q, v in (r6.get("quantiles") or {}).items():
            rows.append({"test": f"R6 FAST alpha effect at RT {q}, ASD - TD", "source": "R6",
                         "estimate": v["diff"], "ci95": v["ci95"], "p": v["welch_p"]})
        add("R6 FAST alpha -> miss, ASD - TD", r6.get("miss_glmm"), "H2", "R6")
    if r7:
        for comp in ("p1", "n1"):
            add(f"R7 FAST alpha -> {comp.upper()}, ASD - TD", r7.get(f"a_alpha_to_{comp}_z"), "H2", "R7")
            add(f"R7 FAST {comp.upper()} -> RT, ASD - TD", r7.get(f"b_{comp}_to_rt"), f"{comp}_z|ASD-TD", "R7")
    return holm(rows)
