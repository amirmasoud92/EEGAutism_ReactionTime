"""Implementation checks of the FAST group model (Additional file 1, Supplementary Methods S5, Table S9).

- ``second_impl``: second implementation (statsmodels MixedLM, several starting values) of the FAST group model
  (engine label ``task_M3_H2``), on copies of its stored engine input, specification and result; term-by-term
  comparison with lme4.
- ``marginality``: refits of three follow-up models with the lower-order product they lacked: temporal expectation
  (iii) + nnt_c x grp; dimensional measures + measure x grp (CPT-3 response style, MABC-2); X2 + t_run x grp.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import second_impl as SI
from modeling import sensitivity as SENS
from statistical_analysis.follow_up_analyses import H2T, _participants_measure, add_cols, by_group, fit_targets, nnt_numeric


def second_impl(r, fit_dir) -> dict:
    """``fit_dir``: the engine directory holding ``task_M3_H2_input.tsv``, ``_spec.json`` and ``_result.json``."""
    src = Path(fit_dir)
    dst = r.work / "second_impl"
    dst.mkdir(parents=True, exist_ok=True)
    for suf in ("_input.tsv", "_spec.json", "_result.json"):
        shutil.copy2(src / f"task_M3_H2{suf}", dst / f"task_M3_H2{suf}")
    res = json.loads((dst / "task_M3_H2_result.json").read_text(encoding="utf-8"))
    spec = json.loads((dst / "task_M3_H2_spec.json").read_text(encoding="utf-8"))
    d = pd.read_csv(dst / "task_M3_H2_input.tsv", sep="\t")
    t0 = time.time()
    sm = SI.statsmodels_fit(d, spec["fixed_formula"], res["final_state"])
    cmp = SI.compare_with_lme4(sm, res)
    rows = {row["term"]: row for row in cmp["rows"]}
    out = {"label": "task_M3_H2", "pass": bool(cmp["pass"]), "max_fe_diff_rel_se": cmp["max_fe_diff_rel_se"],
           "loglik_diff_statsmodels_minus_lme4": cmp["loglik_diff"], "se_ratio_range": cmp["se_ratio_range"],
           "terms_compared": cmp["terms_compared"], "terms_missing": cmp["terms_missing"],
           "n_ok_starts": sm["n_ok"], "tolerance": cmp["tolerance"], "seconds": round(time.time() - t0, 1),
           "a_wb:grp": rows.get("a_wb:grp"), "a_wb": rows.get("a_wb"),
           "definition": "second implementation on the stored M3-H2 fit (statsmodels MixedLM, multiple starts) "
                         "compared term by term with lme4"}
    r.write("second_implementation", out)
    return out


def marginality(r, participants_tsv) -> dict:
    """Follow-up models refitted with every lower-order product."""
    st = r.state("S2")
    md = r.provider.model("M3-H2")
    f = md.frame
    g, a = f["grp"].to_numpy(float), f["a_wb"].to_numpy(float)
    out: dict = {}
    # temporal expectation (iii) with nnt_c x grp
    nn = nnt_numeric(f)
    nc = nn - float(np.mean(nn))
    m = add_cols(md, {"nnt_x_grp": nc * g, "awb_x_nnt": a * nc, "awb_x_nnt_x_grp": a * nc * g})
    out["R5iii_full"] = fit_targets(r, m, "marginality_sequence", {**by_group("awb_x_nnt", "awb_x_nnt_x_grp"), **H2T()},
                                    state=st)
    # X2 with t_run x grp
    t = f["t_run"].to_numpy(float)
    m = add_cols(md, {"trun_x_grp": t * g, "awb_x_trun_x_grp": a * t * g})
    out["X2_full"] = fit_targets(r, m, "marginality_time", {
        "a_wb:t_run|TD": {"a_wb:t_run": 1.0, "awb_x_trun_x_grp": -0.5},
        "a_wb:t_run|ASD": {"a_wb:t_run": 1.0, "awb_x_trun_x_grp": 0.5},
        "a_wb:t_run|ASD-TD": {"awb_x_trun_x_grp": 1.0}, **H2T()}, state=st)
    # dimensional measures with measure x grp
    for col in ("mabc_total_ss", "cpt_response_style"):
        vals = _participants_measure(participants_tsv, col)
        v = f["pid"].map(vals).astype(float)
        ok = np.isfinite(v.to_numpy(float))
        m0 = SENS.refinalise(SENS._replace(md, frame=f.loc[ok].reset_index(drop=True)), reason=f"measure {col}")
        mf = m0.frame
        kids = mf.groupby("pid")["pid"].first().index
        kv = np.array([vals[k] for k in kids])
        x = (mf["pid"].map(vals).astype(float).to_numpy() - kv.mean()) / kv.std(ddof=1)
        aa, gg = mf["a_wb"].to_numpy(float), mf["grp"].to_numpy(float)
        m = add_cols(m0, {"meas_z": x, "meas_x_grp": x * gg, "awb_x_meas": aa * x, "awb_x_meas_x_grp": aa * x * gg})
        out[f"R9_full_{col}"] = fit_targets(r, m, f"marginality_dimensional_{col}", {
            **by_group("awb_x_meas", "awb_x_meas_x_grp"), "awb_x_meas|pooled": {"awb_x_meas": 1.0}, **H2T()},
            state=st)
    out["definition"] = ("temporal expectation (iii), time on task by group and the dimensional-measure models "
                         "refitted with the lower-order product they lacked")
    r.write("lower_order_products", out)
    return out
