"""Analysis data frame of one model x task x cohort x analysis set.

Inputs
------
- the cohort's combined predictor table ``features/<TASK>/<TASK>_<cohort>_features_predictor.parquet`` (alpha
  decomposition ``a_wb`` / ``a_slow`` / ``a_hist5``, window validity, ocular z, block labels);
- the trial table ``events/<TASK>_trials.parquet`` (behaviour and timing: analysis-set flags, previous outcome, time
  since the last press, SOA, block positions), reduced to a behaviour table;
- ``configs/model_knots.yaml`` (knots and centring medians), participants (age, sex, group), the session table (task
  start time).

The two tables are joined one-to-one on ``subject``, ``task``, ``run`` and ``sample``. Cohorts: ``dev`` = unaffected
siblings (used to develop the processing); ``conf`` = autistic and typically developing children (``grp`` is
attached only for ``conf``).

Rows: the trial-table analysis subset of ``configs/events.yaml`` (``primary_rt`` for the RT models,
``predictor_set`` for M2), intersected with the predictor table's set flag and a valid alpha window with a finite
``a_wb`` / ``a_slow`` (and ``a_hist5`` in M1-hist). The predictor-table flag must equal the trial-table predictor set
on every joined row (a mismatch raises).

Covariates (definitions in :mod:`modeling.knots`):

- ``prev_miss`` +0.5 / -0.5 / 0 from ``prev_outcome`` (miss / hit / none);
- ``log_tsp`` = ln(min(tsp, 10 s)) on the concatenated clock, plus ``tsp_join`` (``prev_press_crosses_join``);
  ``soa`` / ``soa_prev`` with ``soa_join``; IC ``log_tsfl`` with ``tsfl_join`` (``prev_target_crosses_join``) and
  ``t_since_prev_contour`` with ``contour_join``. A join indicator enters only when non-constant;
- ``t_run`` / ``tib``: z within child x task over the final model rows; they are mean 0 within child and are not
  re-centred, so ``beta(a_wb)`` is the slope at mean ``t_run``;
- splines: ``ns(x, knots, Boundary.knots)`` with the knot table (columns ``ns_<x>_<df>_<k>``,
  :func:`modeling.splines.ns_basis`);
- linear continuous covariates centred at the pooled medians of the knot table (``soa_prev``, ``task_start_h``);
  ``age_c`` at the pooled mean age of all children of the task; ``sex_c`` male +0.5;
- AVSRT ``cVA`` / ``cAV`` and ``prev_cond`` (A / V / AV, sum contrasts -> ``pcA``, ``pcV``); FAST ``cat`` (face +0.5),
  ``nnt`` (7 levels of ``nnt_level``, sum contrasts -> ``nnt1``..``nnt6``), ``half_status`` +-0.5; IC
  ``prev_contour_type`` (illusory +0.5), ``seq180_c`` = ``seq180_index`` - 2;
- ocular ``heog_z`` / ``vfp_z`` with their missingness indicators (:func:`modeling.model_inputs.ocular_terms`);
- the block of the block intercept is the decomposition block (``pid_run_block``);
- group models: ``grp`` (TD -0.5, ASD +0.5) and the child alpha reliability moderator ``relq_c``.

Child exclusions: missing age or sex; FAST / IC behavioural non-performance; fewer than 30 model rows after every row
rule. Rows with a missing non-ocular covariate are dropped (complete case, counted) before the 30-row rule.

Outcome modes: ``"none"`` (design only; ``rt_ms`` is not read) and ``"real"`` (``lrt`` = ln(RT in s) or ``miss``).
The child identifier column ``pid`` holds the BIDS participant ID.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modeling import knots as K
from modeling import model_inputs
from modeling.model_specs import S_TASK, ModelDef, RandomDef, get_model
from modeling.splines import ns_basis, ns_column_names
from utils.config import ProjectConfig

__all__ = ["ModelData", "build_model_data", "combine_m5h", "write_model_input", "load_knot_table",
           "load_group_labels", "PREV_COND_LEVELS", "NNT_LEVELS", "MIN_ROWS", "predictor_variant", "GROUP_CODES"]

MIN_ROWS = 30
PREV_COND_LEVELS = ("A", "V", "AV")                 # sum contrasts: A (1, 0), V (0, 1), AV (-1, -1)
NNT_LEVELS = ("<=2", "3", "4", "5", "6", "7", ">=8")
_COHORTS = {"dev": "DEV", "conf": "CONF"}
#: Group coding of the group models.
GROUP_CODES = {"TD": -0.5, "ASD": 0.5}
#: Normalisation of the participants.tsv group labels.
GROUP_NORMALISATION = {"ASD SIBLING": "SIB", "NAN": "n/a"}


@dataclass
class ModelData:
    """Model rows and metadata of one model fit (see the module docstring)."""

    model_def: ModelDef
    task: str
    cohort: str
    analysis_set: str
    variant: str
    frame: pd.DataFrame
    fixed_terms: list[str]
    splines: list[dict]
    outcome_column: str
    outcome_kind: str
    has_grp: bool = False
    intercept: bool = True
    log: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n_children(self) -> int:
        return int(self.frame["pid"].nunique())

    def columns_needed(self) -> list[str]:
        """Columns of the model-input TSV: ids, outcome, every fixed-term column, random terms, spline sources."""
        cols: list[str] = ["pid", "run", "sample", "trial_index_in_run", "pid_run_block", "block"]
        if self.outcome_column in self.frame.columns:
            cols.append(self.outcome_column)
        for t in self.fixed_terms:
            cols.extend(t.split(":"))
        rnd = self.model_def.random
        cols.extend(t for t in (*rnd.corr, *rnd.uncorrelated) if t != "1")
        for s in self.splines:
            cols.append(s["variable"])
            cols.extend(s["columns"])
        for extra in ("task", "a_wb", "a_slow"):
            if extra in self.frame.columns:
                cols.append(extra)
        return [c for c in dict.fromkeys(cols) if c in self.frame.columns]


# --------------------------------------------------------------------------- configuration helpers
def load_knot_table(cfg: ProjectConfig) -> dict:
    with open(Path(cfg.root) / "configs" / "model_knots.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _spline_spec(knot_table: Mapping, knots_key: str, variable: str, df: int) -> dict:
    term = knot_table["models"][knots_key]["terms"].get(f"ns({variable}, {df})")
    if term is None:
        raise KeyError(f"configs/model_knots.yaml has no ns({variable}, {df}) for {knots_key}")
    return {"variable": variable, "df": int(df), "interior_knots": [float(k) for k in term["interior_knots"]],
            "boundary_knots": [float(k) for k in term["boundary_knots"]],
            "columns": ns_column_names(variable, df), "knots_key": knots_key}


_WINDOWS = {"s_w500", "s_w1000", "s_w400"}
_VALID_OF = {"s_ica": "valid_s_ica", "s_roi": "valid_s_roi", "e7_mu": "valid_e7_mu"}


def predictor_variant(variant: str) -> dict:
    """Column names of a predictor variant: set flag, validity, decomposition and ocular columns. ``""`` =
    primary."""
    v = variant or ""
    suf = f"__{v}" if v else ""
    if v in _WINDOWS:
        return {"set_flag": f"in_{v}_predictor_set", "valid": f"valid_{v}", "dec_suffix": suf, "ocular_suffix": suf,
                "tt_predictor_set": f"{v}_predictor_set"}
    return {"set_flag": "in_predictor_set", "valid": _VALID_OF.get(v, "valid"), "dec_suffix": suf,
            "ocular_suffix": "", "tt_predictor_set": "predictor_set"}


def _subset_mask(trials: pd.DataFrame, tc, name: str) -> pd.Series:
    from trials.schema import apply_subset

    if name not in tc.analysis_subsets:
        raise KeyError(f"analysis set {name!r} is not defined for {tc.name if hasattr(tc, 'name') else 'task'}")
    return apply_subset(trials, tc.analysis_subsets[name])


def _subset_flag_columns(tc, names: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for n in names:
        sub = tc.analysis_subsets[n]
        cols.extend([*(sub.get("require_true") or []), *(sub.get("require_false") or [])])
    return list(dict.fromkeys(cols))


_TT_TASK_COLUMNS = {
    "AVSRT": ("prev_modality",),
    "FAST": ("nnt_level", "block_status"),
    "IC": ("prev_target_crosses_join", "prev_contour_crosses_join", "prev_contour_type"),
}
#: Trial-table columns that the predictor table also carries (taken from the predictor side after the join).
_SHARED = ("onset_s", "trial_key", "trial_index_in_run", "value", "condition", "is_target", "block_index_in_run",
           "n_joins_before", "modality", "block_status", "half_status", "category", "seq180_index")


def _behaviour_table(trials: pd.DataFrame, task: str, tc, analysis_set: str, tt_pred_set: str,
                     subjects: set[str], real: bool, extra_row_flags: Mapping | None = None,
                     extra_trial_columns: Sequence[str] = (), real_outcome_columns: Sequence[str] = (),
                     rt_column: str = "rt_ms") -> pd.DataFrame:
    """Behaviour table: keys, analysis-set membership, behaviour / timing covariates (no predictor column).

    ``extra_row_flags`` (``{"require_true": [...], "require_false": [...]}``, boolean trial-table columns; a missing
    value never satisfies a requirement) becomes the membership column ``tt_extra_rows``; only the boolean membership
    is kept, as for the analysis sets. ``extra_trial_columns`` are copied as they are (timing / structure columns);
    ``real_outcome_columns`` are read only with ``real`` (as ``real_<c>``).
    """
    from trials.schema import apply_subset

    tr = trials[trials["subject"].isin(subjects)].reset_index(drop=True)
    in_set = _subset_mask(tr, tc, analysis_set).to_numpy(bool)
    in_pset = _subset_mask(tr, tc, tt_pred_set).to_numpy(bool)
    cov = K.covariates(tr, task, np.ones(len(tr), bool))           # raw values; z-scores recomputed later
    out = pd.DataFrame({"subject": tr["subject"].to_numpy(), "task": tr["task"].to_numpy(),
                        "run": tr["run"].to_numpy(), "sample": tr["sample"].to_numpy()})
    out[f"tt_in_{analysis_set}"] = in_set
    out["tt_in_predictor_set"] = in_pset
    out["log_tsp"] = cov["log_tsp"].to_numpy(float)
    out["soa"] = cov["soa"].to_numpy(float)
    out["t_run_raw"] = cov["t_run_raw"].to_numpy(float)
    out["tib_raw"] = cov["tib_raw"].to_numpy(float)
    if task == "IC":
        out["log_tsfl"] = cov["log_tsfl"].to_numpy(float)
        out["t_since_prev_contour"] = cov["t_since_prev_contour"].to_numpy(float)
    out["tsp_join"] = _bool01(tr["prev_press_crosses_join"])
    out["soa_join"] = _bool01(tr["soa_prev_crosses_join"])
    out["prev_outcome"] = tr["prev_outcome"].astype("string").to_numpy()
    for c in _TT_TASK_COLUMNS.get(task, ()):
        if c == "block_status":
            continue
        if c.endswith("_crosses_join"):
            out[{"prev_target_crosses_join": "tsfl_join", "prev_contour_crosses_join": "contour_join"}[c]] = \
                _bool01(tr[c])
        else:
            out[c] = tr[c].astype("string").to_numpy()
    if extra_row_flags:
        out["tt_extra_rows"] = apply_subset(tr, extra_row_flags).to_numpy(bool)
    for c in extra_trial_columns:
        out[c] = tr[c].to_numpy()
    if real:
        out["rt_ms"] = pd.to_numeric(tr[rt_column], errors="coerce").to_numpy(float)
        out["responded"] = tr["responded"].fillna(False).astype(bool).to_numpy()
        for c in real_outcome_columns:
            out[f"real_{c}"] = tr[c].to_numpy()
    elif real_outcome_columns:
        raise ValueError("real-outcome columns are read only with outcome='real'")
    return out


def _bool01(s: pd.Series) -> np.ndarray:
    return pd.array(s, dtype="boolean").fillna(False).to_numpy(bool).astype(float)


def _task_start(cfg: ProjectConfig, task: str, subjects: Sequence[str]) -> pd.Series:
    """Child-level ``task_start_h``: ``time_in_session_h`` of the first run of the task on the main day; when the
    task has no main-day run, the first run on its own day (h from that day's first recording)."""
    sess = pd.read_parquet(cfg.paths.derivatives / "session" / "session_table.parquet",
                           columns=["subject", "task", "run", "is_main_day", "day_index", "time_in_session_h"])
    s = sess[(sess["task"] == task) & sess["subject"].isin(list(subjects))].copy()
    s["main"] = s["is_main_day"].astype(bool)
    s = s.sort_values(["subject", "main", "day_index", "run"], ascending=[True, False, True, True])
    first = s.groupby("subject")[["time_in_session_h", "main"]].first()
    return first["time_in_session_h"].astype(float)


def _pooled_mean_age(cfg: ProjectConfig, trial_subjects: Sequence[str], participants: pd.DataFrame) -> float:
    p = participants[participants["participant_id"].isin(list(trial_subjects))]
    a = pd.to_numeric(p["age"], errors="coerce")
    return float(a.mean())


def _sum_contrasts(values: pd.Series, levels: Sequence[str], prefix: str, names: Sequence[str] | None = None
                   ) -> pd.DataFrame:
    """Sum-contrast columns (R ``contr.sum``): level k < K -> e_k, last level -> -1; NA for unknown values."""
    v = values.astype("string")
    k = len(levels)
    cols = names or [f"{prefix}{i}" for i in range(1, k)]
    out = pd.DataFrame(np.nan, index=values.index, columns=cols)
    for i, lev in enumerate(levels):
        m = (v == lev).fillna(False).to_numpy(bool)
        row = np.zeros(k - 1)
        if i < k - 1:
            row[i] = 1.0
        else:
            row[:] = -1.0
        out.loc[m, :] = row
    return out


# --------------------------------------------------------------------------- main builder
def build_model_data(model: str, cohort: str, *, analysis_set: str | None = None, variant: str = "",
                     outcome: str = "none", cfg: ProjectConfig | None = None, pseudonymizer=None,
                     predictor_path: str | os.PathLike[str] | None = None,
                     trials_path: str | os.PathLike[str] | None = None, extra_predictor_columns: Sequence[str] = (),
                     extra_row_flags: Mapping | None = None, extra_trial_columns: Sequence[str] = (),
                     real_outcome_columns: Sequence[str] = (), min_rows: int = MIN_ROWS,
                     rt_column: str = "rt_ms", allow_missing_rt: bool = False,
                     apply_nonperformance: bool = True) -> ModelData:
    """Build the analysis data of ``model`` (e.g. ``"M1"``) for ``cohort`` (``"dev"`` / ``"conf"``).

    Parameters
    ----------
    analysis_set
        ``configs/events.yaml`` subset of the model rows (default: the model's, e.g. ``primary_rt``).
    variant
        Predictor variant (``""`` primary; ``"s_w500"`` ... for the sensitivity windows).
    outcome
        ``"none"`` (default; the RT / miss values are not read) or ``"real"``.
    predictor_path, trials_path
        Paths of the predictor and trial tables (default: the configured derivative paths).
    extra_predictor_columns, extra_row_flags, extra_trial_columns, real_outcome_columns, min_rows, rt_column
        Options of the sensitivity analyses (:mod:`modeling.sensitivity`): extra predictor columns kept on the rows
        (e.g. ``blink_in_window``); an extra boolean row requirement on trial-table flags (e.g.
        ``responded_sens_100_1000`` for S-RT); extra timing / structure columns of the trial table (e.g.
        ``t_to_bound_ms``); response-dependent columns read only with ``outcome="real"`` (stored as
        ``real_<name>``); the minimum rows per child (S-MIN, S-ALT); the RT column of the outcome (S-RT windows);
        ``allow_missing_rt`` (S-CENS) leaves ``lrt`` NaN on the rows without an RT (misses, censored downstream)
        instead of refusing them.
    """
    if outcome not in ("none", "real"):
        raise ValueError("outcome must be 'none' or 'real'")
    if real_outcome_columns and outcome != "real":
        raise ValueError("real-outcome columns are read only with outcome='real'")
    if cfg is None:
        from utils.config import load_project_config

        cfg = load_project_config()
    from utils.config import load_events_config

    mdef = get_model(model)
    task = mdef.task
    coh = _COHORTS[str(cohort).strip().lower()]
    aset = analysis_set or mdef.default_set
    pv = predictor_variant(variant)
    ev = load_events_config(root=cfg.root)
    tc = ev.task(task)
    knot_table = load_knot_table(cfg)
    log: dict = {"model": mdef.name, "task": task, "cohort": coh, "analysis_set": aset, "variant": variant}

    # 1. predictor table
    ppath = Path(predictor_path) if predictor_path else (
        cfg.paths.derivatives / "features" / task / f"{task}_{coh.lower()}_features_predictor.parquet")
    suf, osuf = pv["dec_suffix"], pv["ocular_suffix"]
    dec_cols = [f"a_wb{suf}", f"a_slow{suf}", f"a_wb_raw{suf}"]
    hist_cols = ["a_hist5"] if not variant else []
    oc_cols = [f"heog_z{osuf}", f"vfp_z{osuf}", f"heog_missing{osuf}", f"vfp_missing{osuf}"]
    block_col = f"block{suf}"
    want = ["subject", "task", "run", "sample", "trial_index_in_run", pv["set_flag"], pv["valid"], "block",
            *dec_cols, *hist_cols, *oc_cols, *{"AVSRT": ["modality"], "FAST": ["category", "half_status"],
                                               "IC": ["seq180_index"]}[task]]
    want += [c for c in extra_predictor_columns if c not in want]
    import pyarrow.parquet as pq

    names = set(pq.read_schema(ppath).names)
    if block_col in names and block_col not in want:
        want.append(block_col)
    missing = [c for c in want if c not in names]
    if missing:
        raise KeyError(f"predictor table {ppath.name} lacks {missing}")
    pred = pd.read_parquet(ppath, columns=list(want))
    if block_col != "block" and block_col in pred.columns:
        pred["block"] = pred[block_col]
        pred = pred.drop(columns=[block_col])
    subjects = set(pred["subject"].astype(str))
    log["predictor_rows"] = int(len(pred))
    log["predictor_children"] = int(len(subjects))

    # 2. behaviour table from the trial table
    tpath = Path(trials_path) if trials_path else cfg.paths.derivatives / "events" / f"{task}_trials.parquet"
    tt_cols = sorted(set(K.TRIAL_COLUMNS[task]) | set(_subset_flag_columns(tc, [aset, pv["tt_predictor_set"]]))
                     | {"task", "prev_press_crosses_join", "soa_prev_crosses_join", "prev_outcome"}
                     | set(_TT_TASK_COLUMNS.get(task, ())) | ({"rt_ms", "responded"} if outcome == "real" else set())
                     | ({rt_column} if outcome == "real" else set())
                     | set((extra_row_flags or {}).get("require_true", []) or [])
                     | set((extra_row_flags or {}).get("require_false", []) or [])
                     | set(extra_trial_columns) | (set(real_outcome_columns) if outcome == "real" else set())
                     | ({"block_status", "half_status"} if task == "FAST" else set())
                     | ({"seq180_index"} if task == "IC" else set()))
    trials = pd.read_parquet(tpath, columns=tt_cols)
    trial_subjects_all = sorted(set(trials["subject"].astype(str)))
    beh = _behaviour_table(trials, task, tc, aset, pv["tt_predictor_set"], subjects, real=(outcome == "real"),
                           extra_row_flags=extra_row_flags, extra_trial_columns=extra_trial_columns,
                           real_outcome_columns=real_outcome_columns, rt_column=rt_column)

    # 3. the one-to-one join
    merged = pred.merge(beh, on=["subject", "task", "run", "sample"], how="left", validate="one_to_one")
    if len(merged) != len(pred):
        raise ValueError("predictor rows did not join one-to-one to the trial table")
    n_unjoined = int(merged["tt_in_predictor_set"].isna().sum())
    if n_unjoined:
        raise ValueError(f"{n_unjoined} predictor rows have no trial-table row")
    log["joined_once"] = True
    tt_pset = merged["tt_in_predictor_set"].astype(bool).to_numpy()
    p_pset = pd.array(merged[pv["set_flag"]], dtype="boolean").fillna(False).to_numpy(bool)
    if (tt_pset != p_pset).any():
        raise ValueError(f"predictor-table {pv['set_flag']} differs from the trial-table predictor set on "
                         f"{int((tt_pset != p_pset).sum())} rows")

    # 4. row rules
    valid = pd.array(merged[pv["valid"]], dtype="boolean").fillna(False).to_numpy(bool)
    in_set = merged[f"tt_in_{aset}"].astype(bool).to_numpy()
    a_wb = pd.to_numeric(merged[f"a_wb{suf}"], errors="coerce").to_numpy(float)
    a_slow = pd.to_numeric(merged[f"a_slow{suf}"], errors="coerce").to_numpy(float)
    rows = in_set & p_pset & valid & np.isfinite(a_wb) & np.isfinite(a_slow)
    if extra_row_flags:
        extra = merged["tt_extra_rows"].astype(bool).to_numpy()
        log["rows_extra_flags"] = {"flags": {k: list(v or []) for k, v in extra_row_flags.items()},
                                   "rows_failing": int((rows & ~extra).sum())}
        rows &= extra
    log["rows_in_set"] = int(in_set.sum())
    log["rows_after_window_rules"] = int(rows.sum())
    for req in mdef.requires:
        rows &= np.isfinite(pd.to_numeric(merged[req], errors="coerce").to_numpy(float))
        log[f"rows_with_finite_{req}"] = int(rows.sum())
    d = merged.loc[rows].copy().reset_index(drop=True)
    d["a_wb"] = pd.to_numeric(d[f"a_wb{suf}"], errors="coerce").astype(float)
    d["a_slow"] = pd.to_numeric(d[f"a_slow{suf}"], errors="coerce").astype(float)
    d["a_wb_raw"] = pd.to_numeric(d[f"a_wb_raw{suf}"], errors="coerce").astype(float)

    # 5. covariates
    cent = knot_table["centring_medians"][mdef.knots_key]
    d["prev_miss"] = d["prev_outcome"].map({"miss": 0.5, "hit": -0.5, "none": 0.0}).astype(float)
    if task == "AVSRT":
        mod = d["modality"].astype("string")
        d["cVA"] = mod.map({"A": -0.5, "V": 0.5, "AV": 0.0}).astype(float)
        d["cAV"] = mod.map({"AV": 2.0 / 3.0, "A": -1.0 / 3.0, "V": -1.0 / 3.0}).astype(float)
        pc = _sum_contrasts(d["prev_modality"], PREV_COND_LEVELS, "pc", names=["pcA", "pcV"])
        d[["pcA", "pcV"]] = pc.to_numpy()
    elif task == "FAST":
        d["cat"] = d["category"].astype("string").map({"face": 0.5, "object": -0.5}).astype(float)
        nn = _sum_contrasts(d["nnt_level"], NNT_LEVELS, "nnt")
        d[nn.columns] = nn.to_numpy()
        d["soa_prev_c"] = d["soa"] - float(cent["soa_prev"])
        d["half_status"] = pd.to_numeric(d["half_status"], errors="coerce").map({1: -0.5, 2: 0.5}).astype(float)
    else:
        d["prev_contour_type"] = d["prev_contour_type"].astype("string").map(
            {"Illusory_contours": 0.5, "Non_Illusory_contours": -0.5}).astype(float)
        d["seq180_c"] = pd.to_numeric(d["seq180_index"], errors="coerce").astype(float) - 2.0
    # child level
    from utils.participants import load_participants

    parts = load_participants(cfg, subjects=sorted(subjects | set(trial_subjects_all)))
    age_mean = _pooled_mean_age(cfg, trial_subjects_all, parts)
    pmap = parts.set_index("participant_id")
    d["age"] = d["subject"].map(pd.to_numeric(pmap["age"], errors="coerce")).astype(float)
    d["sex_c"] = d["subject"].map(pmap["sex"].astype("string").map({"m": 0.5, "f": -0.5})).astype(float)
    d["age_c"] = d["age"] - age_mean
    ts = _task_start(cfg, task, sorted(subjects))
    d["task_start_h"] = d["subject"].map(ts).astype(float)
    d["task_start_h_c"] = d["task_start_h"] - float(cent["task_start_h (child level)"])
    if osuf:
        for base in ("heog_z", "vfp_z", "heog_missing", "vfp_missing"):
            d[base] = d[f"{base}{osuf}"]
    for ind in ("heog_missing", "vfp_missing"):
        d[ind] = pd.array(d[ind], dtype="boolean").fillna(False).to_numpy(bool)

    # 6. complete cases on the model's non-ocular covariates
    need_raw = [v for v, _ in mdef.splines if v not in ("t_run", "tib")] + ["prev_miss", "t_run_raw", "tib_raw"]
    need_raw += {"AVSRT": ["cVA", "cAV"] + (["pcA", "pcV"] if "pcA" in " ".join(mdef.fixed) else []),
                 "FAST": ["cat", "nnt1", "soa_prev_c", "half_status"],
                 "IC": ["prev_contour_type", "seq180_c"]}[task]
    cc_missing = {c: int((~np.isfinite(d[c].to_numpy(float))).sum()) for c in need_raw}
    ok = np.ones(len(d), bool)
    for c in need_raw:
        ok &= np.isfinite(d[c].to_numpy(float))
    log["complete_case_missing_by_covariate"] = cc_missing
    log["rows_dropped_complete_case"] = int((~ok).sum())
    d = d.loc[ok].reset_index(drop=True)

    # 7. child exclusions
    excl: dict[str, int] = {}
    bad_child = ~(np.isfinite(d["age_c"].to_numpy(float)) & np.isfinite(d["sex_c"].to_numpy(float)))
    excl["missing_age_or_sex"] = int(d.loc[bad_child, "subject"].nunique())
    d = d.loc[~bad_child].reset_index(drop=True)
    if not np.isfinite(d["task_start_h_c"].to_numpy(float)).all():
        nb = d.loc[~np.isfinite(d["task_start_h_c"].to_numpy(float)), "subject"].nunique()
        raise ValueError(f"task_start_h undefined for {nb} child(ren) with task data")
    if task in ("FAST", "IC") and apply_nonperformance:   # False only for S-ALT
        from trials.child_qc import nonperformance_table

        full = pd.read_parquet(tpath)
        full = full[full["subject"].isin(subjects)]
        npf = nonperformance_table(full, tc)
        del full
        np_ex = set(npf.loc[npf["nonperformance_excluded"], "subject"])
        excl["nonperformance"] = int(len(np_ex & set(d["subject"])))
        d = d[~d["subject"].isin(np_ex)].reset_index(drop=True)
    counts = d.groupby("subject").size()
    few = set(counts.index[counts < min_rows])
    excl[f"fewer_than_{min_rows}_rows"] = int(len(few))
    d = d[~d["subject"].isin(few)].reset_index(drop=True)
    log["children_excluded"] = excl

    # 8. within-child z of t_run / tib over the final rows, then the spline bases
    d["t_run"] = K.zscore_within(d["t_run_raw"], d["subject"])
    d["tib"] = K.zscore_within(d["tib_raw"], d["subject"])
    for c in ("t_run", "tib"):
        bad = ~np.isfinite(d[c].to_numpy(float))
        if bad.any():           # a child with a constant position (SD 0) cannot be z-scored
            raise ValueError(f"{c} undefined (zero within-child SD) on {int(bad.sum())} rows")
    splines = [_spline_spec(knot_table, mdef.knots_key, v, df) for v, df in mdef.splines]
    for s in splines:
        B = ns_basis(d[s["variable"]].to_numpy(float), s["interior_knots"], s["boundary_knots"])
        for j, c in enumerate(s["columns"]):
            d[c] = B[:, j]
        x = d[s["variable"]].to_numpy(float)
        s["n_outside_boundary"] = int(((x < s["boundary_knots"][0]) | (x > s["boundary_knots"][1])).sum())

    # 9. ocular terms and the join indicators, only when non-constant
    oc_terms = model_inputs.ocular_terms(d)
    for ind in ("heog_missing", "vfp_missing"):
        d[ind] = d[ind].astype(bool).astype(float)
    avail = set(d.columns)
    for ind in ("heog_missing", "vfp_missing"):
        if ind not in oc_terms:
            avail.discard(ind)
    for j in ("tsp_join", "soa_join", "tsfl_join", "contour_join"):
        if j in d.columns and np.nanstd(d[j].to_numpy(float)) == 0:
            avail.discard(j)
    has_grp = False
    if coh == "CONF":
        has_grp = _attach_group(d, cfg)
        avail.add("grp")
    if mdef.h2:
        if not has_grp:
            raise ValueError(f"{mdef.name} needs grp (conf cohort): it is a group contrast")
        _attach_relq(d, cfg, task, ppath)
        avail.add("relq_c")
        log["relq_missing_children"] = int(d.loc[d["relq_missing"] > 0, "subject"].nunique())
        if d["relq_missing"].any():
            avail.add("relq_missing")
        else:
            avail.discard("relq_missing")
    from modeling.model_specs import expand_fixed

    terms = list(mdef.fixed) + (list(mdef.grp_terms) if has_grp else [])
    fixed_terms = expand_fixed(terms, sorted(avail), optional=mdef.optional)
    log["fixed_terms_omitted_constant"] = [t for t in mdef.optional if t not in fixed_terms]

    # 10. outcome
    out_col = mdef.outcome
    if outcome == "real":
        if out_col == "lrt":
            rt = d["rt_ms"].to_numpy(float)
            okrt = np.isfinite(rt) & (rt > 0)
            if not okrt.all() and not allow_missing_rt:
                raise ValueError("RT rows without a positive RT")
            d["lrt"] = np.where(okrt, np.log(np.where(okrt, rt, 1.0) / 1000.0), np.nan)
        else:
            d["miss"] = (~d["responded"].astype(bool)).astype(float)
    d = d.drop(columns=[c for c in ("rt_ms", "responded", "tt_extra_rows") if c in d.columns
                        and (outcome != "real" or c == "tt_extra_rows")])

    # 11. child identifier, the block factor, row order
    if pseudonymizer is None:
        raise ValueError("pseudonymizer (subject -> pid, utils.child_ids.Pseudonymizer) is required")
    mp = {s: pseudonymizer(s) for s in sorted(set(d["subject"]))}
    d.insert(0, "pid", d["subject"].map(mp))
    d["pid_run_block"] = d["pid"] + "|" + d["block"].astype(str)
    if not str(d["block"].iloc[0]).startswith(f"{d['run'].iloc[0]}:"):
        d["pid_run_block"] = d["pid"] + "|" + d["run"].astype(str) + ":" + d["block"].astype(str)
    d = d.drop(columns=["subject"]).sort_values(["pid", "run", "sample"], kind="stable").reset_index(drop=True)
    log["n_rows"] = int(len(d))
    log["n_children"] = int(d["pid"].nunique())
    log["rows_per_child_median"] = float(d.groupby("pid").size().median())
    log["n_blocks"] = int(d["pid_run_block"].nunique())
    meta = {"centring": {"age_mean": age_mean, "task_start_h_median": float(cent["task_start_h (child level)"]),
                         **({"soa_prev_median": float(cent["soa_prev"])} if task == "FAST" else {})},
            "knots_key": mdef.knots_key, "predictor_table": str(Path(ppath).name),
            "trial_table": str(Path(tpath).name)}
    return ModelData(mdef, task, coh, aset, variant, d, fixed_terms, splines, out_col,
                     "real" if outcome == "real" else "none", has_grp, True, log, meta)


def load_group_labels(cfg: ProjectConfig) -> pd.DataFrame:
    """``participant_id`` and normalised ``group`` (``ASD SIBLING`` -> ``SIB``, ``NAN`` -> ``n/a``) from
    ``participants.tsv``."""
    df = pd.read_csv(cfg.paths.bids_root / "participants.tsv", sep="\t", dtype=str,
                     usecols=["participant_id", "group"], keep_default_na=False, encoding="utf-8-sig")
    df["group"] = df["group"].str.strip().replace(GROUP_NORMALISATION)
    return df


def _attach_group(d: pd.DataFrame, cfg: ProjectConfig) -> bool:
    """``grp`` (TD -0.5, ASD +0.5) for the rows of the ``conf`` cohort."""
    g = load_group_labels(cfg)
    gm = dict(zip(g["participant_id"], g["group"]))
    d["grp"] = d["subject"].map(lambda s: GROUP_CODES.get(gm.get(s))).astype(float)
    if not np.isfinite(d["grp"].to_numpy(float)).all():
        raise ValueError("conf rows without a TD / ASD label")
    return True


def _attach_relq(d: pd.DataFrame, cfg: ProjectConfig, task: str, ppath: Path) -> None:
    """``relq_c``: child x task split-channel Spearman-Brown reliability of alpha, centred at the pooled median over
    all children (stored in ``results/alpha_reliability.json``)."""
    from feature_extraction import reliability as R

    p = pd.read_parquet(ppath, columns=["subject", "task", "run", "sample", "in_predictor_set", "valid", "alpha",
                                        "alpha_half_left", "alpha_half_right", "block"])
    rel = R.child_table(p)
    col = "rel_split_channel" if "rel_split_channel" in rel.columns else [c for c in rel.columns
                                                                         if "split_channel" in c][0]
    rj = json.loads((Path(cfg.root) / "results" / "alpha_reliability.json").read_text(encoding="utf-8"))
    med = float(rj["tasks"][task]["raw"]["split_channel"]["all"]["median"])
    m = dict(zip(rel["subject"] if "subject" in rel.columns else rel.index, rel[col]))
    r = d["subject"].map(m).astype(float) - med
    # A child without a computable split-channel reliability (too few windows with both hemisphere halves) is set
    # to the centre with a missingness indicator, like the ocular covariates.
    d["relq_missing"] = (~np.isfinite(r.to_numpy(float))).astype(float)
    d["relq_c"] = r.fillna(0.0)


# --------------------------------------------------------------------------- M5h
_TASK_LETTER = {"AVSRT": "A", "FAST": "F", "IC": "I"}


def combine_m5h(parts: Mapping[str, ModelData], *, include_grp: bool = False) -> ModelData:
    """M5h data (joint model of the three tasks) from the M1, M3 and M4 data with outcomes attached.

    ``y = (lrt - child x task mean) / s_task`` with the fixed s_task; fixed part ``0 + task + task:(<task covariates
    of M1 / M3 / M4 without grp and a_wb terms>) + a_wb:task``; random ``(0 + aA + aF + aI | pid)`` plus the
    task-specific child time slopes ``(0 + t_run_<T> | pid)``, ``(0 + tib_<T> | pid)`` (uncorrelated); no random
    intercept and no block intercept (y is centred within child x task). Targets: the task slopes, S4 (mean of FAST
    and IC), C1 (AVSRT minus the mean of FAST and IC), C2 (FAST minus IC) and the mean of the three.
    """
    frames, terms = [], []
    want = {"AVSRT", "FAST", "IC"}
    if set(parts) != want:
        raise ValueError(f"combine_m5h needs the parts {sorted(want)}")
    letters = [_TASK_LETTER[t] for t in ("AVSRT", "FAST", "IC")]
    all_cols: list[str] = []
    for task in ("AVSRT", "FAST", "IC"):
        md = parts[task]
        if "lrt" not in md.frame.columns:
            raise ValueError(f"{task} part has no outcome (build it with outcome='real')")
        L = _TASK_LETTER[task]
        f = md.frame
        g = pd.DataFrame({"pid": f["pid"].to_numpy(), "run": f["run"].to_numpy(), "sample": f["sample"].to_numpy(),
                          "trial_index_in_run": f["trial_index_in_run"].to_numpy(), "task": task,
                          "pid_run_block": f["pid_run_block"].astype(str).radd(f"{L}|").to_numpy(),
                          "block": f["block"].astype(str).to_numpy(), "a_wb": f["a_wb"].to_numpy(float),
                          "a_slow": f["a_slow"].to_numpy(float)})
        lrt = f["lrt"].to_numpy(float)
        g["y"] = (lrt - f.groupby("pid")["lrt"].transform("mean").to_numpy(float)) / S_TASK[task]
        for LL in letters:
            g[f"t{LL}"] = float(LL == L)
            g[f"a{LL}"] = g["a_wb"] * float(LL == L)
            g[f"trun{LL}"] = f["t_run"].to_numpy(float) * float(LL == L)
            g[f"tib{LL}"] = f["tib"].to_numpy(float) * float(LL == L)
        for t in md.fixed_terms:
            parts_ = t.split(":")
            if "a_wb" in parts_ or "grp" in parts_:
                continue
            name = f"{L}_{'X'.join(parts_)}"
            g[name] = np.prod([f[p].to_numpy(float) for p in parts_], axis=0)
            terms.append(name)
            all_cols.append(name)
        frames.append(g)
    d = pd.concat(frames, ignore_index=True)
    for c in all_cols:
        d[c] = d[c].fillna(0.0)
    d = d.sort_values(["pid", "task", "run", "sample"], kind="stable").reset_index(drop=True)
    fixed = [f"t{L}" for L in letters] + terms + [f"a{L}" for L in letters]
    rnd = RandomDef(corr=tuple(f"a{L}" for L in letters), droppable=(),
                    uncorrelated=tuple(f"trun{L}" for L in letters) + tuple(f"tib{L}" for L in letters),
                    block=False)
    targets = {"beta_A": {"aA": 1.0}, "beta_F": {"aF": 1.0}, "beta_I": {"aI": 1.0},
               "S4": {"aF": 0.5, "aI": 0.5}, "C1": {"aA": 1.0, "aF": -0.5, "aI": -0.5},
               "C2": {"aF": 1.0, "aI": -1.0}, "mean3": {"aA": 1 / 3, "aF": 1 / 3, "aI": 1 / 3}}
    mdef = ModelDef("M5h", "ALL", "lmm", "y", "primary_rt", "M1", (), tuple(fixed), rnd, targets,
                    sesoi={k: 0.03 for k in ("S4", "C1", "C2", "mean3", "beta_A", "beta_F", "beta_I")},
                    note="M5h (S4, cross-task contrasts C1 / C2)")
    kinds = {parts[t].outcome_kind for t in parts}
    cohorts = {parts[t].cohort for t in parts}
    if len(cohorts) != 1:
        raise ValueError("M5h parts from different cohorts")
    log = {t: parts[t].log for t in parts}
    log.update({"n_rows": int(len(d)), "n_children": int(d["pid"].nunique())})
    return ModelData(mdef, "ALL", cohorts.pop(), "primary_rt", "", d, fixed, [], "y",
                     "synthetic" if "synthetic" in kinds else kinds.pop(), include_grp, False, log,
                     {t: parts[t].meta for t in parts})


# --------------------------------------------------------------------------- model-input TSV
def write_model_input(md: ModelData, path: str | os.PathLike[str]) -> Path:
    """Write the model-input TSV read by the R engine (UTF-8, LF, ``%.17g`` floats, ``NA``); return its path."""
    if md.outcome_column not in md.frame.columns:
        raise ValueError("the model data have no outcome (outcome='none')")
    cols = md.columns_needed()
    out = md.frame[cols].copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(float)
    path = Path(path)
    out.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g", lineterminator="\n")
    return path
