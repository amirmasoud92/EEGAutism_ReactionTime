"""Model data for the analyses of the autistic and typically developing children (``conf``; ``dev`` = unaffected
siblings).

:class:`RealProvider` builds every model's data with :func:`modeling.model_data.build_model_data`
(``outcome="real"``; ``grp`` is attached for ``conf``) and the sensitivity analyses with
:func:`modeling.sensitivity.build` / ``attach_outcome``. It also loads the child-level covariates (FSIQ, ADOS-2 CSS,
SRS-2 T, medication) from ``participants.tsv``, the reliability windows, the alpha windows and the context counts.
Child identifiers (``pid``) are BIDS participant IDs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import sensitivity as SENS
from modeling.model_data import build_model_data, load_group_labels

__all__ = ["RealProvider", "InputFileError", "load_child_covariates", "CHILD_COVARIATE_COLUMNS", "ADOS_CSS_RANGE",
           "ados_css_in_range"]

#: participants.tsv columns read by :func:`load_child_covariates` (each with its ``_dup`` copy when present).
CHILD_COVARIATE_COLUMNS = ("fsiq", "ados_css", "srs2_total_t", "medication")
_MISSING = {"", "n/a", "na", "nan", "NAN", "N/A", "-"}
#: ADOS-2 calibrated severity score range; values outside it become missing.
ADOS_CSS_RANGE = (1.0, 10.0)


class InputFileError(RuntimeError):
    """A stored input of a sensitivity analysis (S-ALT step) is missing."""


def ados_css_in_range(values: dict) -> tuple[dict, int]:
    """``(kept, n_set_missing)``: the ADOS-2 CSS values inside :data:`ADOS_CSS_RANGE`."""
    lo, hi = ADOS_CSS_RANGE
    kept = {k: float(v) for k, v in values.items() if v is not None and lo <= float(v) <= hi}
    n_bad = sum(1 for v in values.values() if v is not None and v == v and not lo <= float(v) <= hi)
    return kept, n_bad


def _clean(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s.isna() | s.isin(_MISSING), pd.NA)


def load_child_covariates(cfg, pseudonymizer) -> dict:
    """``{"fsiq": {pid: float}, "ados_css": {pid: float}, "srs2": {pid: float}, "medication": {pid: (orig, dup)},
    "counts": {...}}`` for every participants row. ``_dup`` is used only where the original is missing (for the
    medication strings :func:`modeling.sensitivity.medication_class` applies the same rule). ADOS-2 CSS values
    outside 1-10 become missing."""
    path = cfg.paths.bids_root / "participants.tsv"
    head = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8-sig").columns
    want = ["participant_id"] + [c for base in CHILD_COVARIATE_COLUMNS for c in (base, f"{base}_dup") if c in head]
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=want, keep_default_na=False, encoding="utf-8-sig")
    pid = [pseudonymizer(s) for s in df["participant_id"]]
    out: dict = {"counts": {}}

    def coalesce(base: str) -> pd.Series:
        o = _clean(df[base]) if base in df else pd.Series(pd.NA, index=df.index, dtype="string")
        d = _clean(df[f"{base}_dup"]) if f"{base}_dup" in df else pd.Series(pd.NA, index=df.index, dtype="string")
        return o.where(o.notna(), d)

    for base, key in (("fsiq", "fsiq"), ("ados_css", "ados_css"), ("srs2_total_t", "srs2")):
        v = pd.to_numeric(coalesce(base), errors="coerce")
        vals = {p: float(x) for p, x in zip(pid, v) if pd.notna(x)}
        if key == "ados_css":
            vals, n_bad = ados_css_in_range(vals)
            out["counts"]["ados_css_out_of_range_set_missing"] = int(n_bad)
        out[key] = vals
        out["counts"][f"{key}_non_missing"] = len(vals)
    mo = df["medication"].astype(str) if "medication" in df else pd.Series("", index=df.index)
    md = df["medication_dup"].astype(str) if "medication_dup" in df else pd.Series("", index=df.index)
    out["medication"] = {p: (a, b) for p, a, b in zip(pid, mo, md)}
    out["source"] = "participants.tsv"
    del df
    return out


class RealProvider:
    def __init__(self, cfg, pseudonymizer, *, log=None):
        self.cfg = cfg
        self.ps = pseudonymizer
        self.log = log or (lambda m: None)
        self._cache: dict = {}
        self._covariates = None

    def build(self, model: str, *, cohort: str = "conf", variant: str = "", analysis_set: str | None = None,
              extra_predictor_columns=(), allow_missing_rt: bool = False, **kw):
        key = (model, cohort, variant, analysis_set, tuple(extra_predictor_columns), allow_missing_rt,
               tuple(sorted(kw.items())))
        if key not in self._cache:
            self.log(f"build {model} {cohort} {variant or 'primary'} {analysis_set or ''}")
            md = build_model_data(model, cohort, analysis_set=analysis_set, variant=variant, outcome="real",
                                  cfg=self.cfg, extra_predictor_columns=tuple(extra_predictor_columns),
                                  allow_missing_rt=allow_missing_rt, pseudonymizer=self.ps, **kw)
            self._cache[key] = md
        return self._cache[key]

    def model(self, name: str):
        return self.build(name)

    def sensitivity(self, spec_id: str, model: str, ctx: Mapping):
        md = SENS.build(spec_id, model, "conf", cfg=self.cfg, ctx=ctx, mode="real", pseudonymizer=self.ps)
        return SENS.attach_outcome(md, spec_id, ctx=ctx)

    def miss_model(self, model: str):
        """E8: FAST / IC predictor-set rows with ``miss`` = no hit (the RT model's covariates)."""
        md = self.build(model, analysis_set="predictor_set", allow_missing_rt=True)
        return md

    def dev_m1_sib(self):
        """M1 on the unaffected siblings (``dev`` cohort; no group term)."""
        md = self.build("M1", cohort="dev")
        g = load_group_labels(self.cfg)
        sib = {self.ps(s) for s, lab in zip(g["participant_id"], g["group"]) if lab == "SIB"}
        del g
        f = md.frame[md.frame["pid"].isin(sib)].reset_index(drop=True)
        return SENS.refinalise(SENS._replace(md, frame=f), reason="unaffected siblings only")

    def rel_windows(self, task: str) -> pd.DataFrame:
        from modeling import srel

        return srel.load_rel_windows(task, "conf", cfg=self.cfg, pseudonymizer=self.ps)

    def alpha_windows(self, task: str) -> pd.DataFrame:
        """E3a: predictor-set windows with a valid alpha window (predictor-only)."""
        p = self.cfg.paths.derivatives / "features" / task / f"{task}_conf_features_predictor.parquet"
        t = pd.read_parquet(p, columns=["subject", "in_predictor_set", "valid", "alpha"])
        m = (pd.array(t["in_predictor_set"], dtype="boolean").fillna(False).to_numpy(bool)
             & pd.array(t["valid"], dtype="boolean").fillna(False).to_numpy(bool)
             & np.isfinite(t["alpha"].to_numpy(float)))
        t = t.loc[m]
        mp = {s: self.ps(s) for s in sorted(set(t["subject"]))}
        return pd.DataFrame({"pid": t["subject"].map(mp).to_numpy(), "alpha": t["alpha"].to_numpy(float)})

    def child_covariates(self) -> dict:
        if self._covariates is None:
            self._covariates = load_child_covariates(self.cfg, self.ps)
        return self._covariates

    def alt_step(self) -> dict:
        """S-ALT: the rejection step whose alternative is applied (the step that contributes most to a group
        difference in retained trials), read from the stored retention decomposition."""
        p = Path(self.cfg.paths.results) / "retention_steps.json"
        if not p.is_file():
            raise InputFileError(f"{p.name} missing (S-ALT step)")
        sa = json.loads(p.read_text(encoding="utf-8")).get("s_alt") or {}
        if not sa.get("step"):
            raise InputFileError(f"no S-ALT step in {p.name}")
        return {"step": sa["step"], "applicable": sa.get("applicable"), "source": p.name}

    def retfrac(self) -> dict[str, dict[str, float]]:
        """S-RET: per task, child -> retained fraction (:func:`modeling.sensitivity.retained_fraction`)."""
        return {t: SENS.retained_fraction(t, "conf", cfg=self.cfg, pseudonymizer=self.ps) for t in ("AVSRT", "FAST", "IC")}

    def context_counts(self, task: str) -> pd.DataFrame:
        """E2b: the behaviour-side context counts."""
        p = self.cfg.paths.derivatives / "features" / task / f"{task}_conf_context_counts_behaviour.parquet"
        t = pd.read_parquet(p, columns=["subject", "run", "sample", "n_stim10", "n_press10"])
        mp = {s: self.ps(s) for s in sorted(set(t["subject"]))}
        t.insert(0, "pid", t["subject"].map(mp))
        return t.drop(columns=["subject"])
