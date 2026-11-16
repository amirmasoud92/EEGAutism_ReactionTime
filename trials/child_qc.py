"""Child-level behavioural exclusion from the trial tables (FAST and IC).

Computed per child x task from the combined trial tables.

Behavioural non-performance
    A child x task is excluded when its hit rate is below ``min_hit_rate`` (0.5) **or** its false alarms exceed
    its hits. The rule is configured per task in ``configs/events.yaml`` (``nonperformance:``):

    - FAST, ``false_alarm_unit: nontarget_trials``: false alarms are **non-target trials** with at least one
      non-transition press in ``[onset, next stimulus)`` (column ``false_alarm``). Hits and false alarms are then
      counted in the same unit (one per stimulus). Block-transition presses (<= 0.5 s before a block start) are not
      false alarms.
    - IC, ``false_alarm_unit: presses``: IC has no non-target trial rows, so false alarms are **presses** that no
      flip's primary response window explains (press class ``false_alarm``: latency > 1200 ms from the last flip,
      or before the first flip), attributed to flip intervals by ``n_false_alarms_after_flip`` (plus
      ``n_false_alarms_before_first_flip`` once per run).

    Scope: rows not excluded by a known issue, confirmed by Status, not truncated, and (IC) not after the
    foreign-trigger cut and not beyond the IC sequence cap. The hit rate is computed over the analysed stimuli
    under the primary window: FAST targets; IC in-stream flips (not ``post_pause_first_flip``). The
    false alarms of IC are counted over the same span of the recording, post-pause intervals included.

The minimum of 30 analysable trials per child and task is applied when the model data are assembled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.config import ConfigError, TaskEventsConfig

#: False-alarm units understood by :func:`nonperformance_table`, per task.
FA_UNITS: dict[str, tuple[str, ...]] = {
    "FAST": ("nontarget_trials", "nontarget_presses", "presses_all_rows"),
    "IC": ("presses", "flip_intervals"),
}


@dataclass(frozen=True)
class NonperformanceRule:
    """Non-performance rule of one task (``tasks.<task>.nonperformance`` in ``configs/events.yaml``)."""

    task: str
    min_hit_rate: float = 0.5
    false_alarm_unit: str = "nontarget_trials"

    @classmethod
    def from_task_cfg(cls, cfg: TaskEventsConfig) -> NonperformanceRule | None:
        d: Mapping[str, Any] | None = cfg.extra.get("nonperformance")
        if d is None:
            return None
        where = f"tasks.{cfg.name}.nonperformance"
        if not isinstance(d, Mapping) or set(d) - {"min_hit_rate", "false_alarm_unit"}:
            raise ConfigError(f"{where}: only min_hit_rate and false_alarm_unit are allowed")
        mhr = d.get("min_hit_rate", 0.5)
        if isinstance(mhr, bool) or not isinstance(mhr, (int, float)) or not 0 <= float(mhr) <= 1:
            raise ConfigError(f"{where}.min_hit_rate must be a number in [0, 1]")
        unit = str(d.get("false_alarm_unit", FA_UNITS.get(cfg.name, ("",))[0]))
        if unit not in FA_UNITS.get(cfg.name, ()):
            raise ConfigError(f"{where}.false_alarm_unit {unit!r} not in {FA_UNITS.get(cfg.name, ())}")
        return cls(cfg.name, float(mhr), unit)


def _b(df: pd.DataFrame, col: str, na: bool = False) -> np.ndarray:
    return df[col].fillna(na).to_numpy(bool)


def nonperformance_table(trials: pd.DataFrame, cfg: TaskEventsConfig, rule: NonperformanceRule | None = None
                         ) -> pd.DataFrame:
    """Per-child hit rate, false-alarm counts in every unit and the exclusion decision.

    Parameters
    ----------
    trials
        Combined (child-level) trial table of FAST or IC.
    cfg
        The task's event config (gives the rule when ``rule`` is None).

    Returns
    -------
    pandas.DataFrame
        One row per child with rows: ``subject``, ``task``, ``n_scored`` (targets / in-stream flips scored),
        ``n_hits``, ``hit_rate``, ``n_fa_<unit>`` for every unit of the task, ``n_fa`` (the configured unit),
        ``low_hit_rate``, ``fa_exceed_hits`` and ``nonperformance_excluded``. Keyed by BIDS ID.
    """
    rule = rule or NonperformanceRule.from_task_cfg(cfg)
    if rule is None:
        raise ValueError(f"{cfg.name}: no nonperformance rule configured")
    t = trials
    base = ~_b(t, "excluded_known_issue") & ~_b(t, "stim_unverified") & ~_b(t, "post_stim_truncated")
    kids = sorted(t["subject"].unique())
    out = pd.DataFrame({"subject": kids, "task": cfg.name})
    if cfg.name == "FAST":
        tgt = base & _b(t, "is_target")
        nt = base & ~_b(t, "is_target")
        g = t.loc[tgt].groupby("subject")
        out["n_scored"] = out["subject"].map(g.size()).fillna(0).astype("int64")
        out["n_hits"] = out["subject"].map(g["responded"].sum()).fillna(0).astype("int64")
        ntt = t.loc[nt]
        fa_tr = ntt.assign(x=ntt["false_alarm"].fillna(False).astype(bool)).groupby("subject")["x"].sum()
        out["n_fa_nontarget_trials"] = out["subject"].map(fa_tr).fillna(0).astype("int64")
        out["n_fa_nontarget_presses"] = out["subject"].map(
            ntt.groupby("subject")["n_fa_presses"].sum()).fillna(0).astype("int64")
        out["n_fa_presses_all_rows"] = out["subject"].map(
            t.loc[base].groupby("subject")["n_fa_presses"].sum()).fillna(0).astype("int64")
    elif cfg.name == "IC":
        span = base & ~_b(t, "after_foreign_cut") & ~_b(t, "beyond_ic_cap")
        scored = span & ~_b(t, "post_pause_first_flip")
        g = t.loc[scored].groupby("subject")
        out["n_scored"] = out["subject"].map(g.size()).fillna(0).astype("int64")
        out["n_hits"] = out["subject"].map(g["responded"].sum()).fillna(0).astype("int64")
        sp = t.loc[span]
        after = sp.groupby("subject")["n_false_alarms_after_flip"].sum()
        before = sp.groupby(["subject", "run"])["n_false_alarms_before_first_flip"].first().groupby("subject").sum()
        out["n_fa_presses"] = (out["subject"].map(after).fillna(0) + out["subject"].map(before).fillna(0)
                               ).astype("int64")
        out["n_fa_flip_intervals"] = out["subject"].map(
            sp.assign(x=sp["n_false_alarms_after_flip"] > 0).groupby("subject")["x"].sum()).fillna(0).astype("int64")
    else:
        raise ValueError(f"no non-performance rule for task {cfg.name}")
    out["hit_rate"] = np.where(out["n_scored"] > 0, out["n_hits"] / out["n_scored"].where(out["n_scored"] > 0), np.nan)
    out["n_fa"] = out[f"n_fa_{rule.false_alarm_unit}"].astype("int64")
    out["low_hit_rate"] = (out["hit_rate"] < rule.min_hit_rate).fillna(False).astype(bool)
    out["fa_exceed_hits"] = (out["n_fa"] > out["n_hits"]).astype(bool)
    out["nonperformance_excluded"] = out["low_hit_rate"] | out["fa_exceed_hits"]
    return out
