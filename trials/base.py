"""Shared trial-table logic: one row per trial stimulus, never dropped.

:class:`TaskTrialBuilder` is a template: :meth:`TaskTrialBuilder.build` runs the
shared steps (validation, known-issue recoding, response matching, generic
covariates, lookback contamination, Status blocks, join proximity, history and
misattribution flags, known-issue exclusion flags) and then calls the abstract
hook :meth:`TaskTrialBuilder.add_task_columns` for task-specific columns. The
result is validated against :func:`trials.schema.trial_table_schema`
plus the builder's :meth:`extra_columns` before it is returned.

Timing conventions
------------------
- All intervals use integer samples of the recording; ms = samples * 1000 / sfreq.
- "Before onset" is strict (an event at the onset sample belongs to the trial).
- Lookback contamination for L: an event in ``[t0 - L, t0)``.
- Intervals that span a Status bit-16 join are not real time (recording was
  paused); the main interval columns are NA across joins, ``*_concat`` columns
  keep the concatenated-clock value and ``*_crosses_join`` says which rows
  cross. Models that need these covariates use the ``*_concat`` value plus the
  ``*_crosses_join`` indicator, so no trial is dropped silently by a
  complete-case fit.
- Blocks: ``block_index_in_run`` = 1 + block-start markers at or before the
  onset, in every run and every table (rows before the first marker form block
  1, flagged ``block_start_unmarked``).
- Exclusions never remove rows: they set boolean flags. Stimuli that Status
  does not confirm, or that lie at or beyond the end of the recording, are
  rows too, flagged ``stim_unverified``; events.tsv rows whose code is in no
  vocabulary of the task are counted (:meth:`TaskTrialBuilder.unclassified_events`).

Extension points for task builders
----------------------------------
- :meth:`TaskTrialBuilder.prepare` runs before the shared steps (e.g. IC
  removes a foreign-paradigm segment from the context streams).
- :meth:`TaskTrialBuilder.add_task_columns` adds task columns (declared in
  :meth:`TaskTrialBuilder.extra_columns`).
- :meth:`TaskTrialBuilder.first_marker_sample` decides where the
  ``exclude_events_before_first_marker`` known-issue action starts (FAST pairs
  201 with 202/203).
- :meth:`TaskTrialBuilder.child_extra_columns` /
  :meth:`TaskTrialBuilder.add_child_columns` add child-level columns when runs
  are assembled (:meth:`TaskTrialBuilder.assemble_children`).
- ``rt_targets_only`` (task config) clears RT columns on non-target rows.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from trials.io import read_events_tsv, validate_events
from trials.matching import as_sorted_samples, last_before, match_responses, next_after
from trials.schema import (
    CHILD_BASE_COLUMNS,
    ColumnSpec,
    TableSchema,
    apply_subset,
    soa_stratum_labels,
    trial_table_schema,
)
from trials.status import CrosscheckReport, crosscheck_events, decode_status_file, joins_between
from utils.bids_index import Recording
from utils.config import StatusConfig, TaskEventsConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)

#: Action types that change trial rows (others concern preprocessing / participants only).
ROW_ACTIONS = frozenset({"exclude_recording", "recode_event", "exclude_sample_range", "exclude_time_range",
                         "exclude_events_before_first_marker", "exclude_partial_blocks", "flag_only"})


def nullable_int(a: np.ndarray, valid: np.ndarray) -> pd.arrays.IntegerArray:
    """Nullable Int64 array: ``a`` where ``valid``, NA elsewhere."""
    out = pd.array(np.where(valid, a, 0).astype(np.int64), dtype="Int64")
    out[~np.asarray(valid, bool)] = pd.NA
    return out


def nullable_bool(a: np.ndarray, valid: np.ndarray) -> pd.arrays.BooleanArray:
    """Nullable boolean array: ``a`` where ``valid``, NA elsewhere."""
    out = pd.array(np.asarray(a, bool), dtype="boolean")
    out[~np.asarray(valid, bool)] = pd.NA
    return out


# --------------------------------------------------------------------------- inputs
@dataclass
class RecordingEvents:
    """Everything a builder needs from one recording.

    Attributes
    ----------
    subject, task, run
        Recording keys.
    sfreq
        Sampling rate.
    events
        events.tsv rows in file order (``onset``, ``sample``, ``value``, ...).
    markers
        Structure markers (``sample``, ``value``) from Status (and events.tsv
        rows with a marker code); ``None`` if Status was not read.
    joins
        Samples of bit-16 joins; ``None`` if Status was not read.
    n_times
        Recording length in samples (``None`` if unknown).
    crosscheck
        events.tsv vs Status report, if Status was read.
    sources
        Input files.
    """

    subject: str
    task: str
    run: int
    sfreq: float
    events: pd.DataFrame
    markers: pd.DataFrame | None = None
    joins: np.ndarray | None = None
    n_times: int | None = None
    crosscheck: CrosscheckReport | None = None
    sources: tuple[Path, ...] = ()


def load_recording_events(
    recording: Recording,
    task_cfg: TaskEventsConfig,
    status_cfg: StatusConfig,
    sfreq: float,
    read_status: bool = True,
) -> RecordingEvents:
    """Load events.tsv and (optionally) Status markers and joins of a recording.

    Parameters
    ----------
    recording
        Entry of :class:`~utils.bids_index.RecordingIndex`.
    task_cfg, status_cfg
        Event conventions.
    sfreq
        Expected sampling rate (checked against the BDF header).
    read_status
        Read the BDF Status channel (markers, joins, cross-check).
    """
    if recording.events_path is None:
        raise FileNotFoundError(f"{recording.data_path.name}: no events.tsv")
    ev = read_events_tsv(recording.events_path)
    sources: list[Path] = [recording.events_path]
    markers = joins = crosscheck = n_times = None
    tsv_markers = ev.loc[ev["value"].isin(list(task_cfg.marker_codes)), ["sample", "value"]]
    if read_status and recording.fmt == "bdf":
        dec = decode_status_file(recording.data_path, status_cfg)
        if abs(dec.sfreq - sfreq) > 1e-6:
            raise ValueError(f"{recording.data_path.name}: Status sfreq {dec.sfreq} != configured {sfreq}")
        st_markers = dec.events.loc[dec.events["value"].isin(list(task_cfg.marker_codes)), ["sample", "value"]]
        markers = (pd.concat([st_markers, tsv_markers]).drop_duplicates().sort_values(["sample", "value"])
                   .reset_index(drop=True))
        joins = dec.joins
        crosscheck = crosscheck_events(ev, dec)
        n_times = dec.n_times
        sources.append(recording.data_path)
    return RecordingEvents(recording.subject, recording.task, recording.run, float(sfreq), ev, markers, joins,
                           n_times, crosscheck, tuple(sources))


@dataclass
class BuildContext:
    """Arrays shared between the base steps and the task hook (sample clock)."""

    trial_samples: np.ndarray
    stim_samples: np.ndarray
    stim_values: np.ndarray
    press_samples: np.ndarray
    bound_samples: np.ndarray
    markers: pd.DataFrame | None
    joins: np.ndarray | None
    events: pd.DataFrame
    sfreq: float
    extra: dict = field(default_factory=dict)

    def ms(self, n_samples: np.ndarray | float) -> np.ndarray:
        return np.asarray(n_samples, dtype=float) * 1000.0 / self.sfreq


# --------------------------------------------------------------------------- builder
class TaskTrialBuilder(ABC):
    """Abstract trial-table builder.

    Parameters
    ----------
    task_cfg
        Event conventions of the task (``configs/events.yaml``).
    status_cfg
        Status decoding parameters.
    registry
        Known-issues registry (optional; without it no issue flags are set).

    Notes for subclasses
    --------------------
    - Set the class attribute ``task``.
    - Implement :meth:`add_task_columns` (may return the frame unchanged) and
      declare every added column in :meth:`extra_columns`.
    - Override :meth:`condition_labels` / :meth:`target_mask` if needed.
    """

    task: ClassVar[str]

    def __init__(self, task_cfg: TaskEventsConfig, status_cfg: StatusConfig,
                 registry: KnownIssuesRegistry | None = None) -> None:
        if getattr(self, "task", None) and task_cfg.name != self.task:
            raise ValueError(f"{type(self).__name__} is for task {self.task}, got config for {task_cfg.name}")
        self.cfg = task_cfg
        self.status_cfg = status_cfg
        self.registry = registry

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def add_task_columns(self, trials: pd.DataFrame, rec: RecordingEvents, ctx: BuildContext) -> pd.DataFrame:
        """Add task-specific columns (declare them in :meth:`extra_columns`)."""

    def extra_columns(self) -> tuple[ColumnSpec, ...]:
        """Specs of the columns added by :meth:`add_task_columns`."""
        return ()

    def condition_labels(self, values: np.ndarray) -> np.ndarray:
        """Condition label per trial code (default: ``stimulus_codes`` labels)."""
        return np.array([self.cfg.stimulus_codes.get(int(v), str(v)) for v in values], dtype=object)

    def target_mask(self, values: np.ndarray) -> np.ndarray:
        """Which trial rows are targets (default: ``target_codes``)."""
        return np.isin(values, list(self.cfg.target_codes))

    def schema(self) -> TableSchema:
        """Strict schema of this builder's per-recording trial table."""
        return trial_table_schema(self.cfg, self.extra_columns())

    def prepare(self, rec: RecordingEvents) -> RecordingEvents:
        """Pre-build hook applied by :meth:`build` (default: identity). Must be idempotent."""
        return rec

    def issues_for(self, rec: RecordingEvents) -> list:
        """Known issues of the recording (empty without a registry)."""
        return self.registry.for_recording(rec.subject, rec.task, rec.run) if self.registry else []

    def first_marker_sample(self, ctx: BuildContext, marker_codes: Iterable[int]) -> int | None:
        """Sample where the task proper starts, for ``exclude_events_before_first_marker``.

        Default: the earliest marker with one of ``marker_codes``. Returns
        ``None`` when markers were not read (every row is then flagged), and
        a sample beyond the recording when no such marker exists (no row is
        flagged).
        """
        if ctx.markers is None:
            return None
        m = ctx.markers.loc[ctx.markers["value"].isin(list(marker_codes)), "sample"]
        return int(m.min()) if len(m) else int(np.iinfo(np.int64).max)

    # ------------------------------------------------------------------ child level
    def child_extra_columns(self) -> tuple[ColumnSpec, ...]:
        """Child-level columns added by :meth:`add_child_columns` (beyond the base ones)."""
        return ()

    def add_child_columns(self, df: pd.DataFrame, run_table: pd.DataFrame | None = None) -> pd.DataFrame:
        """Hook: add task child-level columns to the assembled table (default: none).

        Parameters
        ----------
        df
            Concatenated per-run tables with the base child columns.
        run_table
            Optional per-recording table (``subject``, ``task``, ``run`` and
            task counts such as ``n_contours``) covering runs without rows.
        """
        return df

    def child_schema(self) -> TableSchema:
        """Schema of the child-level table (per-recording schema + child columns)."""
        return self.schema().extend(CHILD_BASE_COLUMNS + tuple(self.child_extra_columns()),
                                    name=f"trials_{self.cfg.name}_child")

    def assemble_children(self, tables: Iterable[pd.DataFrame], run_table: pd.DataFrame | None = None) -> pd.DataFrame:
        """Concatenate per-run tables of one or more children and add child-level columns.

        Every interval stays within its run (each run is built separately); the
        child-level indices follow (run, sample). The output is validated
        against :meth:`child_schema`.
        """
        schema = self.child_schema()
        df = concat_recordings(tables)
        if df.empty:
            return schema.conform(empty_frame(schema))
        df = add_child_level_indices(df).reset_index(drop=True)
        df["n_runs_in_child"] = df.groupby(["subject", "task"])["run"].transform("nunique").astype("int64")
        df = self.add_child_columns(df, run_table)
        return schema.conform(df)

    # ------------------------------------------------------------------ main
    def build(self, rec: RecordingEvents) -> pd.DataFrame:
        """Build and validate the trial table of one recording.

        Raises
        ------
        trials.io.EventsIntegrityError
            If events are non-monotonic or samples disagree with onsets.
        trials.schema.SchemaError
            If the output does not match the schema.
        """
        if rec.task != self.cfg.name:
            raise ValueError(f"recording task {rec.task} != builder task {self.cfg.name}")
        rec = self.prepare(rec)
        validate_events(rec.events, rec.sfreq, self.status_cfg.sample_onset_tolerance)
        ev = rec.events.copy()
        ev["value_original"] = ev["value"]
        issues = self.issues_for(rec)
        recode_ids = self.apply_recodes(ev, issues, rec)

        sf = rec.sfreq
        is_trial = ev["value"].isin(list(self.cfg.trial_codes)).to_numpy()
        tr_ev = ev.loc[is_trial].reset_index(drop=True)
        ctx = BuildContext(
            trial_samples=as_sorted_samples(tr_ev["sample"].to_numpy()),
            stim_samples=as_sorted_samples(ev.loc[ev["value"].isin(list(self.cfg.stimulus_codes)), "sample"].to_numpy()),
            stim_values=ev.loc[ev["value"].isin(list(self.cfg.stimulus_codes)), "value"].to_numpy(np.int64),
            press_samples=as_sorted_samples(ev.loc[ev["value"].isin(list(self.cfg.response_codes)), "sample"].to_numpy()),
            bound_samples=as_sorted_samples(ev.loc[ev["value"].isin(list(self.cfg.response_bound_codes)), "sample"].to_numpy()),
            markers=rec.markers, joins=None if rec.joins is None else as_sorted_samples(rec.joins),
            events=ev, sfreq=sf,
        )
        s = ctx.trial_samples
        n = s.size
        vals = tr_ev["value"].to_numpy(np.int64)
        df = pd.DataFrame({
            "subject": pd.array([rec.subject] * n, dtype="string"),
            "task": pd.array([rec.task] * n, dtype="string"),
            "run": np.full(n, rec.run, np.int64),
            "sample": s,
            "onset_s": tr_ev["onset"].to_numpy(float),
            "trial_index_in_run": np.arange(1, n + 1, dtype=np.int64),
        })
        df["trial_key"] = pd.array([f"{rec.subject}_{rec.task}_run-{rec.run:02d}_s{x}" for x in s], dtype="string")
        df["value"] = vals
        df["value_original"] = tr_ev["value_original"].to_numpy(np.int64)
        df["condition"] = pd.array(self.condition_labels(vals), dtype="string")
        df["is_target"] = self.target_mask(vals).astype(bool)

        self._add_matching(df, ctx)
        self._add_sequence_covariates(df, ctx)
        self._add_contamination(df, ctx)
        self._add_blocks(df, ctx)
        self._add_joins(df, ctx)
        self._add_history_flags(df, ctx)
        self._add_integrity(df, rec)
        self._add_press_clear(df, ctx)
        self._add_edge_validity(df, ctx, rec.n_times)
        self.add_known_issue_flags(df, ctx, issues, recode_ids)
        df = self.add_task_columns(df.copy(), rec, ctx)     # copy: de-fragment before the task columns are added
        return self.schema().conform(df)

    def vocabulary(self) -> frozenset[int]:
        """Every event code the task knows: stimulus, response and marker codes."""
        return frozenset(self.cfg.stimulus_codes) | frozenset(self.cfg.response_codes) | frozenset(self.cfg.marker_codes)

    def unclassified_events(self, rec: RecordingEvents) -> pd.DataFrame:
        """events.tsv rows (after :meth:`prepare`) whose code is in no vocabulary of the task.

        They become no row of any table; a recording with such codes needs an
        ``allow_unclassified_codes`` known-issue action.
        """
        ev = self.prepare(rec).events
        return ev.loc[~ev["value"].isin(list(self.vocabulary())), ["sample", "value"]].reset_index(drop=True)

    # ------------------------------------------------------------------ steps
    def apply_recodes(self, ev: pd.DataFrame, issues: list, rec: RecordingEvents) -> dict[int, list[str]]:
        """Apply ``recode_event`` actions to ``ev`` in place; returns sample -> issue ids."""
        hits: dict[int, list[str]] = {}
        for iss in issues:
            for a in iss.actions:
                if a.type != "recode_event":
                    continue
                m = (ev["sample"] == a.get("sample")) & (ev["value"] == a.get("from_value"))
                if not m.any():
                    log.warning("%s: recode target sample %s value %s not found in %s run %s", iss.id,
                                a.get("sample"), a.get("from_value"), rec.subject, rec.run)
                    continue
                ev.loc[m, "value"] = int(a.get("to_value"))
                hits.setdefault(int(a.get("sample")), []).append(iss.id)
        return hits

    def _add_matching(self, df: pd.DataFrame, ctx: BuildContext) -> None:
        wins = {w.name: (w.lo_ms, w.hi_ms) for w in self.cfg.rt_windows.values()}
        mr = match_responses(ctx.trial_samples, ctx.bound_samples, ctx.press_samples, ctx.sfreq, wins,
                             self.cfg.primary_window)
        b = mr.bound_sample
        df["bound_sample"] = nullable_int(b, b >= 0)
        df["t_to_bound_ms"] = np.where(b >= 0, ctx.ms(b - ctx.trial_samples), np.nan)
        df["n_press_until_bound"] = mr.n_press_until_bound
        df["first_press_lat_ms"] = mr.first_press_lat_ms
        df["raw_first_press_lat_ms"] = mr.raw_first_press_lat_ms
        df["raw_first_press_after_bound"] = pd.array(list(mr.raw_first_press_after_bound), dtype="boolean")
        df["n_anticipations"] = mr.n_anticipations
        df["n_late_presses"] = mr.n_late_presses
        prim = self.cfg.primary_window
        df["rt_ms"] = mr.rt_ms[prim]
        df["responded"] = np.isfinite(mr.rt_ms[prim])
        rs = mr.response_sample[prim]
        df["response_sample"] = nullable_int(rs, rs >= 0)
        df["n_extra_presses"] = mr.n_extra_presses
        n_valid = mr.n_press_until_bound - mr.n_anticipations - mr.n_late_presses
        df["n_extra_in_window"] = np.where(df["responded"].to_numpy(bool), np.maximum(n_valid - 1, 0),
                                           0).astype(np.int64)
        tgt = df["is_target"].to_numpy()
        resp = df["responded"].to_numpy()
        df["outcome"] = pd.array(np.where(tgt, np.where(resp, "hit", "miss"),
                                          np.where(resp, "press", "no_press")), dtype="string")
        for w in self.cfg.rt_windows.values():
            if w.name == prim:
                continue
            df[f"rt_ms_{w.name}"] = mr.rt_ms[w.name]
            df[f"responded_{w.name}"] = np.isfinite(mr.rt_ms[w.name])
        if self.cfg.rt_targets_only:
            self._clear_nontarget_rt(df)

    def _clear_nontarget_rt(self, df: pd.DataFrame) -> None:
        """``rt_targets_only``: RT is defined for targets only.

        On non-target rows ``rt_ms`` and the sensitivity-window RTs become NaN,
        ``responded*`` False, ``response_sample`` NA and ``n_extra_presses`` 0;
        ``outcome`` is ``press`` if any press lies in ``[onset, bound)``, else
        ``no_press`` (task builders may refine it). Press counts and first-press
        latencies are kept.
        """
        nt = ~df["is_target"].to_numpy(bool)
        if not nt.any():
            return
        df.loc[nt, "rt_ms"] = np.nan
        df.loc[nt, "responded"] = False
        rs = df["response_sample"].copy()
        rs[nt] = pd.NA
        df["response_sample"] = rs
        df.loc[nt, "n_extra_presses"] = 0
        df.loc[nt, "n_extra_in_window"] = 0
        for w in self.cfg.rt_windows.values():
            if w.name != self.cfg.primary_window:
                df.loc[nt, f"rt_ms_{w.name}"] = np.nan
                df.loc[nt, f"responded_{w.name}"] = False
        out = df["outcome"].to_numpy(dtype=object).copy()
        out[nt] = np.where(df["n_press_until_bound"].to_numpy(np.int64)[nt] > 0, "press", "no_press")
        df["outcome"] = pd.array(out, dtype="string")

    @staticmethod
    def across_join(prev: np.ndarray, s: np.ndarray, joins: np.ndarray | None) -> pd.arrays.BooleanArray:
        """Whether a join lies in (prev, s]; NA where there is no previous event or joins are unknown."""
        has = prev >= 0
        if joins is None:
            return pd.array([pd.NA] * len(prev), dtype="boolean")
        out = pd.array(joins_between(joins, np.where(has, prev, 0), s) > 0, dtype="boolean")
        out[~has] = pd.NA
        return out

    def _add_sequence_covariates(self, df: pd.DataFrame, ctx: BuildContext) -> None:
        s = ctx.trial_samples
        joins = ctx.joins

        def lag(prev: np.ndarray) -> tuple[np.ndarray, pd.arrays.BooleanArray, np.ndarray]:
            concat = np.where(prev >= 0, ctx.ms(s - prev), np.nan)
            cross = self.across_join(prev, s, joins)
            clean = np.where(cross.fillna(False).to_numpy(bool), np.nan, concat)
            return concat, cross, clean

        prev_stim = last_before(s, ctx.stim_samples)
        c, x, v = lag(prev_stim)
        df["soa_prev_ms"], df["soa_prev_ms_concat"], df["soa_prev_crosses_join"] = v, c, x
        df["soa_stratum"] = pd.array(soa_strata(v, self.cfg.soa_strata_ms), dtype="string")
        prev_trial = np.r_[-1, s[:-1]] if s.size else np.array([], np.int64)
        c, x, v = lag(prev_trial)
        df["t_since_prev_trial_ms"], df["t_since_prev_trial_ms_concat"], df["prev_trial_crosses_join"] = v, c, x
        prev_press = last_before(s, ctx.press_samples)
        c, x, v = lag(prev_press)
        df["t_since_prev_press_ms"], df["t_since_prev_press_ms_concat"], df["prev_press_crosses_join"] = v, c, x
        ctx.extra["prev_press_sample"] = prev_press

        # previous target outcome / RT (targets only, within the run)
        n = s.size
        tgt = df["is_target"].to_numpy(bool)
        resp = df["responded"].to_numpy(bool)
        rt = df["rt_ms"].to_numpy(float)
        prev_out = np.full(n, "none", dtype=object)
        prev_rt = np.full(n, np.nan)
        prev_tgt_sample = np.full(n, -1, np.int64)
        prev_tgt_resp = np.full(n, -1, np.int64)
        rs = df["response_sample"].fillna(-1).to_numpy(np.int64)
        last = -1
        for k in range(n):
            if last >= 0:
                prev_out[k] = "hit" if resp[last] else "miss"
                prev_rt[k] = rt[last]
                prev_tgt_sample[k] = s[last]
                prev_tgt_resp[k] = rs[last]
            if tgt[k]:
                last = k
        df["prev_outcome"] = pd.array(prev_out, dtype="string")
        df["prev_rt_ms"] = prev_rt
        df["prev_target_crosses_join"] = self.across_join(prev_tgt_sample, s, joins)
        ctx.extra["prev_target_response_sample"] = prev_tgt_resp
        ctx.extra["prev_target_sample"] = prev_tgt_sample
        npress = df["n_press_until_bound"].to_numpy()
        ptnp = pd.array([pd.NA] + [bool(x == 0) for x in npress[:-1]], dtype="boolean") if n else pd.array([], dtype="boolean")
        df["prev_trial_no_press"] = ptnp
        df["time_in_run_s"] = (s - s[0]) / ctx.sfreq if n else np.array([], float)
        df["first_in_run"] = np.arange(n) == 0
        df["last_in_run"] = np.arange(n) == n - 1

    def marker_samples(self, ctx: BuildContext) -> np.ndarray:
        """Sorted unique samples of every structure marker (``marker_codes``).

        Status markers (which include the events.tsv marker rows) when Status
        was read, else the events.tsv rows with a marker code only.
        """
        codes = list(self.cfg.marker_codes)
        if ctx.markers is not None:
            m = ctx.markers.loc[ctx.markers["value"].isin(codes), "sample"].to_numpy(np.int64)
        else:
            m = ctx.events.loc[ctx.events["value"].isin(codes), "sample"].to_numpy(np.int64)
        return np.unique(m).astype(np.int64)

    def _add_contamination(self, df: pd.DataFrame, ctx: BuildContext) -> None:
        s = ctx.trial_samples
        ps = last_before(s, ctx.press_samples)
        ss = last_before(s, ctx.stim_samples)
        mk = last_before(s, self.marker_samples(ctx))
        lag_p = np.where(ps >= 0, ctx.ms(s - ps), np.inf)
        lag_s = np.where(ss >= 0, ctx.ms(s - ss), np.inf)
        lag_m = np.where(mk >= 0, ctx.ms(s - mk), np.inf)
        for L in self.cfg.lookbacks_ms:
            pin = lag_p <= L + 1e-9
            sin = lag_s <= L + 1e-9
            min_ = lag_m <= L + 1e-9
            df[f"press_in_lookback_{L}"] = pin
            df[f"stim_in_lookback_{L}"] = sin
            df[f"clean_lookback_{L}"] = ~(pin | sin)
            df[f"marker_in_lookback_{L}"] = min_
            df[f"clean_all_lookback_{L}"] = ~(pin | sin | min_)

    def block_start_samples(self, ctx: BuildContext) -> np.ndarray | None:
        """Samples that start a block (default: markers with ``block_start_codes``)."""
        if ctx.markers is None:
            return None
        m = ctx.markers
        return as_sorted_samples(m.loc[m["value"].isin(list(self.cfg.block_start_codes)), "sample"].to_numpy())

    def _add_blocks(self, df: pd.DataFrame, ctx: BuildContext) -> None:
        s = ctx.trial_samples
        n = s.size
        starts = self.block_start_samples(ctx)
        ctx.extra["block_starts"] = starts
        df["markers_available"] = np.full(n, starts is not None, dtype=bool)
        if starts is None or n == 0:
            df["block_index_in_run"] = pd.array([pd.NA] * n, dtype="Int64")
            df["trial_in_block"] = pd.array([pd.NA] * n, dtype="Int64")
            df["block_start_unmarked"] = pd.array([pd.NA] * n, dtype="boolean")
            df["t_since_block_start_s"] = np.full(n, np.nan)
            df["first_in_block"] = df["first_in_run"].to_numpy(bool) if n else np.array([], bool)
            return
        k = np.searchsorted(starts, s, side="right")
        # one convention for every run and every table: block = 1 + block-start markers at or before onset
        # (so block 1 is always the segment before the first marker, empty when the run starts with a marker)
        block = k + 1
        df["block_index_in_run"] = pd.array(block, dtype="Int64")
        df["trial_in_block"] = pd.array(pd.Series(block).groupby(block).cumcount().to_numpy() + 1, dtype="Int64")
        df["block_start_unmarked"] = pd.array(k == 0, dtype="boolean")
        last = starts[np.maximum(k - 1, 0)] if starts.size else np.zeros(n, np.int64)
        df["t_since_block_start_s"] = np.where(k > 0, (s - last) / ctx.sfreq, np.nan)
        df["first_in_block"] = np.r_[True, block[1:] != block[:-1]]

    def _add_joins(self, df: pd.DataFrame, ctx: BuildContext) -> None:
        s = ctx.trial_samples
        n = s.size
        j = ctx.joins
        df["joins_available"] = np.full(n, j is not None, dtype=bool)
        if j is None:
            df["n_joins_before"] = pd.array([pd.NA] * n, dtype="Int64")
            df["t_since_last_join_s"] = np.full(n, np.nan)
            df["t_to_next_join_s"] = np.full(n, np.nan)
            df["join_near_window"] = np.zeros(n, bool)
            return
        nb = np.searchsorted(j, s, side="right")
        df["n_joins_before"] = pd.array(nb, dtype="Int64")
        lastj = np.where(nb > 0, j[np.maximum(nb - 1, 0)] if j.size else 0, -1)
        df["t_since_last_join_s"] = np.where(lastj >= 0, (s - lastj) / ctx.sfreq, np.nan)
        nxt = next_after(s, j)
        df["t_to_next_join_s"] = np.where(nxt >= 0, (nxt - s) / ctx.sfreq, np.nan)
        pre = int(np.ceil((self.cfg.join_lookback_ms + self.cfg.join_padding_ms) * ctx.sfreq / 1000.0))
        post = int(np.floor(self.cfg.join_post_ms * ctx.sfreq / 1000.0))
        lo = np.searchsorted(j, s - pre, side="left")
        hi = np.searchsorted(j, s + post, side="right")
        df["join_near_window"] = (hi - lo) > 0

    def _add_history_flags(self, df: pd.DataFrame, ctx: BuildContext) -> None:
        h = self.cfg.history
        po = df["prev_outcome"].to_numpy(dtype=object)
        ok = (po != "none") if h.prev_outcome == "any" else (po == h.prev_outcome)
        lagp = df["t_since_prev_press_ms"].to_numpy(float)
        ok &= np.isfinite(lagp) & (lagp >= h.press_lag_min_ms - 1e-9) & (lagp <= h.press_lag_max_ms + 1e-9)
        if h.last_press_is_prev_response:
            ok &= ctx.extra["prev_press_sample"] == ctx.extra["prev_target_response_sample"]
            ok &= ctx.extra["prev_target_response_sample"] >= 0
        if h.no_join_since_prev_target:
            ok &= ~df["prev_target_crosses_join"].fillna(True).to_numpy(bool)
        df["history_homogeneous"] = ok.astype(bool)
        thr = self.cfg.misattribution_fast_rt_ms
        flag = (df["responded"].to_numpy(bool) & (df["rt_ms"].to_numpy(float) < thr)
                & df["prev_trial_no_press"].fillna(False).to_numpy(bool))
        # misattribution (a late press to the previous stimulus landing in this window) needs a previous trial
        # row that expects a response and is close in time: AVSRT only (misattribution.applies); NA elsewhere
        df["flag_rt_fast_after_no_press"] = nullable_bool(flag, np.full(len(df), self.cfg.misattribution_applies))

        # previous outcome refers to the immediately preceding target of the same block, no join in between
        s = ctx.trial_samples
        pts = ctx.extra["prev_target_sample"]
        has = pts >= 0
        same_block = np.ones(len(df), bool)
        blk = df["block_index_in_run"]
        if len(df) and blk.notna().all():
            b = blk.to_numpy(np.int64)
            j = np.clip(np.searchsorted(s, pts, side="left"), 0, max(len(df) - 1, 0))
            same_block = b[j] == b
        no_join = ~df["prev_target_crosses_join"].fillna(True).to_numpy(bool)
        df["prev_outcome_valid"] = has & same_block & no_join

    def _add_integrity(self, df: pd.DataFrame, rec: RecordingEvents) -> None:
        """Row-level confirmation of each stimulus by Status and by the recording length.

        ``in_status``: the (sample, logged code) pair is a Status onset (NA when
        Status was not read). ``beyond_recording``: the onset is at or after the
        last sample (NA when the length is unknown). ``post_stim_truncated``:
        less than :attr:`TaskEventsConfig.post_stim_required_s` of data after
        onset (False when the length is unknown). ``stim_unverified``: Status was
        read and does not confirm the stimulus, or it lies beyond the recording.
        Every analysis subset requires ``stim_unverified`` and
        ``post_stim_truncated`` to be False.
        """
        n = len(df)
        s = df["sample"].to_numpy(np.int64)
        if rec.crosscheck is not None:
            only = rec.crosscheck.tsv_only
            miss = pd.MultiIndex.from_frame(only[["sample", "value"]].astype("int64")) if len(only) else None
            key = pd.MultiIndex.from_arrays([s, df["value_original"].to_numpy(np.int64)])
            in_st = ~key.isin(miss) if miss is not None else np.ones(n, bool)
            df["in_status"] = pd.array(np.asarray(in_st, bool), dtype="boolean")
        else:
            df["in_status"] = pd.array([pd.NA] * n, dtype="boolean")
        if rec.n_times is not None:
            post = (int(rec.n_times) - s) / rec.sfreq
            df["t_to_recording_end_s"] = post.astype(float)
            df["beyond_recording"] = pd.array(s >= int(rec.n_times), dtype="boolean")
            df["post_stim_truncated"] = post < self.cfg.post_stim_required_s - 1e-12
        else:
            df["t_to_recording_end_s"] = np.full(n, np.nan)
            df["beyond_recording"] = pd.array([pd.NA] * n, dtype="boolean")
            df["post_stim_truncated"] = np.zeros(n, bool)
        df["stim_unverified"] = ((~df["in_status"].fillna(True).to_numpy(bool))
                                 | df["beyond_recording"].fillna(False).to_numpy(bool))
        if df["stim_unverified"].any():
            log.warning("%s %s run-%02d: %d stimulus row(s) not confirmed by Status or beyond the recording",
                        rec.subject, rec.task, rec.run, int(df["stim_unverified"].sum()))

    def _add_press_clear(self, df: pd.DataFrame, ctx: BuildContext) -> None:
        """``press_clear_<X>``: no press in ``[onset - X ms, onset)`` (concatenated clock; S-PRESS)."""
        s = ctx.trial_samples
        ps = last_before(s, ctx.press_samples)
        lag = np.where(ps >= 0, ctx.ms(s - ps), np.inf)
        for X in self.cfg.press_clear_ms:
            df[f"press_clear_{X}"] = lag > X + 1e-9

    def _add_edge_validity(self, df: pd.DataFrame, ctx: BuildContext, n_times: int | None) -> None:
        """``edge_valid_<window>`` for every configured prestimulus window.

        Window samples are ``[onset + round(start x sfreq / 1000), onset + round(stop x sfreq / 1000))``.
        The recording is split at the bit-16 joins into segments ``[0, j1), [j1, j2), ..., [jk, n_times)``
        (a join is the first sample of a new segment). The window is valid when it lies in one segment with at
        least ``round(prestim_edge_margin_s x sfreq)`` samples between each window edge and the segment's
        edges, i.e. no join and no file edge within the FIR half-length of either window edge (the rule of
        :meth:`preprocessing.segments.SegmentInfo.window_valid` without bad spans). NA when joins or
        the recording length are unknown.
        """
        s = ctx.trial_samples
        n = s.size
        wins = self.cfg.prestim_windows
        if not wins:
            return
        if ctx.joins is None or n_times is None:
            for wn in wins:
                df[f"edge_valid_{wn}"] = pd.array([pd.NA] * n, dtype="boolean")
            return
        df_margin = int(round(self.cfg.prestim_edge_margin_s * ctx.sfreq))
        j = ctx.joins[(ctx.joins > 0) & (ctx.joins < int(n_times))]
        bounds = np.r_[0, j, int(n_times)].astype(np.int64)
        for wn, (a_ms, b_ms) in wins.items():
            a = s + int(round(a_ms * ctx.sfreq / 1000.0))
            b = s + int(round(b_ms * ctx.sfreq / 1000.0))
            k = np.searchsorted(bounds, a, side="right") - 1          # segment [bounds[k], bounds[k + 1])
            inside = (k >= 0) & (k < bounds.size - 1)
            kk = np.clip(k, 0, bounds.size - 2)
            ok = inside & (a - bounds[kk] >= df_margin) & (bounds[kk + 1] - b >= df_margin) & (b > a)
            df[f"edge_valid_{wn}"] = pd.array(ok, dtype="boolean")

    def add_known_issue_flags(self, df: pd.DataFrame, ctx: BuildContext, issues: list,
                              recode_ids: dict[int, list[str]]) -> None:
        """Set ``excluded_known_issue``, ``known_issue_ids`` and ``known_issue_flags`` on ``df``.

        ``df`` needs ``sample``, ``onset_s`` and ``trial_key`` (any row table:
        trials, contours, presses).
        """
        n = len(df)
        excl = np.zeros(n, bool)
        ids: list[set[str]] = [set() for _ in range(n)]
        flags: list[set[str]] = [set() for _ in range(n)]
        s = df["sample"].to_numpy(np.int64)
        on = df["onset_s"].to_numpy(float)
        for iss in issues:
            for a in iss.actions:
                if a.type not in ROW_ACTIONS:
                    continue
                hit = np.zeros(n, bool)
                if a.type == "exclude_recording":
                    if "trials" in a.scopes:
                        hit[:] = True
                        excl |= hit
                elif a.type == "exclude_sample_range":
                    lo = a.get("start_sample")
                    hi = a.get("end_sample")
                    hit = (s >= (lo if lo is not None else -1)) & (s <= (hi if hi is not None else np.iinfo(np.int64).max))
                    excl |= hit
                elif a.type == "exclude_time_range":
                    lo = a.get("start_s")
                    hi = a.get("end_s")
                    hit = (on >= (lo if lo is not None else -np.inf)) & (on <= (hi if hi is not None else np.inf))
                    excl |= hit
                elif a.type == "exclude_events_before_first_marker":
                    first = self.first_marker_sample(ctx, a.get("marker_codes"))
                    if first is None:
                        log.warning("%s: markers unavailable; flagging every row of %s", iss.id,
                                    df["trial_key"].iloc[0] if n else "")
                        hit[:] = True
                    else:
                        hit = s < first
                    excl |= hit
                elif a.type == "flag_only":
                    hit[:] = True
                    for k in np.flatnonzero(hit):
                        flags[k].add(str(a.get("flag")))
                elif a.type == "exclude_partial_blocks":
                    hit[:] = True  # applied by the task builder; recorded here for traceability
                for k in np.flatnonzero(hit):
                    ids[k].add(iss.id)
        for smp, iids in recode_ids.items():
            for k in np.flatnonzero(s == smp):
                ids[k].update(iids)
        df["excluded_known_issue"] = excl
        df["known_issue_ids"] = pd.array([";".join(sorted(x)) for x in ids], dtype="string")
        df["known_issue_flags"] = pd.array([";".join(sorted(x)) for x in flags], dtype="string")


# --------------------------------------------------------------------------- child level
def add_child_level_indices(trials: pd.DataFrame) -> pd.DataFrame:
    """Add per-child indices across runs.

    ``trial_index_in_child``: 1-based over runs sorted by run then sample.
    ``block_index_in_child``: ``block_index_in_run`` plus the highest
    ``block_index_in_run`` of each earlier run of the child (every run keeps
    its block-1 slot, empty when the run starts with a marker); NA where blocks
    are unavailable.
    """
    df = trials.sort_values(["subject", "task", "run", "sample"]).copy()
    df["trial_index_in_child"] = (df.groupby(["subject", "task"]).cumcount() + 1).astype("int64")
    bmax = (df.groupby(["subject", "task", "run"])["block_index_in_run"].max().fillna(0).astype("int64")
            .rename("bmax").reset_index())
    bmax["block_offset"] = bmax.groupby(["subject", "task"])["bmax"].transform(
        lambda x: x.shift(1, fill_value=0).cumsum()).astype("int64")
    off = df[["subject", "task", "run"]].merge(bmax[["subject", "task", "run", "block_offset"]],
                                                on=["subject", "task", "run"], how="left")["block_offset"]
    df["block_index_in_child"] = (df["block_index_in_run"] + off.to_numpy()).astype("Int64")
    return df


def empty_frame(schema: TableSchema) -> pd.DataFrame:
    """Zero-row frame with the dtypes of ``schema``."""
    dt = {"int": "int64", "Int": "Int64", "float": "float64", "bool": "bool", "boolean": "boolean", "str": "string"}
    return pd.DataFrame({c.name: pd.Series(dtype=dt[c.dtype]) for c in schema.columns})


def soa_strata(soa_ms: np.ndarray, edges: tuple[float, ...] | list[float]) -> np.ndarray:
    """Label each SOA with its stratum (labels from :func:`~trials.schema.soa_stratum_labels`).

    NA where the SOA is NA or no edges are configured; ``<lo`` / ``>=hi``
    outside the configured range; ``[a, b)`` intervals inside it.
    """
    x = np.asarray(soa_ms, dtype=float)
    out = np.full(x.size, None, dtype=object)
    if not len(edges):
        return out
    e = np.asarray(edges, float)
    labels = np.asarray(soa_stratum_labels(edges), dtype=object)
    ok = np.isfinite(x)
    out[ok] = labels[np.searchsorted(e, x[ok], side="right")]   # 0: below the first edge; len(e): at/above the last
    return out


def subset_mask(trials: pd.DataFrame, task_cfg: TaskEventsConfig, name: str) -> pd.Series:
    """Boolean mask of a named analysis subset from ``events.yaml``."""
    if name not in task_cfg.analysis_subsets:
        raise KeyError(f"subset {name!r} not configured for {task_cfg.name}")
    return apply_subset(trials, task_cfg.analysis_subsets[name])


def concat_recordings(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-recording trial tables, keeping dtypes."""
    tabs = [t for t in tables if len(t)]
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
