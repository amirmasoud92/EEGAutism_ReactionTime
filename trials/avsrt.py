"""AVSRT trial builder: audiovisual simple reaction time (codes 3 AV, 4 A, 5 V; press 1).

:class:`AvsrtTrialBuilder` adds the AVSRT-specific columns to the shared trial
table of :class:`~trials.base.TaskTrialBuilder`:

Sequence covariates (full stimulus sequence of the run, misses included)
    ``modality``, ``prev_modality``, ``transition``, ``switch_type`` (repeat /
    modality_switch (A<->V) / to_AV / from_AV / none) and the coarse
    ``switch_type_coarse`` (repeat / partial / switch / none).
    ``prev_stim_in_segment`` says whether the previous logged stimulus is a
    trustworthy predecessor: same Status block, no gap > ``post_gap_s`` (no
    pause, no unlogged stimulus) and no bit-16 join in between.
Gap classification (``classify_gaps``)
    Within-run SOAs above ``post_gap_s`` are classed as block boundaries,
    pauses / restarts or dropped stimulus triggers. Outputs
    ``gap_class``, ``est_trial_index``, ``nominal_block``, ``n_missing_before``,
    ``in_aborted_segment``, ``prestim_has_unlogged_stim``,
    ``next_is_dropped_trigger`` and ``post_gap``.
Low-press-rate segments
    Segment = Status block (gap-based nominal block if Status was not read or
    holds no block-start marker) x
    ``in_aborted_segment``; ``low_press_rate_segment`` when the fraction of
    trials with any press before the next stimulus is below ``low_press_rate``.
Response attribution
    ``rt_lt150``, ``rt_lt200``, ``anticipation_before_kept_press``,
    ``n_valid_window_presses``, ``flag_rt_lt150_after_no_press`` and
    ``flag_rt_fast_after_prev_miss`` (the base adds
    ``flag_rt_fast_after_no_press``).
Known-issue recoding
    ``code_recoded`` marks rows whose code was changed by a ``recode_event``
    action (e.g. an OR-collision 1|4 -> 5 recoded to A).

Child level (base :meth:`~trials.base.TaskTrialBuilder.assemble_children`)
    Runs are built separately, so SOA, previous modality and every other
    interval stay within a run, then concatenated with the base child-level
    indices. :meth:`AvsrtTrialBuilder.add_child_columns` adds
    ``in_repeated_block_run``, which marks the surplus run(s) of a child carrying
    the ``repeated_block_run`` known-issue flag: every run except the one with
    the most stimuli (flag only; the runs are kept).

Parameters live in ``configs/events.yaml`` under ``tasks.AVSRT.avsrt``;
missing keys fall back to the defaults in :class:`AvsrtParams`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from trials.base import (
    BuildContext,
    RecordingEvents,
    TaskTrialBuilder,
    add_child_level_indices,
    concat_recordings,
    empty_frame,
    load_recording_events,
)
from trials.schema import CHILD_BASE_COLUMNS, ColumnSpec, TableSchema
from utils.bids_index import Recording
from utils.config import ConfigError, StatusConfig, TaskEventsConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)

#: Gap classes.
GAP_CLASSES = ("none", "block_boundary", "block_boundary_partial_prev", "pause_restart", "dropped_trigger")
SWITCH_TYPES = ("none", "repeat", "modality_switch", "to_AV", "from_AV")
SWITCH_TYPES_COARSE = ("none", "repeat", "partial", "switch")
#: Known-issue ``flag_only`` label that marks a child with a surplus (repeated) block run.
REPEATED_BLOCK_FLAG = "repeated_block_run"


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class AvsrtParams:
    """AVSRT builder parameters.

    Attributes
    ----------
    post_gap_s
        SOA above which a gap is not a within-block SOA (observed within-block
        maximum 3.184 s).
    pause_s
        Gaps longer than this are pauses / restarts, never dropped triggers.
    mean_soa_s
        Mean within-block SOA, used to estimate how many stimuli a gap hides.
    nominal_block_size
        Stimuli per nominal block (100).
    boundary_tol
        A gap whose running nominal count is within this many trials of a
        multiple of ``nominal_block_size`` is a block boundary.
    partial_first_min_next
        A gap after the first segment of a run is ``block_boundary_partial_prev``
        (recording started late) if the next segment has at least this many
        logged stimuli.
    low_press_rate
        Segments whose any-press rate is below this are flagged.
    rt_strict_ms
        Threshold of the stricter response-attribution flags (``rt_lt150``).
    """

    post_gap_s: float = 3.2
    pause_s: float = 8.0
    mean_soa_s: float = 2.166
    nominal_block_size: int = 100
    boundary_tol: int = 2
    partial_first_min_next: int = 98
    low_press_rate: float = 0.2
    rt_strict_ms: float = 150.0

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any] | None, where: str = "tasks.AVSRT.avsrt") -> AvsrtParams:
        """Validate a mapping (unknown keys raise :class:`ConfigError`)."""
        d = dict(d or {})
        known = {f.name: f for f in fields(cls)}
        bad = set(d) - set(known)
        if bad:
            raise ConfigError(f"{where}: unknown keys {sorted(bad)}")
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"{where}.{k}: expected a number, got {v!r}")
            out[k] = int(v) if known[k].type in ("int", int) else float(v)
        p = cls(**out)
        if not (0 < p.mean_soa_s < p.post_gap_s < p.pause_s):
            raise ConfigError(f"{where}: need 0 < mean_soa_s < post_gap_s < pause_s")
        if p.nominal_block_size < 1 or not 0 <= p.boundary_tol < p.nominal_block_size / 2:
            raise ConfigError(f"{where}: invalid nominal_block_size / boundary_tol")
        if not 0 <= p.low_press_rate <= 1:
            raise ConfigError(f"{where}.low_press_rate must be in [0, 1]")
        return p

    @classmethod
    def from_task_cfg(cls, cfg: TaskEventsConfig) -> AvsrtParams:
        return cls.from_mapping(cfg.extra.get("avsrt"))


# --------------------------------------------------------------------------- pure helpers
def classify_gaps(soa_prev_s: np.ndarray, params: AvsrtParams = AvsrtParams()) -> dict[str, np.ndarray]:
    """Classify within-run gaps into block boundaries, pauses / restarts and dropped triggers.

    Parameters
    ----------
    soa_prev_s
        Per stimulus of one run (in order), the SOA to the previous stimulus
        in seconds on the recorded clock (NaN for the first).
    params
        Thresholds.

    Returns
    -------
    dict
        ``gap_class`` (object), ``est_trial_index`` (int, presumed nominal
        1-based position in the run, counting inferred missing stimuli),
        ``nominal_block`` (int), ``in_aborted_segment`` (bool),
        ``n_missing_before`` (int).

    Notes
    -----
    At each gap > ``post_gap_s`` (est = nominal count so far, r = est mod B):

    1. r within ``boundary_tol`` of 0 or B: ``block_boundary``; est rounds
       to the nearest multiple of B.
    2. else gap > ``pause_s``: ``pause_restart``; the preceding segment is
       aborted and est floors to a multiple of B.
    3. else, first segment of the run and the next segment holds at least
       ``partial_first_min_next`` stimuli: ``block_boundary_partial_prev``;
       est ceils to a multiple of B.
    4. else ``dropped_trigger`` with ``max(1, round(gap / mean_soa) - 1)``
       missing stimuli; est advances by that number.
    """
    isi = np.asarray(soa_prev_s, dtype=float)
    n = isi.size
    B = params.nominal_block_size
    gap_idx = [i for i in range(1, n) if isi[i] > params.post_gap_s]
    bounds = [0, *gap_idx, n]
    seg_len = {bounds[k]: bounds[k + 1] - bounds[k] for k in range(len(bounds) - 1)}
    gap_class = np.array(["none"] * n, dtype=object)
    est_idx = np.zeros(n, np.int64)
    aborted = np.zeros(n, bool)
    n_missing = np.zeros(n, np.int64)
    est = 0
    seg_start = 0
    for i in range(n):
        if i > 0 and i in seg_len:
            r = est % B
            if r <= params.boundary_tol or (B - r) <= params.boundary_tol:
                gap_class[i] = "block_boundary"
                est = int(round(est / B) * B)
            elif isi[i] > params.pause_s:
                gap_class[i] = "pause_restart"
                aborted[seg_start:i] = True
                est = int(np.floor(est / B) * B)
            elif seg_len[i] >= params.partial_first_min_next and seg_start == 0:
                gap_class[i] = "block_boundary_partial_prev"
                est = int(np.ceil(est / B) * B)
            else:
                k = max(1, int(round(isi[i] / params.mean_soa_s)) - 1)
                gap_class[i] = "dropped_trigger"
                n_missing[i] = k
                est += k
            seg_start = i
        est += 1
        est_idx[i] = est
    return {"gap_class": gap_class, "est_trial_index": est_idx, "nominal_block": (est_idx - 1) // B + 1,
            "in_aborted_segment": aborted, "n_missing_before": n_missing}


def switch_types(cur: Sequence[str], prev: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
    """Fine and coarse switch type of each trial from current and previous modality.

    Fine: ``repeat`` (same modality), ``modality_switch`` (A<->V), ``to_AV``
    (A or V -> AV), ``from_AV`` (AV -> A or V), ``none`` (no previous).
    Coarse: ``repeat``, ``partial`` (AV involved), ``switch`` (A<->V), ``none``.
    """
    fine = np.empty(len(cur), dtype=object)
    coarse = np.empty(len(cur), dtype=object)
    for k, (c, p) in enumerate(zip(cur, prev)):
        if p is None or (isinstance(p, float) and np.isnan(p)) or p is pd.NA:
            fine[k] = coarse[k] = "none"
        elif c == p:
            fine[k] = coarse[k] = "repeat"
        elif c == "AV":
            fine[k], coarse[k] = "to_AV", "partial"
        elif p == "AV":
            fine[k], coarse[k] = "from_AV", "partial"
        else:
            fine[k], coarse[k] = "modality_switch", "switch"
    return fine, coarse


def _c(name: str, dtype: str, nullable: bool = False, description: str = "") -> ColumnSpec:
    return ColumnSpec(name, dtype, nullable, description)


AVSRT_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("modality", "str", description="AV / A / V of this stimulus (after known-issue recoding)"),
    _c("code_recoded", "bool", description="code changed by a recode_event known issue (value != value_original)"),
    _c("prev_modality", "str", True, "modality of the previous logged stimulus in the run (misses included)"),
    _c("transition", "str", True, "prev_modality->modality"),
    _c("switch_type", "str", description="repeat / modality_switch (A<->V) / to_AV / from_AV / none"),
    _c("switch_type_coarse", "str", description="repeat / partial / switch / none"),
    _c("prev_stim_in_segment", "bool",
       description="previous stimulus in the same Status block, SOA <= post_gap_s, no join in between"),
    _c("first_after_block_marker", "bool", description="first stimulus after a 201 marker"),
    _c("post_gap", "bool", description="SOA > post_gap_s on the recorded clock, or a join intervenes"),
    _c("gap_class", "str", description="class of the gap before this stimulus (classify_gaps)"),
    _c("est_trial_index", "int", description="presumed nominal position in run incl. inferred dropped stimuli"),
    _c("nominal_block", "int", description="(est_trial_index - 1) // nominal_block_size + 1"),
    _c("n_missing_before", "int", description="inferred unlogged stimuli in the gap before this stimulus"),
    _c("prestim_has_unlogged_stim", "bool", description="gap before this stimulus is a dropped trigger"),
    _c("next_is_dropped_trigger", "bool",
       description="gap after this stimulus is a dropped trigger (its window may hold the unlogged stimulus' press)"),
    _c("in_aborted_segment", "bool", description="segment before a pause/restart gap"),
    _c("segment_index_in_run", "int", description="1-based press-rate segment (Status block x aborted flag)"),
    _c("segment_any_press_rate", "float", description="fraction of the segment's trials with any press before bound"),
    _c("low_press_rate_segment", "bool", description="segment_any_press_rate < low_press_rate"),
    _c("rt_lt150", "bool", description="responded with RT < rt_strict_ms"),
    _c("rt_lt200", "bool", description="responded with RT < misattribution threshold"),
    _c("anticipation_before_kept_press", "bool", description="responded and a press < primary lo precedes it"),
    _c("n_valid_window_presses", "int", description="presses in [onset, bound) with latency in the primary window"),
    _c("flag_rt_lt150_after_no_press", "bool", description="RT < rt_strict_ms after a no-press trial"),
    _c("flag_rt_fast_after_prev_miss", "bool",
       description="RT < misattribution threshold after a previous target without a primary-window response"),
)

AVSRT_CHILD_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("repeated_block_subject", "bool", description="child has the repeated_block_run known-issue flag"),
    _c("in_repeated_block_run", "bool",
       description="surplus run of a repeated_block_run child (all runs but the one with most stimuli)"),
)
#: Base + AVSRT child-level columns.
CHILD_COLUMNS: tuple[ColumnSpec, ...] = CHILD_BASE_COLUMNS + AVSRT_CHILD_COLUMNS


# --------------------------------------------------------------------------- builder
class AvsrtTrialBuilder(TaskTrialBuilder):
    """Trial-table builder for AVSRT.

    Parameters
    ----------
    task_cfg, status_cfg, registry
        As for :class:`~trials.base.TaskTrialBuilder`.
    params
        Overrides the ``tasks.AVSRT.avsrt`` config block.
    """

    task: ClassVar[str] = "AVSRT"

    def __init__(self, task_cfg: TaskEventsConfig, status_cfg: StatusConfig,
                 registry: KnownIssuesRegistry | None = None, params: AvsrtParams | None = None) -> None:
        super().__init__(task_cfg, status_cfg, registry)
        self.params = params if params is not None else AvsrtParams.from_task_cfg(task_cfg)
        labels = set(task_cfg.stimulus_codes.values())
        if not {"AV", "A", "V"} <= labels:
            raise ConfigError(f"AVSRT stimulus labels must include AV, A, V; got {sorted(labels)}")

    def extra_columns(self) -> tuple[ColumnSpec, ...]:
        return AVSRT_COLUMNS

    def child_extra_columns(self) -> tuple[ColumnSpec, ...]:
        return AVSRT_CHILD_COLUMNS

    def add_child_columns(self, df: pd.DataFrame, run_table: pd.DataFrame | None = None) -> pd.DataFrame:
        """Add ``repeated_block_subject`` and ``in_repeated_block_run``."""
        return add_repeated_block_columns(df)

    # ------------------------------------------------------------------ hook
    def add_task_columns(self, trials: pd.DataFrame, rec: RecordingEvents, ctx: BuildContext) -> pd.DataFrame:
        df = trials
        p = self.params
        n = len(df)
        s = ctx.trial_samples
        lab = self.cfg.stimulus_codes
        modality = df["condition"].astype(object).to_numpy()
        df["modality"] = pd.array(modality, dtype="string")
        df["code_recoded"] = df["value"].to_numpy() != df["value_original"].to_numpy()

        # previous logged stimulus of the run (full sequence, misses and excluded rows included)
        ip = np.searchsorted(ctx.stim_samples, s, side="left") - 1
        prev = np.array([lab.get(int(ctx.stim_values[i]), str(int(ctx.stim_values[i]))) if i >= 0 else None
                         for i in ip], dtype=object)
        df["prev_modality"] = pd.array(prev, dtype="string")
        df["transition"] = pd.array([f"{a}->{b}" if a is not None else None for a, b in zip(prev, modality)],
                                    dtype="string")
        fine, coarse = switch_types(list(modality), list(prev))
        df["switch_type"] = pd.array(fine, dtype="string")
        df["switch_type_coarse"] = pd.array(coarse, dtype="string")

        # gaps (recorded clock), dropped triggers, aborted segments
        soa_c = df["soa_prev_ms_concat"].to_numpy(float)
        g = classify_gaps(soa_c / 1000.0, p)
        crosses = df["soa_prev_crosses_join"].fillna(False).to_numpy(bool)
        post_gap = (np.nan_to_num(soa_c, nan=-np.inf) > p.post_gap_s * 1000.0 + 1e-9) | crosses
        df["post_gap"] = post_gap
        for k in ("gap_class", "est_trial_index", "nominal_block", "n_missing_before", "in_aborted_segment"):
            df[k] = g[k]
        dropped = g["gap_class"] == "dropped_trigger"
        df["prestim_has_unlogged_stim"] = dropped
        df["next_is_dropped_trigger"] = np.r_[dropped[1:], False] if n else np.array([], bool)

        # Status blocks: first stimulus after a marker; previous stimulus in the same segment
        blk = df["block_index_in_run"]
        starts = ctx.extra.get("block_starts")
        # Status blocks exist only if Status was read AND holds at least one block-start marker; a run without any
        # 201 falls back to the gap-based nominal blocks for the press-rate segments
        has_blocks = bool(n) and starts is not None and np.asarray(starts).size > 0 and blk.notna().all()
        if has_blocks:
            b = blk.to_numpy(np.int64)
            unmarked = df["block_start_unmarked"].fillna(True).to_numpy(bool)
            first_mark = df["first_in_block"].to_numpy(bool) & ~unmarked
            same_block = np.r_[False, b[1:] == b[:-1]] if n else np.array([], bool)
        else:
            first_mark = np.zeros(n, bool)
            same_block = np.r_[False, np.ones(max(n - 1, 0), bool)] if n else np.array([], bool)
        df["first_after_block_marker"] = first_mark
        # the previous *trial* row is the previous stimulus in AVSRT (trial codes = stimulus codes)
        df["prev_stim_in_segment"] = (ip >= 0) & same_block & ~post_gap
        # every AVSRT stimulus is a target: the previous outcome is valid when the previous stimulus is in-segment
        df["prev_outcome_valid"] = df["prev_stim_in_segment"].to_numpy(bool)

        # press-rate segments: Status block (else nominal block) x aborted flag
        seg_block = b if has_blocks else g["nominal_block"]
        key = np.stack([seg_block, g["in_aborted_segment"].astype(np.int64)], axis=1) if n else np.zeros((0, 2))
        change = np.r_[True, np.any(key[1:] != key[:-1], axis=1)] if n else np.array([], bool)
        seg = np.cumsum(change).astype(np.int64)
        anyp = (df["n_press_until_bound"].to_numpy() > 0).astype(float)
        rate = pd.Series(anyp).groupby(seg).transform("mean").to_numpy(float) if n else np.array([], float)
        df["segment_index_in_run"] = seg
        df["segment_any_press_rate"] = rate
        df["low_press_rate_segment"] = rate < p.low_press_rate - 1e-12

        # response attribution (DV7)
        resp = df["responded"].to_numpy(bool)
        rt = df["rt_ms"].to_numpy(float)
        thr = self.cfg.misattribution_fast_rt_ms
        df["rt_lt150"] = resp & (rt < p.rt_strict_ms)
        df["rt_lt200"] = resp & (rt < thr)
        df["anticipation_before_kept_press"] = resp & (df["n_anticipations"].to_numpy() > 0)
        df["n_valid_window_presses"] = (df["n_press_until_bound"].to_numpy(np.int64)
                                        - df["n_anticipations"].to_numpy(np.int64)
                                        - df["n_late_presses"].to_numpy(np.int64))
        no_press = df["prev_trial_no_press"].fillna(False).to_numpy(bool)
        df["flag_rt_lt150_after_no_press"] = df["rt_lt150"].to_numpy() & no_press
        prev_miss = df["prev_outcome"].astype(object).to_numpy() == "miss"
        df["flag_rt_fast_after_prev_miss"] = df["rt_lt200"].to_numpy() & prev_miss
        return df


# --------------------------------------------------------------------------- child level
def child_table_schema(trial_schema: TableSchema) -> TableSchema:
    """Per-recording AVSRT schema plus the child-level columns."""
    return trial_schema.extend(CHILD_COLUMNS, name=f"{trial_schema.name}_child")


def add_repeated_block_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``repeated_block_subject`` / ``in_repeated_block_run`` on a concatenated child table.

    The surplus runs are every run of a flagged child except the one with
    the most rows (ties: the lowest run number).
    """
    if df.empty:
        df["repeated_block_subject"] = pd.Series(dtype=bool)
        df["in_repeated_block_run"] = pd.Series(dtype=bool)
        return df
    flagged = df["known_issue_flags"].fillna("").str.split(";").map(lambda xs: REPEATED_BLOCK_FLAG in xs)
    df["repeated_block_subject"] = flagged.groupby(df["subject"]).transform("any").astype(bool)
    sizes = df.groupby(["subject", "run"]).size().rename("n").reset_index()
    sizes = sizes.sort_values(["subject", "n", "run"], ascending=[True, False, True])
    main_run = sizes.drop_duplicates("subject").set_index("subject")["run"]
    is_main = df["run"].to_numpy() == df["subject"].map(main_run).to_numpy()
    df["in_repeated_block_run"] = df["repeated_block_subject"].to_numpy(bool) & ~is_main
    return df


def assemble_child_table(tables: Iterable[pd.DataFrame], schema: TableSchema | None = None) -> pd.DataFrame:
    """Concatenate per-run AVSRT tables and add the child-level columns.

    Equivalent to :meth:`AvsrtTrialBuilder.assemble_children` without a
    builder instance; validated against ``schema`` if given.
    """
    df = concat_recordings(tables)
    if df.empty:
        return schema.conform(empty_frame(schema)) if schema is not None else df
    df = add_child_level_indices(df).reset_index(drop=True)
    df["n_runs_in_child"] = df.groupby(["subject", "task"])["run"].transform("nunique").astype("int64")
    df = add_repeated_block_columns(df)
    return schema.conform(df) if schema is not None else df


def build_child_trials(
    recordings: Sequence[Recording],
    builder: AvsrtTrialBuilder,
    sfreq: float,
    read_status: bool = True,
) -> pd.DataFrame:
    """Load and build every AVSRT run of the given recordings, then assemble child tables.

    Parameters
    ----------
    recordings
        AVSRT recordings (any number of children).
    builder
        Configured builder.
    sfreq
        Expected sampling rate.
    read_status
        Read Status markers and joins (needed for blocks, joins and
        ``first_after_block_marker``).
    """
    tabs = []
    for r in recordings:
        if r.task != builder.task:
            raise ValueError(f"{r.data_path.name}: not an {builder.task} recording")
        rec = load_recording_events(r, builder.cfg, builder.status_cfg, sfreq, read_status=read_status)
        tabs.append(builder.build(rec))
    return builder.assemble_children(tabs)
