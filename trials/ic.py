"""IC (illusory contours, dot-flip detection) trial builder.

Event model (``configs/events.yaml`` task ``IC``)
-------------------------------------------------
- Contours ``20`` (illusory) / ``21`` (non-illusory) form a jittered stream
  (in-stream SOA about 0.9-1.5 s). They are *context*, not trial rows.
- Dot flips ``77`` are the targets and the trial rows. The response window of a
  flip closes at the next flip (the next contour, a fixed ~0.70 s later, does
  not close it).
- Presses ``100``; pause ``199`` (in events.tsv and Status); ``201`` is
  Status-only: two at run start, one ~3.05 s after each 199 (pause end), and a
  *standalone* one at each 180-contour sequence start.

Tables built from one recording
-------------------------------
- **flips** (trial table): the base :class:`~trials.base.TaskTrialBuilder`
  columns plus the IC covariates declared in :meth:`IcTrialBuilder.extra_columns`.
- **contours**: one row per contour onset (stream covariates, never dropped).
- **presses**: one row per press, classified relative to the last flip at or
  before it (``hit``, ``extra_in_window``, ``anticipation``, ``false_alarm``);
  a false alarm is a press that no flip's response window can explain.

Base hooks used
---------------
- **Foreign-event truncation** (:meth:`IcTrialBuilder.prepare`, run by the
  base :meth:`~trials.base.TaskTrialBuilder.build`). A recording can carry another paradigm's
  triggers (e.g. FAST codes, whose ``21`` was exported into events.tsv as a
  contour). The cut is the earliest of (a) the
  first Status-only onset whose code is not an IC code, when there are at least
  ``foreign_detection.min_count`` of them, and
  (b) ``exclude_time_range.start_s`` of a registry issue whose category mentions
  ``foreign``. Non-trial events and markers at or after the cut are removed from
  the context streams *before* the base build, so foreign "contours" do not enter
  SOA, contamination or contour counts. Flip rows after the cut are never dropped;
  they are flagged ``after_foreign_cut`` (and by the registry exclusion).
- The base computes ``marker_in_lookback_<L>`` (199/201 markers) and
  ``clean_all_lookback_<L>`` (no contour, flip, press or marker); this builder
  adds the per-stream lookback flags (``pause_in_lookback_<L>`` equals the
  base marker flag) and the stricter ``clean_vep_lookback_<L>``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from trials.base import BuildContext, RecordingEvents, TaskTrialBuilder, nullable_int
from trials.matching import as_sorted_samples, count_in, last_before, next_after
from trials.schema import ColumnSpec, TableSchema
from trials.status import joins_between
from utils.config import StatusConfig, TaskEventsConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)

_EPS = 1e-9
PRESS_CLASSES = ("hit", "extra_in_window", "anticipation", "false_alarm")


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class IcParams:
    """IC-specific parameters (``tasks.IC.ic`` in ``configs/events.yaml``).

    Attributes
    ----------
    contour_codes, pause_codes, sequence_marker_codes
        Contour stimulus codes; pause marker (199); Status-only 201.
    pause_resume_201_max_s
        A 201 within this time after a 199 is the pause end; otherwise (and
        after the first contour of the run) it is a standalone sequence start.
    sequence_n_contours
        Contours per sequence (180), for the count-based sequence index.
    nominal_contours
        Contours of a complete run / child (540); flips after that many
        contours are flagged (surplus sessions).
    sequence_complete_min_contours
        A 180-contour sequence is *complete* when it holds at least this many
        contour onsets (observed sizes: 177-180, then 163 and below).
    cap_complete_sequences
        Trial cap: flips after the end of the child's
        ``cap_complete_sequences``-th complete sequence are ``beyond_ic_cap``.
    late_hit_ms
        RT above which a hit is "late" (the next contour comes ~700 ms after a flip).
    vep_gap_ms
        ``clean_vep_lookback_<L>`` also requires the lookback window to start at
        least this long after the previous contour onset.
    foreign_min_count, foreign_allowed_status_codes
        Foreign-trigger detection on Status-only onsets.
    collision_press_codes
        Status-only onsets that carry the press bits (100, and the OR
        collisions 116 = 100|20, 117 = 100|21) but are absent from events.tsv.
    """

    contour_codes: tuple[int, ...] = (20, 21)
    pause_codes: tuple[int, ...] = (199,)
    sequence_marker_codes: tuple[int, ...] = (201,)
    pause_resume_201_max_s: float = 4.0
    sequence_n_contours: int = 180
    nominal_contours: int = 540
    sequence_complete_min_contours: int = 175
    cap_complete_sequences: int = 3
    late_hit_ms: float = 700.0
    vep_gap_ms: float = 500.0
    foreign_min_count: int = 10
    foreign_allowed_status_codes: tuple[int, ...] = (20, 21, 77, 100, 199, 201, 116, 117)
    collision_press_codes: tuple[int, ...] = (100, 116, 117)

    @classmethod
    def from_task_config(cls, cfg: TaskEventsConfig) -> IcParams:
        """Read ``cfg.extra['ic']`` (missing keys keep the defaults)."""
        d: Mapping[str, Any] = cfg.extra.get("ic") or {}
        fd: Mapping[str, Any] = d.get("foreign_detection") or {}
        base = cls()
        known = {"contour_codes", "pause_codes", "sequence_marker_codes", "pause_resume_201_max_s",
                 "sequence_n_contours", "nominal_contours", "late_hit_ms", "vep_gap_ms", "foreign_detection",
                 "collision_press_codes", "sequence_complete_min_contours", "cap_complete_sequences"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"IC config: unknown keys in tasks.IC.ic: {sorted(unknown)}")
        p = replace(
            base,
            contour_codes=tuple(int(x) for x in d.get("contour_codes", base.contour_codes)),
            pause_codes=tuple(int(x) for x in d.get("pause_codes", base.pause_codes)),
            sequence_marker_codes=tuple(int(x) for x in d.get("sequence_marker_codes", base.sequence_marker_codes)),
            pause_resume_201_max_s=float(d.get("pause_resume_201_max_s", base.pause_resume_201_max_s)),
            sequence_n_contours=int(d.get("sequence_n_contours", base.sequence_n_contours)),
            nominal_contours=int(d.get("nominal_contours", base.nominal_contours)),
            sequence_complete_min_contours=int(d.get("sequence_complete_min_contours",
                                                     base.sequence_complete_min_contours)),
            cap_complete_sequences=int(d.get("cap_complete_sequences", base.cap_complete_sequences)),
            late_hit_ms=float(d.get("late_hit_ms", base.late_hit_ms)),
            vep_gap_ms=float(d.get("vep_gap_ms", base.vep_gap_ms)),
            foreign_min_count=int(fd.get("min_count", base.foreign_min_count)),
            foreign_allowed_status_codes=tuple(int(x) for x in fd.get("allowed_status_codes",
                                                                     base.foreign_allowed_status_codes)),
            collision_press_codes=tuple(int(x) for x in d.get("collision_press_codes", base.collision_press_codes)),
        )
        bad = set(p.contour_codes) - set(cfg.stimulus_codes)
        if bad:
            raise ValueError(f"IC config: contour codes {sorted(bad)} not in stimulus_codes")
        if set(p.contour_codes) & set(cfg.trial_codes):
            raise ValueError("IC config: contour codes must not be trial codes")
        if p.sequence_n_contours <= 0 or p.nominal_contours <= 0 or p.cap_complete_sequences <= 0:
            raise ValueError("IC config: contour counts must be positive")
        if not 0 < p.sequence_complete_min_contours <= p.sequence_n_contours:
            raise ValueError("IC config: sequence_complete_min_contours must be in (0, sequence_n_contours]")
        return p


# --------------------------------------------------------------------------- inputs / outputs
@dataclass
class IcRecordingEvents(RecordingEvents):
    """:class:`RecordingEvents` after foreign-event truncation.

    Attributes
    ----------
    foreign_cut_sample
        First sample of the foreign segment (``None`` if there is none).
    n_foreign_events_removed
        events.tsv rows removed from the context streams (non-trial codes at or
        after the cut).
    foreign_cut_sources
        ``status`` and/or registry issue ids that defined the cut.
    """

    foreign_cut_sample: int | None = None
    n_foreign_events_removed: int = 0
    foreign_cut_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class IcRecordingSummary:
    """Per-recording counts (keyed by BIDS ID)."""

    subject: str
    task: str
    run: int
    n_flips: int
    n_contours: int
    n_presses: int
    n_hits: int
    n_false_alarms: int
    n_anticipations: int
    n_extra_in_window: int
    span_s: float
    false_alarms_per_min: float
    foreign_cut_s: float
    n_foreign_events_removed: int
    foreign_cut_sources: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class IcTables:
    """All IC tables of one recording."""

    flips: pd.DataFrame
    contours: pd.DataFrame
    presses: pd.DataFrame
    summary: IcRecordingSummary


# --------------------------------------------------------------------------- streams
@dataclass
class _Streams:
    sfreq: float
    flips: np.ndarray
    contours: np.ndarray
    contour_values: np.ndarray
    presses: np.ndarray
    pauses: np.ndarray
    m201: np.ndarray
    standalone201: np.ndarray
    segment_bounds: np.ndarray
    markers_available: bool
    joins: np.ndarray | None
    extra: dict = field(default_factory=dict)

    def ms(self, n: np.ndarray | float) -> np.ndarray:
        return np.asarray(n, dtype=float) * 1000.0 / self.sfreq


def _streams(events: pd.DataFrame, markers: pd.DataFrame | None, joins: np.ndarray | None, sfreq: float,
             cfg: TaskEventsConfig, p: IcParams) -> _Streams:
    """Sorted sample arrays of every IC event stream (events already truncated / recoded)."""
    v = events["value"].to_numpy(np.int64)
    s = events["sample"].to_numpy(np.int64)
    is_c = np.isin(v, list(p.contour_codes))
    contours = as_sorted_samples(s[is_c])
    pause_tsv = s[np.isin(v, list(p.pause_codes))]
    if markers is not None:
        mv = markers["value"].to_numpy(np.int64)
        ms_ = markers["sample"].to_numpy(np.int64)
        pauses = np.unique(np.r_[pause_tsv, ms_[np.isin(mv, list(p.pause_codes))]]).astype(np.int64)
        m201 = np.unique(ms_[np.isin(mv, list(p.sequence_marker_codes))]).astype(np.int64)
    else:
        pauses = np.unique(pause_tsv).astype(np.int64)
        m201 = np.array([], np.int64)
    # standalone 201: after the first contour and not within pause_resume_201_max_s after a pause
    if contours.size and m201.size:
        win = int(round(p.pause_resume_201_max_s * sfreq))
        after_first = m201 > contours[0]
        near_pause = count_in(pauses, m201 - win, m201) > 0
        standalone = m201[after_first & ~near_pause]
    else:
        standalone = np.array([], np.int64)
    return _Streams(
        sfreq=float(sfreq),
        flips=as_sorted_samples(s[np.isin(v, list(cfg.trial_codes))]),
        contours=contours, contour_values=v[is_c],
        presses=as_sorted_samples(s[np.isin(v, list(cfg.response_codes))]),
        pauses=pauses, m201=m201, standalone201=standalone,
        segment_bounds=np.unique(np.r_[pauses, standalone]).astype(np.int64),
        markers_available=markers is not None,
        joins=None if joins is None else as_sorted_samples(joins),
    )


def _lag(st: _Streams, prev: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, pd.arrays.BooleanArray, np.ndarray]:
    """(concat ms, crosses-join, join-clean ms) for intervals prev -> s (prev = -1: none)."""
    concat = np.where(prev >= 0, st.ms(s - prev), np.nan)
    cross = TaskTrialBuilder.across_join(prev, s, st.joins)
    clean = np.where(cross.fillna(False).to_numpy(bool), np.nan, concat)
    return concat, cross, clean


def _lead(st: _Streams, s: np.ndarray, nxt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(concat ms, join-clean ms) for intervals s -> nxt (nxt = -1: none)."""
    has = nxt >= 0
    concat = np.where(has, st.ms(nxt - s), np.nan)
    if st.joins is None:
        return concat, concat.copy()
    cross = has & (joins_between(st.joins, s, np.where(has, nxt, s)) > 0)
    return concat, np.where(cross, np.nan, concat)


def _since_s(st: _Streams, s: np.ndarray, ev: np.ndarray) -> np.ndarray:
    prev = last_before(s, ev)
    return np.where(prev >= 0, (s - prev) / st.sfreq, np.nan)


def _count_le(ev: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.searchsorted(ev, s, side="right").astype(np.int64)


def classify_presses(press_samples: np.ndarray, flip_samples: np.ndarray, matched_samples: np.ndarray,
                     lo_ms: float, hi_ms: float, sfreq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify each press relative to the last flip at or before it.

    Parameters
    ----------
    press_samples, flip_samples
        Sorted press and flip onsets.
    matched_samples
        Per flip, the sample of its matched response under the window (``-1`` if none).
    lo_ms, hi_ms
        Closed response window.
    sfreq
        Sampling rate.

    Returns
    -------
    classes, flip_index, latency_ms
        ``classes`` in :data:`PRESS_CLASSES`. ``hit``: the matched response of
        its flip; ``extra_in_window``: another press in that flip's window;
        ``anticipation``: latency ``< lo``; ``false_alarm``: before any flip or
        with latency ``> hi`` (no flip's window explains it). ``flip_index`` is
        -1 before the first flip; latency is NaN there.
    """
    p = as_sorted_samples(press_samples)
    f = as_sorted_samples(flip_samples)
    m = np.asarray(matched_samples, np.int64)
    k = np.searchsorted(f, p, side="right") - 1
    has = k >= 0
    lat = np.where(has, (p - f[np.maximum(k, 0)]) * 1000.0 / sfreq if f.size else np.nan, np.nan)
    cls = np.full(p.size, "false_alarm", dtype=object)
    if p.size and f.size:
        is_hit = has & (m[np.maximum(k, 0)] == p)
        in_win = has & (lat >= lo_ms - _EPS) & (lat <= hi_ms + _EPS)
        antic = has & (lat < lo_ms - _EPS)
        cls[in_win] = "extra_in_window"
        cls[antic] = "anticipation"
        cls[is_hit] = "hit"
    return cls, k.astype(np.int64), lat


def _c(name: str, dtype: str, nullable: bool = False, description: str = "") -> ColumnSpec:
    return ColumnSpec(name, dtype, nullable, description)


# --------------------------------------------------------------------------- builder
#: Key columns of the contour and press tables (same definitions as the flip table).
_KEY_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("subject", "str", description="BIDS participant_id"),
    _c("task", "str", description="BIDS task label"),
    _c("run", "int", description="BIDS run number"),
    _c("sample", "int", description="event onset sample on the recording's concatenated sample clock (key)"),
    _c("onset_s", "float", description="event onset from events.tsv (= sample / sfreq)"),
)
#: Known-issue columns of the contour and press tables.
_ISSUE_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("excluded_known_issue", "bool", description="row excluded by a configs/known_issues.yaml action"),
    _c("known_issue_ids", "str", description="';'-joined registry ids whose actions touch this row ('' if none)"),
    _c("known_issue_flags", "str", description="';'-joined flag_only labels ('' if none)"),
)


class IcTrialBuilder(TaskTrialBuilder):
    """Trial builder for the IC task (rows = dot flips).

    Parameters
    ----------
    task_cfg, status_cfg, registry
        As for :class:`~trials.base.TaskTrialBuilder`; ``task_cfg``
        may carry an ``ic`` mapping (:class:`IcParams`).

    Examples
    --------
    >>> b = IcTrialBuilder(events_cfg.task("IC"), events_cfg.status, registry)   # doctest: +SKIP
    >>> tables = b.build_all(load_recording_events(rec, b.cfg, events_cfg.status, 512.0))  # doctest: +SKIP
    """

    task: ClassVar[str] = "IC"

    def __init__(self, task_cfg: TaskEventsConfig, status_cfg: StatusConfig,
                 registry: KnownIssuesRegistry | None = None) -> None:
        super().__init__(task_cfg, status_cfg, registry)
        self.params = IcParams.from_task_config(task_cfg)
        want = set(self.params.pause_codes) | set(self.params.sequence_marker_codes)
        if set(task_cfg.marker_codes) != want:
            raise ValueError(f"IC config: marker_codes {sorted(task_cfg.marker_codes)} must equal pause + sequence "
                             f"marker codes {sorted(want)} (pause_in_lookback is the base marker flag)")

    # ------------------------------------------------------------------ child level
    def child_extra_columns(self) -> tuple[ColumnSpec, ...]:
        n = self.params.nominal_contours
        return (
            _c("n_contours_before_in_child", "int",
               description="contours of the child's earlier runs (by run number) plus n_contours_before_in_run"),
            _c("beyond_nominal_contours_child", "bool",
               description=f"more than {n} contours precede the flip across the child's runs (count-based; "
               "reference only, the cap is beyond_ic_cap)"),
            _c("n_complete_seq180_before_in_child", "int",
               description="complete 180-contour sequences (seq180_complete) of the child that precede the flip's "
               "sequence, over runs in run order then sequence order"),
            _c("beyond_ic_cap", "bool",
               description=f"cap: the flip lies after the end of complete 180-contour sequence "
               f"number {self.params.cap_complete_sequences} of the child "
               f"(n_complete_seq180_before_in_child >= {self.params.cap_complete_sequences}); children with fewer "
               "complete sequences are not capped"),
        )

    def add_child_columns(self, df: pd.DataFrame, run_table: pd.DataFrame | None = None) -> pd.DataFrame:
        """Child-level contour counts (:func:`add_ic_child_contour_counts`).

        ``run_table`` with an ``n_contours`` column (one row per recording,
        including runs without flips) gives the run offsets; otherwise the
        flips' ``n_contours_in_run`` is used.
        """
        rt = run_table if run_table is not None and "n_contours" in run_table.columns else None
        out = add_ic_child_contour_counts(df, self.params.nominal_contours, rt)
        out = add_ic_cap(out, self.params.cap_complete_sequences)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ preparation
    def prepare(self, rec: RecordingEvents) -> IcRecordingEvents:
        """Apply the foreign-event truncation (idempotent).

        Returns
        -------
        IcRecordingEvents
            Copy of ``rec`` whose events and markers exclude the foreign segment
            (trial-code events are kept).
        """
        if isinstance(rec, IcRecordingEvents):
            return rec
        p = self.params
        cuts: list[int] = []
        sources: list[str] = []
        if rec.crosscheck is not None and p.foreign_min_count > 0:
            so = rec.crosscheck.status_only
            fo = so.loc[~so["value"].isin(list(p.foreign_allowed_status_codes))]
            if len(fo) >= p.foreign_min_count:
                cuts.append(int(fo["sample"].min()))
                sources.append("status")
        for iss in self.issues_for(rec):
            if "foreign" not in iss.category:
                continue
            for a in iss.actions:
                if a.type == "exclude_time_range" and a.get("start_s") is not None:
                    cuts.append(int(math.ceil(float(a.get("start_s")) * rec.sfreq - 1e-9)))
                    sources.append(iss.id)
        ev = rec.events
        markers = rec.markers
        n_removed = 0
        cut: int | None = None
        if cuts:
            cut = min(cuts)
            drop = (ev["sample"].to_numpy() >= cut) & ~ev["value"].isin(list(self.cfg.trial_codes)).to_numpy()
            n_removed = int(drop.sum())
            ev = ev.loc[~drop].reset_index(drop=True)
            if markers is not None:
                markers = markers.loc[markers["sample"] < cut].reset_index(drop=True)
            if n_removed:
                log.info("%s %s run-%02d: foreign segment from sample %d (%s): %d events removed from the "
                         "context streams", rec.subject, rec.task, rec.run, cut, ",".join(sources), n_removed)
        return IcRecordingEvents(
            subject=rec.subject, task=rec.task, run=rec.run, sfreq=rec.sfreq, events=ev, markers=markers,
            joins=rec.joins, n_times=rec.n_times, crosscheck=rec.crosscheck, sources=rec.sources,
            foreign_cut_sample=cut, n_foreign_events_removed=n_removed, foreign_cut_sources=tuple(sources),
        )

    def _status_only_presses(self, rec: RecordingEvents, cut: int | None) -> np.ndarray | None:
        """Sorted Status-only onsets with press bits (``None`` if Status was not read)."""
        if rec.crosscheck is None:
            return None
        so = rec.crosscheck.status_only
        x = so.loc[so["value"].isin(list(self.params.collision_press_codes)), "sample"].to_numpy(np.int64)
        if cut is not None:
            x = x[x < cut]
        return np.unique(x).astype(np.int64)

    def build_all(self, rec: RecordingEvents) -> IcTables:
        """Flip, contour and press tables plus a per-recording summary."""
        clean = self.prepare(rec)
        flips = self.build(clean)
        contours = self.build_contours(clean)
        presses = self.build_presses(clean, flips)
        return IcTables(flips, contours, presses, self.summarise(clean, flips, contours, presses))

    # ------------------------------------------------------------------ flip columns
    def extra_columns(self) -> tuple[ColumnSpec, ...]:
        p = self.params
        cols = [
            _c("t_since_prev_flip_ms", "float", True, "time since previous flip (= t_since_prev_trial_ms); NA across joins"),
            _c("t_since_prev_flip_ms_concat", "float", True, "same on the concatenated sample clock"),
            _c("n_contours_since_prev_flip", "int", description="contours in (previous flip, onset); run start if none"),
            _c("prev_contour_value", "Int", True, "code of the last contour before onset"),
            _c("prev_contour_type", "str", description="label of the last contour before onset, or 'none'"),
            _c("t_since_prev_contour_ms", "float", True, "time since last contour onset; NA if none or across a join"),
            _c("t_since_prev_contour_ms_concat", "float", True, "same on the concatenated sample clock (ignores joins)"),
            _c("prev_contour_crosses_join", "boolean", True, "a bit-16 join lies in (previous contour, onset]; NA if no "
               "previous contour or joins unknown"),
            _c("t_to_next_contour_ms", "float", True, "time to the next contour onset; NA if none or across a join"),
            _c("t_to_next_contour_ms_concat", "float", True, "same on the concatenated sample clock (ignores joins)"),
            _c("n_contours_before_in_run", "int", description="contour onsets of the run strictly before the flip onset"),
            _c("n_contours_in_run", "int", description="contour onsets in the run (after the foreign-segment cut)"),
            _c("t_since_run_first_contour_s", "float", True, "onset minus first contour onset (concatenated clock)"),
            _c("t_since_prev_pause_s", "float", True, "since last 199 (concatenated clock)"),
            _c("t_since_prev_201_s", "float", True, "since last Status 201 (any role); NA without markers"),
            _c("post_pause_first_flip", "bool", description="a pause precedes and no contour lies between it and the flip"),
            _c("post_sequence_start_first_flip", "bool",
               description="a standalone 201 precedes and no contour lies between it and the flip"),
            _c("seq180_index", "Int", True, "1-based 180-contour sequence (1 + standalone 201s before); NA without markers"),
            _c("seq180_index_by_count", "int",
               description=f"1 + n_contours_before_in_run // {p.sequence_n_contours} (marker-free)"),
            _c("seq180_n_contours", "int", description="contour onsets of the run in the flip's 180-contour sequence "
               "(seq180_index; seq180_index_by_count without markers)"),
            _c("seq180_complete", "bool", description=f"seq180_n_contours >= {p.sequence_complete_min_contours} "
               "(a complete sequence; up to 5 lost contour triggers allowed)"),
            _c("segment_index", "Int", True, "1-based segment between 199 / standalone-201 boundaries; NA without markers"),
            _c("t_since_segment_start_s", "float", True, "since the segment boundary; NA in the leading unmarked segment"),
            _c("rt_gt_late_threshold", "bool", description=f"responded with RT > {p.late_hit_ms:g} ms"),
            _c("response_after_next_contour", "bool", description="matched response at or after the next contour onset"),
            _c("late_hit", "bool", description="rt_gt_late_threshold or response_after_next_contour"),
        ]
        for w in self.cfg.rt_windows.values():
            sfx = "" if w.name == self.cfg.primary_window else f"_{w.name}"
            cols.append(_c(f"n_status_only_press_in_window{sfx}", "Int", True,
                           f"Status-only press-bit onsets ({'/'.join(map(str, p.collision_press_codes))}, absent "
                           f"from events.tsv) in [onset+{w.lo_ms:g}, onset+{w.hi_ms:g}] ms; NA without Status"))
        for L in self.cfg.lookbacks_ms:
            cols += [
                _c(f"contour_in_lookback_{L}", "bool", description=f"contour onset in [t0-{L} ms, t0)"),
                _c(f"flip_in_lookback_{L}", "bool", description=f"flip onset in [t0-{L} ms, t0)"),
                _c(f"pause_in_lookback_{L}", "bool",
                   description=f"199 or 201 marker in [t0-{L} ms, t0) (= base marker_in_lookback_{L})"),
                _c(f"clean_vep_lookback_{L}", "bool",
                   description=f"clean_all and the lookback starts >= {p.vep_gap_ms:g} ms after the previous contour"),
            ]
        cols += [
            _c("n_false_alarms_since_prev_flip", "int", description="false-alarm presses in [previous flip, onset)"),
            _c("n_false_alarms_after_flip", "int", description="false-alarm presses (primary window) in [onset, next "
               "flip), or to the end of the run for the last flip: the presses attributed to this flip's interval"),
            _c("n_false_alarms_before_first_flip", "int", description="false-alarm presses of the run before its "
               "first flip (constant per run)"),
            _c("after_foreign_cut", "bool", description="onset in the foreign-trigger segment (row kept, flagged)"),
            _c("foreign_cut_s", "float", True, "start of the foreign segment of this recording (NaN: none)"),
            _c("surplus_contours_session", "bool", description="registry flag surplus_contours"),
            _c("beyond_nominal_contours_run", "bool",
               description=f"more than {p.nominal_contours} contours precede the flip in this run"),
        ]
        return tuple(cols)

    def add_task_columns(self, trials: pd.DataFrame, rec: RecordingEvents, ctx: BuildContext) -> pd.DataFrame:
        p = self.params
        df = trials
        st = _streams(ctx.events, ctx.markers, ctx.joins, ctx.sfreq, self.cfg, p)
        s = ctx.trial_samples
        n = s.size

        df["t_since_prev_flip_ms"] = df["t_since_prev_trial_ms"].to_numpy(float)
        df["t_since_prev_flip_ms_concat"] = df["t_since_prev_trial_ms_concat"].to_numpy(float)
        prev_flip = np.r_[-1, s[:-1]] if n else np.array([], np.int64)
        df["n_contours_since_prev_flip"] = count_in(st.contours, np.where(prev_flip >= 0, prev_flip + 1, 0), s)

        ci = np.searchsorted(st.contours, s, side="left") - 1
        has_c = ci >= 0
        pc = np.where(has_c, st.contours[np.maximum(ci, 0)] if st.contours.size else -1, -1).astype(np.int64)
        pcv = np.where(has_c, st.contour_values[np.maximum(ci, 0)] if st.contours.size else 0, 0)
        df["prev_contour_value"] = nullable_int(pcv, has_c)
        df["prev_contour_type"] = pd.array(
            [self.cfg.stimulus_codes.get(int(x), str(int(x))) if h else "none" for x, h in zip(pcv, has_c)],
            dtype="string")
        c, x, v = _lag(st, pc, s)
        df["t_since_prev_contour_ms"], df["t_since_prev_contour_ms_concat"], df["prev_contour_crosses_join"] = v, c, x
        nc = next_after(s, st.contours)
        c, v = _lead(st, s, nc)
        df["t_to_next_contour_ms"], df["t_to_next_contour_ms_concat"] = v, c
        n_before = np.searchsorted(st.contours, s, side="left").astype(np.int64)
        df["n_contours_before_in_run"] = n_before
        df["n_contours_in_run"] = np.full(n, st.contours.size, np.int64)
        df["t_since_run_first_contour_s"] = ((s - st.contours[0]) / st.sfreq if st.contours.size
                                             else np.full(n, np.nan))
        df["t_since_prev_pause_s"] = _since_s(st, s, st.pauses)
        df["t_since_prev_201_s"] = _since_s(st, s, st.m201) if st.markers_available else np.full(n, np.nan)

        pp = last_before(s, st.pauses)
        df["post_pause_first_flip"] = (pp >= 0) & (pc < pp)
        ps = last_before(s, st.standalone201)
        df["post_sequence_start_first_flip"] = (ps >= 0) & (pc < ps)
        if st.markers_available:
            df["seq180_index"] = pd.array(_count_le(st.standalone201, s) + 1, dtype="Int64")
            kseg = _count_le(st.segment_bounds, s)
            df["segment_index"] = pd.array(kseg + 1, dtype="Int64")
            lastb = st.segment_bounds[np.maximum(kseg - 1, 0)] if st.segment_bounds.size else np.zeros(n, np.int64)
            df["t_since_segment_start_s"] = np.where(kseg > 0, (s - lastb) / st.sfreq, np.nan)
        else:
            df["seq180_index"] = pd.array([pd.NA] * n, dtype="Int64")
            df["segment_index"] = pd.array([pd.NA] * n, dtype="Int64")
            df["t_since_segment_start_s"] = np.full(n, np.nan)
        df["seq180_index_by_count"] = n_before // p.sequence_n_contours + 1
        # contours per sequence of the run (marker sequences: 1 + standalone 201s at or before the contour onset)
        if st.markers_available:
            seq_c = _count_le(st.standalone201, st.contours) + 1
            seq_f = df["seq180_index"].to_numpy(np.int64)
        else:
            seq_c = np.arange(st.contours.size, dtype=np.int64) // p.sequence_n_contours + 1
            seq_f = df["seq180_index_by_count"].to_numpy(np.int64)
        top = int(max(seq_f.max(initial=0), seq_c.max(initial=0))) + 1
        sizes = np.bincount(seq_c, minlength=top).astype(np.int64)
        df["seq180_n_contours"] = sizes[seq_f] if n else np.array([], np.int64)
        df["seq180_complete"] = df["seq180_n_contours"].to_numpy(np.int64) >= p.sequence_complete_min_contours

        resp = df["responded"].to_numpy(bool)
        rt = df["rt_ms"].to_numpy(float)
        rsamp = df["response_sample"].fillna(-1).to_numpy(np.int64)
        late_rt = resp & (rt > p.late_hit_ms + _EPS)
        after_nc = resp & (nc >= 0) & (rsamp >= nc)
        df["rt_gt_late_threshold"] = late_rt
        df["response_after_next_contour"] = after_nc
        df["late_hit"] = late_rt | after_nc

        cut = getattr(rec, "foreign_cut_sample", None)
        so_press = self._status_only_presses(rec, cut)
        for w in self.cfg.rt_windows.values():
            sfx = "" if w.name == self.cfg.primary_window else f"_{w.name}"
            if so_press is None:
                df[f"n_status_only_press_in_window{sfx}"] = pd.array([pd.NA] * n, dtype="Int64")
            else:
                lo_s = s + np.ceil(w.lo_ms * st.sfreq / 1000.0 - _EPS).astype(np.int64)
                hi_s = s + np.floor(w.hi_ms * st.sfreq / 1000.0 + _EPS).astype(np.int64)
                df[f"n_status_only_press_in_window{sfx}"] = pd.array(count_in(so_press, lo_s, hi_s + 1),
                                                                     dtype="Int64")

        lag_c = np.where(pc >= 0, st.ms(s - pc), np.inf)
        lag_f = np.where(prev_flip >= 0, st.ms(s - prev_flip), np.inf)
        for L in self.cfg.lookbacks_ms:
            df[f"contour_in_lookback_{L}"] = lag_c <= L + _EPS
            df[f"flip_in_lookback_{L}"] = lag_f <= L + _EPS
            df[f"pause_in_lookback_{L}"] = df[f"marker_in_lookback_{L}"].to_numpy(bool)   # markers = 199 / 201
            clean = df[f"clean_all_lookback_{L}"].to_numpy(bool)                          # base: no stim/press/marker
            df[f"clean_vep_lookback_{L}"] = clean & ((pc < 0) | (lag_c - L >= p.vep_gap_ms - _EPS))

        lo, hi = self.cfg.primary.lo_ms, self.cfg.primary.hi_ms
        cls, k, _ = classify_presses(st.presses, s, rsamp, lo, hi, st.sfreq)
        fa_k = k[cls == "false_alarm"]
        # a false alarm assigned to flip j (or -1: before the first flip) precedes flip j + 1
        df["n_false_alarms_since_prev_flip"] = np.bincount(fa_k + 1, minlength=n + 1)[:n].astype(np.int64)
        df["n_false_alarms_after_flip"] = np.bincount(fa_k[fa_k >= 0], minlength=n)[:n].astype(np.int64)
        df["n_false_alarms_before_first_flip"] = np.full(n, int((fa_k < 0).sum()), np.int64)

        df["after_foreign_cut"] = (s >= cut) if cut is not None else np.zeros(n, bool)
        df["foreign_cut_s"] = np.full(n, cut / st.sfreq if cut is not None else np.nan)
        flags = df["known_issue_flags"].astype(str).to_numpy()
        df["surplus_contours_session"] = np.array(["surplus_contours" in f.split(";") for f in flags], bool)
        df["beyond_nominal_contours_run"] = n_before > p.nominal_contours
        return df

    # ------------------------------------------------------------------ contour table
    def contour_schema(self) -> TableSchema:
        """Schema of the contour-stream table."""
        cols = [
            *_KEY_COLUMNS,
            _c("contour_index_in_run", "int", description="1-based position among the contour onsets of the run"),
            _c("contour_key", "str", description="stable key subject_task_run-XX_c<sample>"),
            _c("value", "int", description="contour code after known-issue recoding (20 illusory, 21 non-illusory)"),
            _c("value_original", "int", description="contour code as logged in events.tsv"),
            _c("condition", "str", description="label of value from stimulus_codes"),
            _c("soa_prev_contour_ms", "float", True, "since previous contour; NA if none or across a join"),
            _c("soa_prev_contour_ms_concat", "float", True, "same on the concatenated sample clock (ignores joins)"),
            _c("soa_prev_contour_crosses_join", "boolean", True, "a join lies in (previous contour, onset]; NA if none"),
            _c("flip_since_prev_contour", "bool", description="a flip lies in (previous contour, onset)"),
            _c("t_since_prev_flip_ms", "float", True, "onset minus the last flip strictly before it; NA if none or across "
               "a join"),
            _c("t_since_prev_flip_ms_concat", "float", True, "same on the concatenated sample clock (ignores joins)"),
            _c("t_to_next_flip_ms_concat", "float", True, "next flip strictly after onset minus onset, concatenated clock; "
               "NA if none"),
            _c("t_since_prev_press_ms_concat", "float", True, "onset minus the last press strictly before it, "
               "concatenated clock; NA if none"),
            _c("first_after_pause", "bool", description="a pause precedes and no contour lies between"),
            _c("block_index_in_run", "Int", True, "as the flip table (1 + pauses before; NA without markers)"),
            _c("seq180_index", "Int", True, "as the flip table: 1 + standalone 201 markers before onset; NA without "
               "markers"),
            _c("segment_index", "Int", True, "as the flip table: 1 + 199 / standalone-201 boundaries before onset"),
            _c("beyond_nominal_contours_run", "bool", description="contour_index_in_run > nominal contours"),
            _c("after_foreign_cut", "bool", description="onset at or after the foreign-trigger cut (row kept, flagged)"),
            _c("n_joins_before", "Int", True, "bit-16 joins at or before onset; NA without Status"),
        ]
        for L in self.cfg.lookbacks_ms:
            cols += [_c(f"press_in_lookback_{L}", "bool", description=f"a press in [onset - {L} ms, onset)"),
                     _c(f"flip_in_lookback_{L}", "bool", description=f"a flip onset in [onset - {L} ms, onset)")]
        cols += list(_ISSUE_COLUMNS)
        return TableSchema("contours_IC", tuple(cols), allow_extra=False, keys=("subject", "task", "run", "sample"))

    def _ctx_like(self, rec: IcRecordingEvents, ev: pd.DataFrame) -> BuildContext:
        e = np.array([], np.int64)
        return BuildContext(trial_samples=e, stim_samples=e, stim_values=e, press_samples=e, bound_samples=e,
                            markers=rec.markers, joins=None if rec.joins is None else as_sorted_samples(rec.joins),
                            events=ev, sfreq=rec.sfreq)

    def _issue_columns(self, df: pd.DataFrame, rec: IcRecordingEvents, ctx: BuildContext, key: str) -> None:
        tmp = pd.DataFrame({"sample": df["sample"].to_numpy(np.int64), "onset_s": df["onset_s"].to_numpy(float),
                            "trial_key": df[key].to_numpy(object)})
        self.add_known_issue_flags(tmp, ctx, self.issues_for(rec), {})
        for col in ("excluded_known_issue", "known_issue_ids", "known_issue_flags"):
            df[col] = tmp[col].to_numpy()

    def _recoded_events(self, rec: IcRecordingEvents) -> pd.DataFrame:
        ev = rec.events.copy()
        ev["value_original"] = ev["value"]
        self.apply_recodes(ev, self.issues_for(rec), rec)
        return ev

    def build_contours(self, rec: RecordingEvents) -> pd.DataFrame:
        """Contour-stream table of one recording (one row per contour onset)."""
        rec = self.prepare(rec)
        p = self.params
        ev = self._recoded_events(rec)
        st = _streams(ev, rec.markers, rec.joins, rec.sfreq, self.cfg, p)
        cev = ev.loc[ev["value"].isin(list(p.contour_codes))].reset_index(drop=True)
        s = st.contours
        n = s.size
        df = pd.DataFrame({
            "subject": pd.array([rec.subject] * n, dtype="string"), "task": pd.array([rec.task] * n, dtype="string"),
            "run": np.full(n, rec.run, np.int64), "sample": s, "onset_s": cev["onset"].to_numpy(float),
            "contour_index_in_run": np.arange(1, n + 1, dtype=np.int64),
        })
        df["contour_key"] = pd.array([f"{rec.subject}_{rec.task}_run-{rec.run:02d}_c{x}" for x in s], dtype="string")
        df["value"] = cev["value"].to_numpy(np.int64)
        df["value_original"] = cev["value_original"].to_numpy(np.int64)
        df["condition"] = pd.array([self.cfg.stimulus_codes.get(int(x), str(x)) for x in df["value"]], dtype="string")
        prev = np.r_[-1, s[:-1]] if n else np.array([], np.int64)
        c, x, v = _lag(st, prev, s)
        df["soa_prev_contour_ms"], df["soa_prev_contour_ms_concat"], df["soa_prev_contour_crosses_join"] = v, c, x
        df["flip_since_prev_contour"] = count_in(st.flips, np.where(prev >= 0, prev + 1, 0), s) > 0 if n else np.zeros(0, bool)
        pf = last_before(s, st.flips)
        c, x, v = _lag(st, pf, s)
        df["t_since_prev_flip_ms"], df["t_since_prev_flip_ms_concat"] = v, c
        nf = next_after(s, st.flips)
        df["t_to_next_flip_ms_concat"] = np.where(nf >= 0, st.ms(nf - s), np.nan)
        pr = last_before(s, st.presses)
        df["t_since_prev_press_ms_concat"] = np.where(pr >= 0, st.ms(s - pr), np.nan)
        pp = last_before(s, st.pauses)
        df["first_after_pause"] = (pp >= 0) & (prev < pp)
        if st.markers_available:
            starts = as_sorted_samples(st.pauses)
            df["block_index_in_run"] = pd.array(_count_le(starts, s) + 1, dtype="Int64")
            df["seq180_index"] = pd.array(_count_le(st.standalone201, s) + 1, dtype="Int64")
            df["segment_index"] = pd.array(_count_le(st.segment_bounds, s) + 1, dtype="Int64")
        else:
            for col in ("block_index_in_run", "seq180_index", "segment_index"):
                df[col] = pd.array([pd.NA] * n, dtype="Int64")
        df["beyond_nominal_contours_run"] = df["contour_index_in_run"].to_numpy() > p.nominal_contours
        cut = rec.foreign_cut_sample
        df["after_foreign_cut"] = (s >= cut) if cut is not None else np.zeros(n, bool)
        df["n_joins_before"] = (pd.array(np.searchsorted(st.joins, s, side="right"), dtype="Int64")
                                if st.joins is not None else pd.array([pd.NA] * n, dtype="Int64"))
        lag_p = np.where(pr >= 0, st.ms(s - pr), np.inf)
        lag_f = np.where(pf >= 0, st.ms(s - pf), np.inf)
        for L in self.cfg.lookbacks_ms:
            df[f"press_in_lookback_{L}"] = lag_p <= L + _EPS
            df[f"flip_in_lookback_{L}"] = lag_f <= L + _EPS
        self._issue_columns(df, rec, self._ctx_like(rec, ev), "contour_key")
        return self.contour_schema().conform(df)

    # ------------------------------------------------------------------ press table
    def press_schema(self) -> TableSchema:
        """Schema of the press table."""
        cols = [
            *_KEY_COLUMNS,
            _c("press_index_in_run", "int", description="1-based position among the presses (code 100) of the run"),
            _c("press_key", "str", description="stable key of the press (subject_task_run-XX_ plus sample)"),
            _c("prev_flip_sample", "Int", True, "last flip at or before the press"),
            _c("lat_from_prev_flip_ms", "float", True, "press minus prev_flip_sample (concatenated clock); NA before the "
               "first flip"),
            _c("t_since_prev_contour_ms_concat", "float", True, "press minus the last contour strictly before it, "
               "concatenated clock; NA if none"),
            _c("press_class", "str", description="primary window: " + "/".join(PRESS_CLASSES)),
            _c("is_false_alarm", "bool", description="press_class == false_alarm (primary window)"),
        ]
        for w in self.cfg.rt_windows.values():
            if w.name != self.cfg.primary_window:
                cols += [_c(f"press_class_{w.name}", "str", description=f"press_class under the sensitivity window "
                            f"[{w.lo_ms:g}, {w.hi_ms:g}] ms"),
                         _c(f"is_false_alarm_{w.name}", "bool", description=f"press_class_{w.name} == false_alarm")]
        cols += [
            _c("block_index_in_run", "Int", True, "1 + 199 pauses at or before the press; NA without markers"),
            _c("after_foreign_cut", "bool", description="press at or after the foreign-trigger cut (row kept, flagged)"),
            *_ISSUE_COLUMNS,
        ]
        return TableSchema("presses_IC", tuple(cols), allow_extra=False, keys=("subject", "task", "run", "sample"))

    def build_presses(self, rec: RecordingEvents, flips: pd.DataFrame | None = None) -> pd.DataFrame:
        """Press table of one recording.

        Parameters
        ----------
        rec
            Recording events (truncated if needed).
        flips
            The recording's flip table (built if omitted); its matched
            responses define the ``hit`` class.
        """
        rec = self.prepare(rec)
        if flips is None:
            flips = self.build(rec)
        ev = self._recoded_events(rec)
        st = _streams(ev, rec.markers, rec.joins, rec.sfreq, self.cfg, self.params)
        pev = ev.loc[ev["value"].isin(list(self.cfg.response_codes))]
        n_dup = int(pev.duplicated("sample").sum())
        if n_dup:  # two press rows on one sample are one physical press; the table is keyed by sample
            log.warning("%s %s run-%02d: %d duplicated press samples merged in the press table", rec.subject,
                        rec.task, rec.run, n_dup)
        pev = pev.drop_duplicates("sample").reset_index(drop=True)
        s = as_sorted_samples(pev["sample"].to_numpy(np.int64))
        n = s.size
        fs = flips["sample"].to_numpy(np.int64)
        df = pd.DataFrame({
            "subject": pd.array([rec.subject] * n, dtype="string"), "task": pd.array([rec.task] * n, dtype="string"),
            "run": np.full(n, rec.run, np.int64), "sample": s, "onset_s": pev["onset"].to_numpy(float),
            "press_index_in_run": np.arange(1, n + 1, dtype=np.int64),
        })
        df["press_key"] = pd.array([f"{rec.subject}_{rec.task}_run-{rec.run:02d}_p{x}" for x in s], dtype="string")
        for w in self.cfg.rt_windows.values():
            if w.name == self.cfg.primary_window:
                matched = flips["response_sample"].fillna(-1).to_numpy(np.int64)
            else:
                rt = flips[f"rt_ms_{w.name}"].to_numpy(float)
                matched = np.where(np.isfinite(rt), fs + np.round(np.nan_to_num(rt) * rec.sfreq / 1000.0), -1
                                   ).astype(np.int64)
            cls, k, lat = classify_presses(s, fs, matched, w.lo_ms, w.hi_ms, rec.sfreq)
            sfx = "" if w.name == self.cfg.primary_window else f"_{w.name}"
            df[f"press_class{sfx}"] = pd.array(cls, dtype="string")
            df[f"is_false_alarm{sfx}"] = cls == "false_alarm"
            if not sfx:
                df["prev_flip_sample"] = nullable_int(np.where(k >= 0, fs[np.maximum(k, 0)] if fs.size else -1, -1),
                                                       k >= 0)
                df["lat_from_prev_flip_ms"] = lat
        pc = last_before(s, st.contours)
        df["t_since_prev_contour_ms_concat"] = np.where(pc >= 0, st.ms(s - pc), np.nan)
        if st.markers_available:
            df["block_index_in_run"] = pd.array(_count_le(st.pauses, s) + 1, dtype="Int64")
        else:
            df["block_index_in_run"] = pd.array([pd.NA] * n, dtype="Int64")
        cut = rec.foreign_cut_sample
        df["after_foreign_cut"] = (s >= cut) if cut is not None else np.zeros(n, bool)
        self._issue_columns(df, rec, self._ctx_like(rec, ev), "press_key")
        return self.press_schema().conform(df)

    # ------------------------------------------------------------------ summary
    def summarise(self, rec: IcRecordingEvents, flips: pd.DataFrame, contours: pd.DataFrame,
                  presses: pd.DataFrame) -> IcRecordingSummary:
        """Per-recording counts; the span runs from the first to the last kept event."""
        on = rec.events["onset"].to_numpy(float)
        span = float(on.max() - on.min()) if on.size else float("nan")
        n_fa = int(presses["is_false_alarm"].sum())
        cut = rec.foreign_cut_sample
        cls = presses["press_class"].astype(str)
        return IcRecordingSummary(
            subject=rec.subject, task=rec.task, run=rec.run, n_flips=len(flips), n_contours=len(contours),
            n_presses=len(presses), n_hits=int(flips["responded"].sum()), n_false_alarms=n_fa,
            n_anticipations=int((cls == "anticipation").sum()), n_extra_in_window=int((cls == "extra_in_window").sum()),
            span_s=span, false_alarms_per_min=(n_fa / (span / 60.0)) if span and span > 0 else float("nan"),
            foreign_cut_s=(cut / rec.sfreq) if cut is not None else float("nan"),
            n_foreign_events_removed=rec.n_foreign_events_removed,
            foreign_cut_sources=";".join(rec.foreign_cut_sources),
        )


# --------------------------------------------------------------------------- child level
def add_ic_cap(flips: pd.DataFrame, cap_complete_sequences: int = 3) -> pd.DataFrame:
    """Flag flips after the end of the child's ``cap_complete_sequences``-th complete sequence.

    Sequences are ordered by (run, sequence index) within child x task; the sequence index is ``seq180_index``
    (``seq180_index_by_count`` where it is NA). ``n_complete_seq180_before_in_child`` counts the complete
    sequences (``seq180_complete``) strictly before the flip's sequence; ``beyond_ic_cap`` is that count
    ``>= cap_complete_sequences``. Sequences without any flip do not enter the count (a complete sequence
    holds about 36 flips, so none is missed in practice).
    """
    df = flips.copy()
    if df.empty:
        df["n_complete_seq180_before_in_child"] = pd.Series(dtype="int64")
        df["beyond_ic_cap"] = pd.Series(dtype=bool)
        return df
    seq = df["seq180_index"].astype("Int64").fillna(df["seq180_index_by_count"].astype("Int64")).astype("int64")
    key = pd.DataFrame({"subject": df["subject"].to_numpy(), "task": df["task"].to_numpy(),
                        "run": df["run"].to_numpy(np.int64), "seq": seq.to_numpy(np.int64),
                        "complete": df["seq180_complete"].to_numpy(bool)})
    u = key.drop_duplicates(["subject", "task", "run", "seq"]).sort_values(["subject", "task", "run", "seq"])
    c = u["complete"].astype("int64")
    u["n_before"] = (c.groupby([u["subject"], u["task"]]).cumsum() - c).astype("int64")
    m = key.merge(u[["subject", "task", "run", "seq", "n_before"]], on=["subject", "task", "run", "seq"],
                  how="left")["n_before"].to_numpy(np.int64)
    df["n_complete_seq180_before_in_child"] = m
    df["beyond_ic_cap"] = m >= int(cap_complete_sequences)
    return df


def add_ic_child_contour_counts(flips: pd.DataFrame, nominal_contours: int = 540,
                                contours_per_run: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add child-level contour counts across runs (surplus-contour sessions).

    Parameters
    ----------
    flips
        Concatenated flip tables of one or more children.
    nominal_contours
        Contours of a complete session (540).
    contours_per_run
        Optional ``subject, task, run, n_contours`` table (e.g. from the
        recording summaries) so that runs without any flip still count; by
        default ``n_contours_in_run`` of the flip table is used.

    Returns
    -------
    pandas.DataFrame
        ``flips`` sorted by subject/task/run/sample with
        ``n_contours_before_in_child`` (contours of earlier runs, by run number,
        plus ``n_contours_before_in_run``) and ``beyond_nominal_contours_child``
        (more than ``nominal_contours`` contours precede the flip). Flags only.
    """
    df = flips.sort_values(["subject", "task", "run", "sample"]).copy()
    if contours_per_run is not None:
        per_run = (contours_per_run.rename(columns={"n_contours": "nrun"})[["subject", "task", "run", "nrun"]]
                   .sort_values(["subject", "task", "run"]).reset_index(drop=True))
    else:
        per_run = df.groupby(["subject", "task", "run"])["n_contours_in_run"].first().rename("nrun").reset_index()
    per_run["offset"] = per_run.groupby(["subject", "task"])["nrun"].transform(
        lambda x: x.shift(1, fill_value=0).cumsum()).astype("int64")
    off = df[["subject", "task", "run"]].merge(per_run[["subject", "task", "run", "offset"]],
                                                on=["subject", "task", "run"], how="left")["offset"].to_numpy()
    df["n_contours_before_in_child"] = (df["n_contours_before_in_run"].to_numpy(np.int64) + off).astype("int64")
    df["beyond_nominal_contours_child"] = df["n_contours_before_in_child"].to_numpy() > nominal_contours
    return df
