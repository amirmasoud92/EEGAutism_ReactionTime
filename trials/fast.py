"""FAST (face/object shadow-detection oddball) trial builder.

Every FAST stimulus is a row (non-targets 21/22/31/32, shadow targets
121/122/131/132). The shared steps of :class:`~trials.base.TaskTrialBuilder`
give the response matching (RT for targets under the primary 150-1000 ms rule,
bounded by the next stimulus), SOA, press history, lookback flags, joins and
known-issue flags. This module adds the FAST-specific columns.

Block structure (Status-only markers)
-------------------------------------
- A **block start** is a ``201`` followed by a ``202`` (face) or ``203``
  (object) within ``block_structure.pair_max_s`` (0.2 s). Other Status codes
  may lie between the two (e.g. a button-press pulse). The block-start time is
  the ``201`` sample, for the
  FAST columns *and* the base block columns (``block_index_in_run``,
  ``trial_in_block``, ``first_in_block``, ``t_since_block_start_s``):
  :meth:`FastTrialBuilder.block_start_samples` returns these paired starts, so
  one table has one block-start convention (first stimulus 0.273 s after it).
- A ``201`` that does not start a block is a **mid-block pause**.
- ``block_status``: the number of block starts at or before the onset
  (0 = before the first block start).
  ``half_status`` = 1 + number of pauses after the block start and before the
  onset (in block 0 every earlier pause counts).
- ``fast_block`` is the corrected block: marker blocks, further split where
  the stimulus category changes inside a block (fallback for a block start
  whose markers were lost; ``fast_block_start_inferred``). Numbered from 1 in
  each run; a leading unmarked block is block 1. ``fast_half`` counts pauses
  after the block's first stimulus.

Responses
---------
- RT is defined for targets only (``rt_targets_only: true`` in the FAST config;
  the base clears ``rt_ms``, the sensitivity-window RTs, ``responded`` and
  ``response_sample`` on non-target rows). This builder then sets the
  non-target ``outcome`` to ``press`` / ``no_press`` by false-alarm presses.
- A **block-transition press** is a press at most
  ``block_transition_press_max_s`` (0.5 s) before a block start. It is not a
  false alarm.
- ``n_fa_presses``: presses in ``[onset, bound)`` that are neither the row's
  matched target response nor transition presses. ``false_alarm`` is defined
  for non-targets only (NA for targets). Presses before the first stimulus
  belong to no row; their number is ``n_presses_before_first_stim`` (constant
  per run).

Known issues applied here
-------------------------
- ``exclude_events_before_first_marker``: :meth:`FastTrialBuilder.first_marker_sample`
  returns the first *block start* (paired 201 + 202/203), so the base step
  excludes the rows before it (the earliest raw marker would be an IC ``201``
  in the IC-contaminated recording). Those rows are also ``foreign_trigger``.
- ``exclude_partial_blocks``: rows of a partial block (fewer than
  ``nominal_block_size - missing_trigger_tolerance`` stimuli) are
  ``excluded_partial_block``, and ``excluded_known_issue`` is set for them.
- ``flag_only: thirteen_block_session`` -> ``thirteen_block_session``.

Rows are never dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import ClassVar

import numpy as np
import pandas as pd

from trials.base import (
    BuildContext,
    RecordingEvents,
    TaskTrialBuilder,
    load_recording_events,
    nullable_bool,
    nullable_int,
)
from trials.matching import as_sorted_samples
from trials.schema import ColumnSpec
from trials.status import joins_between
from utils.bids_index import Recording
from utils.config import ConfigError, StatusConfig, TaskEventsConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)

FACE_CODES = frozenset({21, 22, 121, 122})
UPRIGHT_CODES = frozenset({21, 31, 121, 131})
PAUSE_CODE = 201
CATEGORY_MARKERS = {202: "face", 203: "object"}

#: Defaults used when ``configs/events.yaml`` (FAST section) does not set them.
DEFAULTS = {
    "block_transition_press_max_s": 0.5,
    "pair_max_s": 0.2,
    "nominal_block_size": 60,
    "missing_trigger_tolerance": 1,
    "lost_trigger_soa_ms": 1400.0,
    "nominal_soa_ms": 1217.0,
    "segment_gap_s": 2.0,
}
#: Levels of the FAST ``nnt`` model factor ({2, 3, 4, 5, 6, 7, >= 8}); counts below 2 (not in the
#: design; the observed ones follow lost triggers) are pooled into the lowest level, labelled ``<=2``.
NNT_LEVELS: tuple[str, ...] = ("<=2", "3", "4", "5", "6", "7", ">=8")
#: ``block_structure`` keys and their types (strict: unknown keys and wrong types raise ConfigError).
BLOCK_STRUCTURE_KEYS: dict[str, type] = {"pair_max_s": float, "nominal_block_size": int,
                                         "missing_trigger_tolerance": int, "lost_trigger_soa_ms": float,
                                         "nominal_soa_ms": float, "segment_gap_s": float}


def _param(v, typ: type, where: str):
    """Strictly typed parameter value (bools are not numbers; ints are accepted for floats)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (typ is int and not isinstance(v, int)):
        raise ConfigError(f"{where}: expected {'an integer' if typ is int else 'a number'}, got {v!r}")
    return typ(v)


def _c(name: str, dtype: str, nullable: bool = False, description: str = "") -> ColumnSpec:
    return ColumnSpec(name, dtype, nullable, description)


_nint = nullable_int
_nbool = nullable_bool


def _nstr(a: Sequence, valid: np.ndarray) -> pd.arrays.StringArray:
    return pd.array([x if v else pd.NA for x, v in zip(a, valid)], dtype="string")


# --------------------------------------------------------------------------- marker parsing
def parse_block_markers(markers: pd.DataFrame, sfreq: float,
                        pair_max_s: float = DEFAULTS["pair_max_s"]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split FAST Status markers into block starts and mid-block pauses.

    Parameters
    ----------
    markers
        Marker rows (``sample``, ``value``) with codes 201/202/203; other codes
        are ignored.
    sfreq
        Sampling rate.
    pair_max_s
        A 201 starts a block if a 202/203 follows it within this time
        (strictly less). Codes between them are
        allowed.

    Returns
    -------
    block_start_samples : ndarray of int64
        Sorted samples of block starts (the 201 sample; for a 202/203 without
        a preceding 201, its own sample).
    block_categories : ndarray of object
        ``"face"`` / ``"object"`` per block start.
    pause_samples : ndarray of int64
        Sorted samples of the 201s that do not start a block.
    """
    m = markers.sort_values(["sample", "value"], kind="stable")
    s = m["sample"].to_numpy(np.int64)
    v = m["value"].to_numpy(np.int64)
    s201 = s[v == PAUSE_CODE]
    used = np.zeros(s201.size, bool)
    starts: list[int] = []
    cats: list[str] = []
    lim = pair_max_s * sfreq
    for smp, val in zip(s[np.isin(v, list(CATEGORY_MARKERS))], v[np.isin(v, list(CATEGORY_MARKERS))]):
        i = int(np.searchsorted(s201, smp, side="right")) - 1   # latest 201 at or before the category marker
        if i >= 0 and not used[i] and (smp - s201[i]) < lim:
            used[i] = True
            starts.append(int(s201[i]))
        else:
            log.warning("category marker %s at sample %s without a preceding 201; used as block start", val, smp)
            starts.append(int(smp))
        cats.append(CATEGORY_MARKERS[int(val)])
    order = np.argsort(np.asarray(starts, np.int64), kind="stable")
    return (np.asarray(starts, np.int64)[order], np.asarray(cats, dtype=object)[order],
            as_sorted_samples(s201[~used]))


def transition_press_mask(press_samples: np.ndarray, block_starts: np.ndarray, sfreq: float,
                          max_s: float = DEFAULTS["block_transition_press_max_s"]) -> np.ndarray:
    """Presses at most ``max_s`` before (or at) the next block start."""
    p = as_sorted_samples(press_samples)
    bs = as_sorted_samples(block_starts)
    if p.size == 0 or bs.size == 0:
        return np.zeros(p.size, bool)
    i = np.searchsorted(bs, p, side="left")
    nxt = np.where(i < bs.size, bs[np.minimum(i, bs.size - 1)], -1)
    lag = nxt - p
    return (nxt >= 0) & (lag >= 0) & (lag <= max_s * sfreq + 1e-9)


def nontargets_since_target_in_block(is_target: np.ndarray, block_index: pd.Series | np.ndarray) -> np.ndarray:
    """Non-target rows strictly between max(previous target, first row of the block) and each row.

    Parameters
    ----------
    is_target
        Target flag per row of one run (in onset order).
    block_index
        Block of each row (``block_index_in_run``); NA anywhere means blocks
        are unknown and the count resets at the run start only.

    Returns
    -------
    ndarray of int64
    """
    tgt = np.asarray(is_target, bool)
    n = tgt.size
    if n == 0:
        return np.array([], np.int64)
    blk = pd.Series(block_index)
    key = blk.to_numpy(np.int64) if blk.notna().all() else np.ones(n, np.int64)
    out = np.zeros(n, np.int64)
    count = 0
    for k in range(n):
        if k > 0 and key[k] != key[k - 1]:
            count = 0                     # block start: the hazard clock restarts
        out[k] = count
        count = 0 if tgt[k] else count + 1
    return out


def nnt_levels(count: np.ndarray) -> np.ndarray:
    """Factor level of each ``nnt`` count (:data:`NNT_LEVELS`)."""
    c = np.asarray(count, np.int64)
    out = np.where(c <= 2, "<=2", np.where(c >= 8, ">=8", c.astype(str))).astype(object)
    return out


# --------------------------------------------------------------------------- builder
class FastTrialBuilder(TaskTrialBuilder):
    """Trial table of one FAST recording (see module docstring for the rules)."""

    task: ClassVar[str] = "FAST"

    def __init__(self, task_cfg: TaskEventsConfig, status_cfg: StatusConfig,
                 registry: KnownIssuesRegistry | None = None) -> None:
        super().__init__(task_cfg, status_cfg, registry)
        ex = task_cfg.extra
        w = f"tasks.{task_cfg.name}"
        bs = ex.get("block_structure") or {}
        if not isinstance(bs, dict):
            raise ConfigError(f"{w}.block_structure must be a mapping")
        bad = set(bs) - set(BLOCK_STRUCTURE_KEYS)
        if bad:
            raise ConfigError(f"{w}.block_structure: unknown keys {sorted(bad)}; allowed {sorted(BLOCK_STRUCTURE_KEYS)}")
        par = {k: _param(bs.get(k, DEFAULTS[k]), t, f"{w}.block_structure.{k}") for k, t in BLOCK_STRUCTURE_KEYS.items()}
        self.transition_max_s = _param(ex.get("block_transition_press_max_s", DEFAULTS["block_transition_press_max_s"]),
                                       float, f"{w}.block_transition_press_max_s")
        self.pair_max_s = par["pair_max_s"]
        self.nominal_block_size = par["nominal_block_size"]
        self.missing_trigger_tolerance = par["missing_trigger_tolerance"]
        self.lost_trigger_soa_ms = par["lost_trigger_soa_ms"]
        self.nominal_soa_ms = par["nominal_soa_ms"]
        self.segment_gap_s = par["segment_gap_s"]
        self.truncated_post_s = task_cfg.post_stim_required_s
        if self.pair_max_s <= 0 or self.nominal_block_size < 1 or self.nominal_soa_ms <= 0 or self.segment_gap_s <= 0:
            raise ConfigError(f"{w}.block_structure: pair_max_s, nominal_block_size, nominal_soa_ms and "
                              "segment_gap_s must be > 0")
        if not task_cfg.rt_targets_only:
            raise ConfigError("FAST config must set rt_targets_only: true (RT is defined for targets only)")

    def block_start_samples(self, ctx: BuildContext) -> np.ndarray | None:
        """Block starts for the base block columns: the 201 sample of each 201 + 202/203 pair.

        Same convention as ``block_status`` / ``fast_block`` (``parse_block_markers``),
        so the base ``t_since_block_start_s`` of a block's first stimulus is
        about 0.273 s, not the 0.223 s from the 202/203 sample.
        """
        if ctx.markers is None:
            return None
        starts, _, _ = parse_block_markers(ctx.markers, ctx.sfreq, self.pair_max_s)
        return starts

    def first_marker_sample(self, ctx: BuildContext, marker_codes: Iterable[int]) -> int | None:
        """First FAST block start (201 paired with 202/203), not the earliest raw marker."""
        if ctx.markers is None:
            return None
        bstarts, _, _ = parse_block_markers(ctx.markers, ctx.sfreq, self.pair_max_s)
        return int(bstarts[0]) if bstarts.size else int(np.iinfo(np.int64).max)

    # ------------------------------------------------------------------ schema
    def extra_columns(self) -> tuple[ColumnSpec, ...]:
        cols = [
            _c("category", "str", description="face / object"),
            _c("orientation", "str", description="upright / inverted"),
            _c("prev_value", "Int", True, "code of the previous stimulus in the run"),
            _c("prev_category", "str", True, "category (face/object) of the previous stimulus in the run; NA for the first"),
            _c("prev_orientation", "str", True, "orientation (upright/inverted) of the previous stimulus; NA for the first"),
            _c("prev_is_target", "boolean", True, "previous stimulus in the run was a target; NA for the first"),
            _c("n_nontargets_since_prev_target", "Int", True,
               "non-targets strictly between the previous target of the run and this onset; NA if none. Counts "
               "across block starts, pauses and joins"),
            _c("n_nontargets_since_prev_target_in_block", "int", description=
               "non-targets strictly between max(previous target, start of the base block) and this onset: the "
               "target-hazard count, reset at every block start; without markers it resets at run start only"),
            _c("nnt_count", "int", description="model value of the nnt covariate: "
               "n_nontargets_since_prev_target (across block starts), or, for the first target of "
               "a run (no previous target; NA there), n_nontargets_since_prev_target_in_block (non-targets since the "
               "block start)"),
            _c("nnt_from_block_start", "bool", description="nnt_count came from the in-block count because "
               "n_nontargets_since_prev_target is NA (first target of the run)"),
            _c("nnt_level", "str", description="factor level of nnt_count: '<=2', '3' ... '7', '>=8' (levels "
               "{2, ..., >= 8}; counts 0-1 are not in the design and are pooled into '<=2')"),
            _c("nnt_below_design", "bool", description="nnt_count < 2 (pooled into level '<=2')"),
            _c("t_since_prev_target_ms", "float", True, "onset minus previous target onset; NA if none or across a join"),
            _c("t_since_prev_target_ms_concat", "float", True, "same on the concatenated sample clock"),
            # Status block structure
            _c("block_status", "Int", True, "block starts (201+202/203) at or before onset; 0 = before the first"),
            _c("block_category_status", "str", True, "face/object from the block's 202/203 marker"),
            _c("half_status", "Int", True, "1 + mid-block pauses (lone 201) after the block start"),
            _c("before_first_block_marker", "boolean", True, "block_status == 0: onset precedes the first 201+202/203 "
               "block start; NA without markers"),
            _c("category_matches_block", "boolean", True, "stimulus category equals the block marker category"),
            _c("fast_block", "Int", True, "corrected block in run (1-based; marker blocks split at category changes)"),
            _c("fast_block_start_inferred", "boolean", True, "block started at a category change without a marker"),
            _c("fast_block_unmarked", "boolean", True, "block has no block-start marker (leading or inferred)"),
            _c("fast_block_category", "str", True, "marker category, else the majority stimulus category"),
            _c("fast_block_size", "Int", True, "stimuli in the corrected block"),
            _c("fast_trial_in_block", "Int", True, "1-based position in the corrected block"),
            _c("fast_half", "Int", True, "1 + pauses after the block's first stimulus and before onset"),
            _c("first_after_break", "boolean", True, "first stimulus of a (corrected) block or half-block"),
            _c("first_in_segment", "bool", description="first stimulus of an events-only segment: first stimulus "
               "of the run, or the SOA to the "
               "previous stimulus on the concatenated clock exceeds block_structure.segment_gap_s (2.0 s), or a "
               "bit-16 join lies in between. Catches block starts, pauses and lost-trigger gaps without markers"),
            _c("soa_prev_within_half_ms", "float", True, "SOA if the previous stimulus is in the same half; NA across joins"),
            _c("n_lost_triggers_in_block", "Int", True,
               "stimuli hidden in within-half SOA gaps (round(SOA / nominal SOA) - 1 for SOA > lost_trigger_soa_ms)"),
            _c("partial_block", "boolean", True, "block size + lost triggers < nominal - missing-trigger tolerance"),
            _c("restarted_block", "boolean", True, "partial block followed by a block of the same category"),
            _c("excluded_partial_block", "bool", description="partial block in a recording with exclude_partial_blocks"),
            _c("n_full_blocks_in_run", "Int", True, "corrected blocks that are not partial"),
            # responses
            _c("n_transition_presses", "Int", True, "block-transition presses in [onset, bound)"),
            _c("n_fa_presses", "Int", True, "presses in [onset, bound) that are not the matched hit nor transition presses"),
            _c("false_alarm", "boolean", True, "non-target with >= 1 FA press; NA for targets"),
            _c("prev_press_is_transition", "boolean", True, "the last press before onset is a block-transition press"),
            _c("n_presses_before_first_stim", "int", description="presses before the first stimulus of the run"),
        ]
        cols += [_c(f"n_press_in_lookback_{L}", "int", description=f"presses in [t0-{L} ms, t0)")
                 for L in self.cfg.lookbacks_ms]
        cols += [
            _c("foreign_trigger", "bool", description="before the first block start in a recording with foreign triggers"),
            _c("thirteen_block_session", "bool", description="registry flag: 13 complete blocks (duplicate unresolved)"),
            _c("restarted_block_session", "bool", description="recording has an exclude_partial_blocks registry entry"),
        ]
        return tuple(cols)

    # ------------------------------------------------------------------ hook
    def add_task_columns(self, trials: pd.DataFrame, rec: RecordingEvents, ctx: BuildContext) -> pd.DataFrame:
        df = trials
        n = len(df)
        s = ctx.trial_samples
        vals = df["value"].to_numpy(np.int64)
        tgt = df["is_target"].to_numpy(bool)
        sf = ctx.sfreq
        face = np.isin(vals, list(FACE_CODES))
        cat = np.where(face, "face", "object").astype(object)
        ori = np.where(np.isin(vals, list(UPRIGHT_CODES)), "upright", "inverted").astype(object)
        df["category"] = pd.array(cat, dtype="string")
        df["orientation"] = pd.array(ori, dtype="string")

        # ---------------- previous stimulus / previous target
        has_prev = np.arange(n) > 0
        pv = np.r_[0, vals[:-1]] if n else np.array([], np.int64)
        df["prev_value"] = _nint(pv, has_prev)
        df["prev_category"] = _nstr(np.r_[[None], cat[:-1]] if n else [], has_prev)
        df["prev_orientation"] = _nstr(np.r_[[None], ori[:-1]] if n else [], has_prev)
        df["prev_is_target"] = _nbool(np.r_[False, tgt[:-1]] if n else np.array([], bool), has_prev)
        ti = np.flatnonzero(tgt)
        k_prev = np.searchsorted(ti, np.arange(n), side="left") - 1          # index into ti of previous target
        has_pt = k_prev >= 0
        j = np.where(has_pt, ti[np.maximum(k_prev, 0)] if ti.size else 0, -1)
        cum_nt = np.cumsum(~tgt)
        before = np.r_[0, cum_nt[:-1]] if n else np.array([], np.int64)     # non-targets before row k
        nnt = before - np.where(has_pt, cum_nt[np.maximum(j, 0)] if n else 0, 0)
        df["n_nontargets_since_prev_target"] = _nint(nnt, has_pt)
        df["n_nontargets_since_prev_target_in_block"] = nontargets_since_target_in_block(
            tgt, df["block_index_in_run"])
        nib = df["n_nontargets_since_prev_target_in_block"].to_numpy(np.int64)
        nnt_c = np.where(has_pt, nnt, nib).astype(np.int64)
        df["nnt_count"] = nnt_c
        df["nnt_from_block_start"] = ~has_pt
        df["nnt_level"] = pd.array(nnt_levels(nnt_c), dtype="string")
        df["nnt_below_design"] = nnt_c < 2
        prev_t_sample = np.where(has_pt, s[np.maximum(j, 0)] if n else 0, -1)
        concat = np.where(has_pt, (s - prev_t_sample) * 1000.0 / sf, np.nan)
        if ctx.joins is None:
            cross = np.zeros(n, bool)
        else:
            cross = has_pt & (joins_between(ctx.joins, np.where(has_pt, prev_t_sample, 0), s) > 0)
        df["t_since_prev_target_ms_concat"] = concat
        df["t_since_prev_target_ms"] = np.where(cross, np.nan, concat)

        # ---------------- events-only segments (first_in_segment)
        soa_c = df["soa_prev_ms_concat"].to_numpy(float)
        cross = df["soa_prev_crosses_join"].fillna(False).to_numpy(bool)
        df["first_in_segment"] = (~np.isfinite(soa_c)) | (soa_c > self.segment_gap_s * 1000.0 + 1e-9) | cross

        # ---------------- block structure
        issues = self.registry.for_recording(rec.subject, rec.task, rec.run) if self.registry else []
        acts = [(i, a) for i in issues for a in i.actions]
        partial_actions = [a for _, a in acts if a.type == "exclude_partial_blocks"]
        if ctx.markers is not None:
            bstarts, bcats, pauses = parse_block_markers(ctx.markers, sf, self.pair_max_s)
        else:
            bstarts = bcats = pauses = None
        self._add_block_columns(df, ctx, cat, bstarts, bcats, pauses, partial_actions)

        # ---------------- presses: transition / false alarms / lookback counts
        p = ctx.press_samples
        if bstarts is not None:
            trans = transition_press_mask(p, bstarts, sf, self.transition_max_s)
        else:
            trans = np.zeros(p.size, bool)
        row = np.searchsorted(s, p, side="right") - 1                          # press at onset -> that trial
        bound = df["bound_sample"].fillna(-1).to_numpy(np.int64)
        in_row = (row >= 0) & ((bound[np.maximum(row, 0)] < 0) | (p < bound[np.maximum(row, 0)])) if n else \
            np.zeros(p.size, bool)
        resp_s = df["response_sample"].fillna(-1).to_numpy(np.int64)
        matched = np.zeros(p.size, bool)
        hit_resp = resp_s[tgt & (resp_s >= 0)]
        matched[np.searchsorted(p, hit_resp, side="left")] = True           # the matched press itself
        n_tr = np.zeros(n, np.int64)
        n_fa = np.zeros(n, np.int64)
        np.add.at(n_tr, row[in_row & trans & ~matched], 1)
        np.add.at(n_fa, row[in_row & ~trans & ~matched], 1)
        ok = np.full(n, bstarts is not None)
        df["n_transition_presses"] = _nint(n_tr, ok)
        df["n_fa_presses"] = _nint(n_fa, ok)
        df["false_alarm"] = _nbool(n_fa > 0, ok & ~tgt)
        pps = ctx.extra.get("prev_press_sample", np.full(n, -1, np.int64))
        trans_set = p[trans]
        df["prev_press_is_transition"] = _nbool(np.isin(pps, trans_set), ok & (pps >= 0))
        df["n_presses_before_first_stim"] = np.full(n, int(np.sum(p < s[0])) if n else 0, np.int64)
        pf = p.astype(float)
        for L in self.cfg.lookbacks_ms:
            lo = s - L * sf / 1000.0 - 1e-9
            df[f"n_press_in_lookback_{L}"] = (np.searchsorted(p, s, side="left")
                                              - np.searchsorted(pf, lo, side="left")).astype(np.int64)

        # ---------------- foreign triggers, session flags (recording end: base t_to_recording_end_s, post_stim_truncated)
        foreign = np.zeros(n, bool)
        if any(a.type == "exclude_events_before_first_marker" for _, a in acts):
            first = self.first_marker_sample(ctx, ())
            foreign = np.ones(n, bool) if first is None else s < first   # same rows the base step excluded
        df["foreign_trigger"] = foreign
        flags = {str(a.get("flag")) for _, a in acts if a.type == "flag_only"}
        df["thirteen_block_session"] = np.full(n, "thirteen_block_session" in flags)
        df["restarted_block_session"] = np.full(n, bool(partial_actions))
        self._nontarget_outcome(df, tgt, n_fa, ok)
        df["excluded_known_issue"] = (df["excluded_known_issue"].to_numpy(bool)
                                      | df["excluded_partial_block"].to_numpy(bool))
        return df

    # ------------------------------------------------------------------ non-target outcome
    def _nontarget_outcome(self, df: pd.DataFrame, tgt: np.ndarray, n_fa: np.ndarray, ok: np.ndarray) -> None:
        """Non-target ``outcome`` from false-alarm presses.

        The base (``rt_targets_only``) has already cleared the RT columns of
        non-target rows. ``outcome`` becomes ``press`` if the row has a
        false-alarm press (any press in ``[onset, bound)`` except transition
        presses; any press at all when Status markers are unavailable), else
        ``no_press``.
        """
        nt = ~tgt
        if not nt.any():
            return
        pressed = np.where(ok, n_fa > 0, df["n_press_until_bound"].to_numpy(np.int64) > 0)
        out = df["outcome"].to_numpy(dtype=object).copy()
        out[nt] = np.where(pressed[nt], "press", "no_press")
        df["outcome"] = pd.array(out, dtype="string")

    # ------------------------------------------------------------------ blocks
    def _add_block_columns(self, df: pd.DataFrame, ctx: BuildContext, cat: np.ndarray,
                           bstarts: np.ndarray | None, bcats: np.ndarray | None, pauses: np.ndarray | None,
                           partial_actions: list) -> None:
        n = len(df)
        s = ctx.trial_samples
        names_int = ("block_status", "half_status", "fast_block", "fast_block_size", "fast_trial_in_block",
                     "fast_half", "n_full_blocks_in_run")
        names_bool = ("before_first_block_marker", "category_matches_block", "fast_block_start_inferred",
                      "fast_block_unmarked", "first_after_break", "partial_block", "restarted_block")
        if bstarts is None or n == 0:
            for c in names_int:
                df[c] = pd.array([pd.NA] * n, dtype="Int64")
            for c in names_bool:
                df[c] = pd.array([pd.NA] * n, dtype="boolean")
            for c in ("block_category_status", "fast_block_category"):
                df[c] = pd.array([pd.NA] * n, dtype="string")
            df["soa_prev_within_half_ms"] = np.full(n, np.nan)
            df["n_lost_triggers_in_block"] = pd.array([pd.NA] * n, dtype="Int64")
            df["excluded_partial_block"] = np.zeros(n, bool)
            if partial_actions and n:
                log.warning("exclude_partial_blocks needs Status markers; not applied (%s)", df["trial_key"].iloc[0])
            return
        allv = np.ones(n, bool)
        b = np.searchsorted(bstarts, s, side="right")                       # block_status
        df["block_status"] = _nint(b, allv)
        bc = np.array([bcats[k - 1] if k > 0 else None for k in b], dtype=object)
        df["block_category_status"] = _nstr(bc, b > 0)
        b0 = np.where(b > 0, bstarts[np.maximum(b - 1, 0)] if bstarts.size else 0, np.iinfo(np.int64).min)
        # pauses strictly after the block start and strictly before the onset
        half = 1 + (np.searchsorted(pauses, s, side="left") - np.searchsorted(pauses, b0, side="right"))
        df["half_status"] = _nint(half, allv)
        df["before_first_block_marker"] = _nbool(b == 0, allv)
        df["category_matches_block"] = _nbool(cat == bc, b > 0)

        # corrected blocks: split marker blocks at category changes
        change = np.r_[False, (cat[1:] != cat[:-1]) & (b[1:] == b[:-1])]
        new_block = np.r_[True, b[1:] != b[:-1]]
        inferred = change
        seg_start = new_block | inferred
        seg = np.cumsum(seg_start)                                            # 1-based
        if inferred.any():
            log.warning("%s: %d FAST block start(s) inferred from a category change without a marker",
                        df["trial_key"].iloc[0], int(inferred.sum()))
        first_idx = np.flatnonzero(seg_start)
        seg_first_row = first_idx[seg - 1]
        sizes = np.bincount(seg)[seg]
        df["fast_block"] = _nint(seg, allv)
        df["fast_block_start_inferred"] = _nbool(inferred[seg_first_row], allv)
        unmarked = inferred[seg_first_row] | (b[seg_first_row] == 0)
        df["fast_block_unmarked"] = _nbool(unmarked, allv)
        seg_cat = []
        for k in range(1, seg.max() + 1):
            rows = np.flatnonzero(seg == k)
            r0 = rows[0]
            if b[r0] > 0 and not inferred[r0]:
                seg_cat.append(bcats[b[r0] - 1])
            else:
                vals, cnt = np.unique(cat[rows], return_counts=True)
                seg_cat.append(vals[np.argmax(cnt)])
        seg_cat = np.asarray(seg_cat, dtype=object)
        df["fast_block_category"] = pd.array(seg_cat[seg - 1], dtype="string")
        df["fast_block_size"] = _nint(sizes, allv)
        df["fast_trial_in_block"] = _nint(np.arange(n) - seg_first_row + 1, allv)
        ref = s[seg_first_row]
        fh = 1 + (np.searchsorted(pauses, s, side="left") - np.searchsorted(pauses, ref, side="right"))
        df["fast_half"] = _nint(fh, allv)
        brk = np.r_[True, (seg[1:] != seg[:-1]) | (fh[1:] != fh[:-1])]
        df["first_after_break"] = _nbool(brk, allv)
        soa = df["soa_prev_ms"].to_numpy(float)
        df["soa_prev_within_half_ms"] = np.where(brk, np.nan, soa)

        # lost stimulus triggers: a within-half SOA of k nominal SOAs hides k - 1 stimuli
        soa_wh = df["soa_prev_within_half_ms"].to_numpy(float)
        gap = np.isfinite(soa_wh) & (soa_wh > self.lost_trigger_soa_ms)
        lost_gap = np.where(gap, np.maximum(np.rint(np.where(gap, soa_wh, 0.0) / self.nominal_soa_ms) - 1, 0),
                            0).astype(np.int64)
        lost = np.bincount(seg, weights=lost_gap).astype(np.int64)[seg]
        df["n_lost_triggers_in_block"] = _nint(lost, allv)
        thr = self.nominal_block_size - self.missing_trigger_tolerance
        part = (sizes + lost) < thr
        df["partial_block"] = _nbool(part, allv)
        nseg = int(seg.max())
        seg_partial = part[first_idx]
        restarted = np.zeros(nseg, bool)
        for k in range(nseg - 1):
            restarted[k] = seg_partial[k] and seg_cat[k] == seg_cat[k + 1]
        df["restarted_block"] = _nbool(restarted[seg - 1], allv)
        df["n_full_blocks_in_run"] = _nint(np.full(n, int((~seg_partial).sum())), allv)
        excl = np.zeros(n, bool)
        for a in partial_actions:
            nominal = int(a.get("nominal_block_size", self.nominal_block_size))
            excl |= (sizes + lost) < nominal - self.missing_trigger_tolerance
        df["excluded_partial_block"] = excl


# --------------------------------------------------------------------------- convenience
def build_fast_recording(recording: Recording, task_cfg: TaskEventsConfig, status_cfg: StatusConfig,
                         sfreq: float, registry: KnownIssuesRegistry | None = None,
                         read_status: bool = True) -> pd.DataFrame:
    """Load one FAST recording (events.tsv + Status) and build its trial table."""
    rec = load_recording_events(recording, task_cfg, status_cfg, sfreq, read_status=read_status)
    return FastTrialBuilder(task_cfg, status_cfg, registry).build(rec)
