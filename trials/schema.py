"""Output schemas for trial tables, validated before anything is written.

dtype vocabulary
----------------
``int``          non-nullable int64
``Int``          nullable Int64
``float``        float64 (NaN allowed when ``nullable``)
``bool``         non-nullable bool
``boolean``      nullable boolean
``str``          string (object or pandas string dtype)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from utils.config import TaskEventsConfig

_DTYPES = {"int", "Int", "float", "bool", "boolean", "str"}


class SchemaError(ValueError):
    """Raised when a table does not match its schema."""


@dataclass(frozen=True)
class ColumnSpec:
    """One column of a table schema."""

    name: str
    dtype: str
    nullable: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r} for column {self.name}")


@dataclass(frozen=True)
class TableSchema:
    """Ordered set of columns.

    Parameters
    ----------
    name
        Schema name.
    columns
        Column specs (required columns, in output order).
    allow_extra
        Whether columns outside the schema are tolerated (they are appended).
    """

    name: str
    columns: tuple[ColumnSpec, ...]
    allow_extra: bool = False
    keys: tuple[str, ...] = field(default=())

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def extend(self, cols: Iterable[ColumnSpec], name: str | None = None) -> TableSchema:
        """New schema with extra columns appended (names must be new)."""
        extra = tuple(cols)
        clash = set(self.names) & {c.name for c in extra}
        if clash:
            raise ValueError(f"columns already in schema: {sorted(clash)}")
        return TableSchema(name or self.name, self.columns + extra, self.allow_extra, self.keys)

    # ------------------------------------------------------------------ validation
    def validate(self, df: pd.DataFrame) -> None:
        """Raise :class:`SchemaError` listing every violation."""
        problems: list[str] = []
        missing = [c for c in self.names if c not in df.columns]
        if missing:
            problems.append(f"missing columns: {missing}")
        extra = [c for c in df.columns if c not in self.names]
        if extra and not self.allow_extra:
            problems.append(f"unexpected columns: {extra}")
        if df.columns.duplicated().any():
            problems.append(f"duplicated columns: {df.columns[df.columns.duplicated()].tolist()}")
        for spec in self.columns:
            if spec.name not in df.columns:
                continue
            s = df[spec.name]
            err = _check(spec, s)
            if err:
                problems.append(f"{spec.name}: {err}")
        if self.keys and all(k in df.columns for k in self.keys) and df.duplicated(list(self.keys)).any():
            problems.append(f"duplicate keys {list(self.keys)}")
        if problems:
            raise SchemaError(f"schema {self.name!r}: " + "; ".join(problems))

    def conform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce dtypes, order columns, then validate."""
        out = df.copy()
        for spec in self.columns:
            if spec.name in out.columns:
                out[spec.name] = _coerce(spec, out[spec.name])
        order = self.names + [c for c in out.columns if c not in self.names]
        out = out[[c for c in order if c in out.columns]]
        self.validate(out)
        return out


def _check(spec: ColumnSpec, s: pd.Series) -> str | None:
    na = s.isna()
    if not spec.nullable and na.any() and spec.dtype not in ("float",):
        return f"{int(na.sum())} missing values in non-nullable column"
    if spec.dtype == "float" and not spec.nullable and na.any():
        return f"{int(na.sum())} NaN values in non-nullable float column"
    dt = s.dtype
    if spec.dtype == "int" and str(dt) != "int64":
        return f"dtype {dt} != int64"
    if spec.dtype == "Int" and str(dt) not in ("Int64", "int64"):
        return f"dtype {dt} != Int64"
    if spec.dtype == "float" and not pd.api.types.is_float_dtype(dt):
        return f"dtype {dt} != float64"
    if spec.dtype == "bool" and str(dt) != "bool":
        return f"dtype {dt} != bool"
    if spec.dtype == "boolean" and str(dt) not in ("boolean", "bool"):
        return f"dtype {dt} != boolean"
    if spec.dtype == "str" and not (pd.api.types.is_string_dtype(dt) or dt == object):
        return f"dtype {dt} is not a string dtype"
    return None


def _coerce(spec: ColumnSpec, s: pd.Series) -> pd.Series:
    try:
        if spec.dtype == "int":
            return s.astype("int64")
        if spec.dtype == "Int":
            return s.astype("Int64")
        if spec.dtype == "float":
            return pd.to_numeric(s, errors="raise").astype("float64")
        if spec.dtype == "bool":
            return s.astype("bool") if not s.isna().any() else s
        if spec.dtype == "boolean":
            return s.astype("boolean")
        if spec.dtype == "str":
            return s.astype("string")
    except (TypeError, ValueError) as e:
        raise SchemaError(f"cannot coerce {spec.name} to {spec.dtype}: {e}") from e
    return s


# --------------------------------------------------------------------------- trial-table schema
def _c(name: str, dtype: str, nullable: bool = False, description: str = "") -> ColumnSpec:
    return ColumnSpec(name, dtype, nullable, description)


BASE_TRIAL_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("subject", "str", description="BIDS participant_id"),
    _c("task", "str", description="BIDS task label"),
    _c("run", "int", description="BIDS run number (1 when the file name has no run entity)"),
    _c("sample", "int", description="stimulus onset sample on the recording's concatenated sample clock (key)"),
    _c("onset_s", "float", description="stimulus onset from events.tsv (= sample / sfreq, checked exactly)"),
    _c("trial_index_in_run", "int", description="1-based position among the trial rows of the run"),
    _c("trial_key", "str", description="stable key subject_task_run-XX_s<sample>"),
    _c("value", "int", description="stimulus code after known-issue recoding (recode_event)"),
    _c("value_original", "int", description="stimulus code as logged in events.tsv"),
    _c("condition", "str", description="label of value from stimulus_codes in configs/events.yaml"),
    _c("is_target", "bool", description="value is in target_codes (a response is expected)"),
    # response matching
    _c("bound_sample", "Int", True, "onset of the next response_bound event of the run, which closes the response "
       "window; NA after the last one"),
    _c("t_to_bound_ms", "float", True, "bound_sample minus onset; NA without a bound"),
    _c("n_press_until_bound", "int", description="presses in [onset, bound) (any latency)"),
    _c("first_press_lat_ms", "float", True, "latency of the first press in [onset, bound); NA if none"),
    _c("raw_first_press_lat_ms", "float", True, "latency of the first press at/after onset, ignoring window and bound"),
    _c("raw_first_press_after_bound", "boolean", True, "the raw first press lies at/after bound_sample; NA if no press"),
    _c("n_anticipations", "int", description="presses in [onset, bound) with latency below the primary window lo"),
    _c("n_late_presses", "int", description="presses in [onset, bound) with latency above the primary window hi"),
    _c("rt_ms", "float", True, "primary-window RT: first press in [onset, bound) with latency in [lo, hi] "
       "(closed); NA if none (and on non-targets when rt_targets_only)"),
    _c("responded", "bool", description="rt_ms is defined (a primary-window response was matched)"),
    _c("response_sample", "Int", True, "sample of the matched primary-window press; NA if none"),
    _c("n_extra_presses", "int", description="presses after the matched one, before bound, at any latency "
       "(including late presses > hi; see n_extra_in_window)"),
    _c("n_extra_in_window", "int", description="presses after the matched one whose latency is also in the primary "
       "window (= n_valid_window_presses - 1 on responded rows); 0 if not responded"),
    _c("outcome", "str", description="hit/miss on targets; press/no_press on non-targets"),
    # generic covariates
    _c("soa_prev_ms", "float", True, "time since the previous stimulus of any stimulus code in the run; NA for the "
       "first stimulus or when a bit-16 join intervenes"),
    _c("soa_prev_ms_concat", "float", True, "same on the concatenated sample clock (ignores joins)"),
    _c("soa_prev_crosses_join", "boolean", True, "a bit-16 join lies in (previous stimulus, onset]; NA if no previous "
       "stimulus or joins unknown"),
    _c("soa_stratum", "str", True, "stratum of soa_prev_ms by the task's soa_strata_ms edges: '[a,b)', '<lo' or "
       "'>=hi'; NA when soa_prev_ms is NA or no edges are configured"),
    _c("t_since_prev_trial_ms", "float", True, "time since the previous trial row of the run; NA if none or across a "
       "join"),
    _c("t_since_prev_trial_ms_concat", "float", True, "same on the concatenated sample clock"),
    _c("prev_trial_crosses_join", "boolean", True, "a join lies between the previous trial row and onset"),
    _c("t_since_prev_press_ms", "float", True, "time since the last press strictly before onset; NA if none or "
       "across a join"),
    _c("t_since_prev_press_ms_concat", "float", True, "same on the concatenated sample clock"),
    _c("prev_press_crosses_join", "boolean", True, "a join lies between the previous press and onset"),
    _c("prev_outcome", "str", description="outcome of the previous logged target row of the run: hit / miss / none. "
       "Across a dropped-trigger gap, a block boundary or a join it can refer to stimulus n-2 or to the previous "
       "block: use prev_outcome_valid"),
    _c("prev_rt_ms", "float", True, "primary-window RT of the previous target of the run; NA after a miss or none "
       "(same caveat as prev_outcome)"),
    _c("prev_outcome_valid", "bool", description="prev_outcome refers to the immediately preceding target of the same "
       "block with no join in between (AVSRT: = prev_stim_in_segment, which also excludes dropped-trigger gaps and "
       "pauses); False when joins are unknown"),
    _c("prev_target_crosses_join", "boolean", True, "a join lies between the previous target and onset"),
    _c("prev_trial_no_press", "boolean", True, "the previous trial row had no press in its [onset, bound); NA for the "
       "first row"),
    _c("time_in_run_s", "float", description="onset minus the first trial onset of the run (concatenated clock)"),
    _c("first_in_run", "bool", description="first trial row of the run"),
    _c("last_in_run", "bool", description="last trial row of the run"),
    # blocks from markers
    _c("markers_available", "bool", description="Status markers were read for this recording"),
    _c("block_index_in_run", "Int", True, "1 + block-start markers at or before onset, in every run (rows before the "
       "first marker form block 1, which is empty when the run starts with a marker); FAST block starts are the 201 "
       "sample of each 201 + 202/203 pair; NA without markers"),
    _c("trial_in_block", "Int", True, "1-based position of the row in its block"),
    _c("block_start_unmarked", "boolean", True, "the row's block starts before the first marker (no marker)"),
    _c("t_since_block_start_s", "float", True, "onset minus the last block-start marker (FAST: the 201 of the pair); "
       "NA in the unmarked block"),
    _c("first_in_block", "bool", description="first trial row of its block (first_in_run without markers)"),
    # joins
    _c("joins_available", "bool", description="Status bit-16 joins were read for this recording"),
    _c("n_joins_before", "Int", True, "bit-16 joins at or before onset"),
    _c("t_since_last_join_s", "float", True, "onset minus the last join at or before it; NA if none"),
    _c("t_to_next_join_s", "float", True, "next join after onset minus onset; NA if none"),
    _c("join_near_window", "bool", description="a join lies in [onset - join lookback - join_padding_ms, onset + "
       "join_post_ms] (join lookback: join_near_lookback_ms, default the longest lookback). A coarse proxy; the "
       "edge-validity rule used for the windows is edge_valid_<window>"),
    # flags
    _c("history_homogeneous", "bool", description="previous target outcome as configured (hit), last press within "
       "the task's history_homogeneous.press_lag_ms before onset (AVSRT 1000-3000, FAST 3000-9500, IC 2500-11000 ms) "
       "and that press is the previous target's matched response, no join since the previous target (per-task lag "
       "window)"),
    _c("flag_rt_fast_after_no_press", "boolean", True, "responded with RT < misattribution fast_rt_ms (200) after a "
       "trial row with no press (possible misattributed late response); NA where misattribution.applies is false "
       "(FAST, IC: the previous row expects no response, or is seconds away)"),
    # row-level event integrity
    _c("in_status", "boolean", True, "the stimulus (sample, logged code) is a Status onset; NA when Status was not read"),
    _c("t_to_recording_end_s", "float", True, "recording end minus onset; NA if the length is unknown"),
    _c("beyond_recording", "boolean", True, "onset at or after the last sample of the recording; NA if unknown"),
    _c("post_stim_truncated", "bool", description="less than truncated_post_s (default: primary window hi) of data "
       "after onset; False when the length is unknown"),
    _c("stim_unverified", "bool", description="Status was read and does not confirm the stimulus, or the onset lies "
       "beyond the recording (excluded by every analysis subset)"),
    _c("excluded_known_issue", "bool", description="row excluded by a known-issues registry action"),
    _c("known_issue_ids", "str", description="';'-joined registry ids whose actions touch this row ('' if none)"),
    _c("known_issue_flags", "str", description="';'-joined flag_only labels ('' if none)"),
)

#: Child-level columns added by :meth:`TaskTrialBuilder.assemble_children` (every task).
CHILD_BASE_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("trial_index_in_child", "int", description="1-based over the child's runs, ordered by (run, sample)"),
    _c("block_index_in_child", "Int", True, "block_index_in_run plus the highest block_index_in_run of each earlier "
       "run of the child; NA without markers"),
    _c("n_runs_in_child", "int", description="runs of this task for the child that have at least one row"),
)


def soa_stratum_labels(edges: Iterable[float]) -> list[str]:
    """Labels of the SOA strata defined by increasing ``edges`` (ms).

    ``['<e0', '[e0,e1)', ..., '>=en']``: index ``k`` is the label of values
    with ``searchsorted(edges, x, side='right') == k``.
    """
    e = [float(x) for x in edges]
    fmt = lambda x: f"{x:g}"  # noqa: E731
    return [f"<{fmt(e[0])}"] + [f"[{fmt(a)},{fmt(b)})" for a, b in zip(e, e[1:])] + [f">={fmt(e[-1])}"]


def trial_table_schema(task: TaskEventsConfig, extra: Iterable[ColumnSpec] = ()) -> TableSchema:
    """Schema of a task trial table: base columns + per-window + per-lookback + task extras."""
    cols = list(BASE_TRIAL_COLUMNS)
    for w in task.rt_windows.values():
        if w.name == task.primary_window:
            continue
        cols += [_c(f"rt_ms_{w.name}", "float", True, f"RT under the sensitivity window [{w.lo_ms:g}, {w.hi_ms:g}] ms "
                    "(same rule as rt_ms)"),
                 _c(f"responded_{w.name}", "bool", description=f"rt_ms_{w.name} is defined")]
    for L in task.lookbacks_ms:
        cols += [_c(f"press_in_lookback_{L}", "bool", description=f"a press in [onset - {L} ms, onset)"),
                 _c(f"stim_in_lookback_{L}", "bool", description=f"a stimulus onset (any stimulus code) in "
                    f"[onset - {L} ms, onset)"),
                 _c(f"clean_lookback_{L}", "bool", description=f"neither press nor stimulus in [onset - {L} ms, onset)"),
                 _c(f"marker_in_lookback_{L}", "bool", description=f"a structure marker (marker_codes, incl. "
                    f"Status-only) in [onset - {L} ms, onset)"),
                 _c(f"clean_all_lookback_{L}", "bool", description=f"no press, stimulus or marker in "
                    f"[onset - {L} ms, onset)")]
    for X in task.press_clear_ms:
        cols.append(_c(f"press_clear_{X}", "bool", description=f"no press in [onset - {X} ms, onset), concatenated "
                       f"clock"))
    m = task.prestim_edge_margin_s
    for wn, (a, b) in task.prestim_windows.items():
        cols.append(_c(f"edge_valid_{wn}", "boolean", True,
                       f"prestimulus window '{wn}' [{a:g}, {b:g}) ms (samples onset + round(ms x sfreq / 1000)) lies "
                       f"in one join-delimited segment of the recording with >= {m:g} s (round(margin x sfreq) "
                       f"samples) between each window edge and the nearest join or file edge (FIR half-length); NA "
                       f"if joins or the recording length are unknown. Bad spans (rail, flat) are handled in "
                       f"preprocessing"))
    schema = TableSchema(f"trials_{task.name}", tuple(cols), allow_extra=False,
                         keys=("subject", "task", "run", "sample"))
    extra = tuple(extra)
    return schema.extend(extra) if extra else schema


def apply_subset(df: pd.DataFrame, subset: Mapping[str, Iterable[str]]) -> pd.Series:
    """Boolean mask for an analysis subset ``{require_true: [...], require_false: [...]}``.

    Missing values count as not satisfying a requirement. Only boolean
    columns are accepted (a float column would otherwise mean "non-zero").

    Raises
    ------
    KeyError
        If a named column is missing.
    TypeError
        If a named column is not boolean.
    """
    mask = pd.Series(True, index=df.index)
    for req in ("require_true", "require_false"):
        for c in subset.get(req, []) or []:
            if c not in df.columns:
                raise KeyError(f"subset column {c!r} not in table")
            if str(df[c].dtype) not in ("bool", "boolean"):
                raise TypeError(f"subset column {c!r} has dtype {df[c].dtype}; subset flags must be boolean")
            if req == "require_true":
                mask &= df[c].fillna(False).astype(bool)
            else:
                mask &= ~df[c].fillna(True).astype(bool)
    return mask
