"""Load and validate the project configuration and ``configs/events.yaml``.

Both loaders return immutable dataclasses. The project configuration (paths, acquisition, task lists, session and
runtime settings) is site-specific and not part of this repository; :class:`ProjectConfig` documents the keys it
needs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for a missing key, a wrong type or an inconsistent value in a config file."""


# --------------------------------------------------------------------------- helpers
def read_yaml(path: str | os.PathLike[str]) -> dict:
    """Parse a YAML file into a dict (empty file -> ``{}``)."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def find_project_root() -> Path:
    """Repository root (the directory that holds ``configs/``)."""
    return Path(__file__).resolve().parent.parent


def _req(d: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where}: expected a mapping")
    if key not in d:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return d[key]


def _str_list(v: Any, where: str) -> tuple[str, ...]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ConfigError(f"{where}: expected a list of strings")
    return tuple(v)


def _int_list(v: Any, where: str) -> tuple[int, ...]:
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise ConfigError(f"{where}: expected a list of integers")
    return tuple(v)


def _code_map(v: Any, where: str) -> dict[int, str]:
    if not isinstance(v, Mapping) or not v:
        raise ConfigError(f"{where}: expected a non-empty mapping code -> label")
    out = {}
    for k, lab in v.items():
        if not isinstance(k, int) or isinstance(k, bool):
            raise ConfigError(f"{where}: code {k!r} is not an integer")
        out[int(k)] = str(lab)
    return out


def _num(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {v!r}")
    return float(v)


def _int(v: Any, where: str) -> int:
    """Strict integer (``True`` / ``'8'`` / ``8.5`` are rejected)."""
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{where}: expected an integer, got {v!r}")
    return int(v)


def _bool(v: Any, where: str) -> bool:
    """Strict boolean (the string ``'false'`` is rejected, not read as True)."""
    if not isinstance(v, bool):
        raise ConfigError(f"{where}: expected true or false, got {v!r}")
    return v


def _str(v: Any, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{where}: expected a non-empty string, got {v!r}")
    return v


def _known_keys(d: Mapping[str, Any], allowed: set[str], where: str) -> None:
    bad = set(d) - set(allowed)
    if bad:
        raise ConfigError(f"{where}: unknown keys {sorted(map(str, bad))}; allowed: {sorted(allowed)}")


# --------------------------------------------------------------------------- project config
@dataclass(frozen=True)
class PathsConfig:
    """Absolute project paths (resolved against the project root)."""

    root: Path
    bids_root: Path
    derivatives: Path
    cache: Path
    results: Path
    results_tables: Path
    results_diagnostics: Path
    logs: Path


@dataclass(frozen=True)
class AcquisitionConfig:
    sfreq: float
    status_channel: str
    n_eeg_channels: int
    exg_channels: tuple[str, ...]
    line_freq: float
    record_duration_s: float


@dataclass(frozen=True)
class TasksConfig:
    all: tuple[str, ...]
    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    rest: tuple[str, ...]
    undated: tuple[str, ...]


@dataclass(frozen=True)
class SessionConfig:
    main_day_basis: str
    task_recordings: tuple[str, ...]
    tie_break: str


@dataclass(frozen=True)
class RuntimeConfig:
    n_jobs: int
    random_seed: int


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration.

    Top-level keys: ``paths`` (``bids_root``, ``derivatives``, ``results``, ``results_tables``,
    ``results_diagnostics``, ``logs``, optional ``cache``; relative to the project root), ``acquisition``
    (``sfreq``, ``status_channel``, ``n_eeg_channels``, ``exg_channels``, ``line_freq``, ``record_duration_s``),
    ``tasks`` (``all``, ``primary``, ``secondary``, ``rest``, ``undated``), ``session`` (``main_day_basis``,
    ``task_recordings``, ``tie_break``) and ``runtime`` (``n_jobs``, ``random_seed``).
    """

    paths: PathsConfig
    acquisition: AcquisitionConfig
    tasks: TasksConfig
    session: SessionConfig
    runtime: RuntimeConfig
    source: Path
    raw: dict = field(repr=False, compare=False)

    @property
    def root(self) -> Path:
        return self.paths.root


def load_project_config(
    path: str | os.PathLike[str] | None = None, root: str | os.PathLike[str] | None = None
) -> ProjectConfig:
    """Load the project configuration (``<root>/configs/project.yaml``).

    Parameters
    ----------
    path
        Explicit config path. Default: ``<root>/configs/project.yaml``.
    root
        Project root. Default: :func:`find_project_root`
        (or the grandparent of ``path`` when only ``path`` is given).

    Returns
    -------
    ProjectConfig

    Raises
    ------
    ConfigError
        On missing keys or invalid values.
    """
    if path is None:
        root_p = Path(root) if root is not None else find_project_root()
        path = root_p / "configs" / "project.yaml"
    else:
        path = Path(path)
        root_p = Path(root) if root is not None else path.resolve().parent.parent
    root_p = root_p.resolve()
    raw = read_yaml(path)
    w = str(path)

    p = _req(raw, "paths", w)
    def rp(key: str) -> Path:
        v = _req(p, key, f"{w}:paths")
        if not isinstance(v, str) or Path(v).is_absolute():
            raise ConfigError(f"{w}:paths.{key} must be a relative path string")
        return (root_p / v).resolve()

    paths = PathsConfig(
        root=root_p, bids_root=rp("bids_root"), derivatives=rp("derivatives"),
        cache=rp("cache") if "cache" in p else rp("derivatives") / "cache",
        results=rp("results"), results_tables=rp("results_tables"),
        results_diagnostics=rp("results_diagnostics"), logs=rp("logs"),
    )

    a = _req(raw, "acquisition", w)
    acq = AcquisitionConfig(
        sfreq=_num(_req(a, "sfreq", f"{w}:acquisition"), f"{w}:acquisition.sfreq"),
        status_channel=str(_req(a, "status_channel", f"{w}:acquisition")),
        n_eeg_channels=_int(_req(a, "n_eeg_channels", f"{w}:acquisition"), f"{w}:acquisition.n_eeg_channels"),
        exg_channels=_str_list(_req(a, "exg_channels", f"{w}:acquisition"), f"{w}:acquisition.exg_channels"),
        line_freq=_num(a.get("line_freq", 60.0), f"{w}:acquisition.line_freq"),
        record_duration_s=_num(a.get("record_duration_s", 1.0), f"{w}:acquisition.record_duration_s"),
    )
    if acq.sfreq <= 0:
        raise ConfigError(f"{w}:acquisition.sfreq must be > 0")

    t = _req(raw, "tasks", w)
    tasks = TasksConfig(**{k: _str_list(_req(t, k, f"{w}:tasks"), f"{w}:tasks.{k}")
                           for k in ("all", "primary", "secondary", "rest", "undated")})
    for k in ("primary", "secondary", "rest", "undated"):
        bad = set(getattr(tasks, k)) - set(tasks.all)
        if bad:
            raise ConfigError(f"{w}:tasks.{k} lists tasks not in tasks.all: {sorted(bad)}")

    s = raw.get("session", {}) or {}
    session = SessionConfig(
        main_day_basis=str(s.get("main_day_basis", "non_motor")),
        task_recordings=_str_list(s.get("task_recordings", list(tasks.primary + tasks.secondary)), f"{w}:session.task_recordings"),
        tie_break=str(s.get("tie_break", "earliest_date")),
    )
    if session.main_day_basis not in ("non_motor", "tasks_only"):
        raise ConfigError(f"{w}:session.main_day_basis must be 'non_motor' or 'tasks_only'")
    if session.tie_break != "earliest_date":
        raise ConfigError(f"{w}:session.tie_break must be 'earliest_date'")

    r = raw.get("runtime", {}) or {}
    runtime = RuntimeConfig(n_jobs=_int(r.get("n_jobs", 1), f"{w}:runtime.n_jobs"),
                            random_seed=_int(r.get("random_seed", 0), f"{w}:runtime.random_seed"))

    return ProjectConfig(paths=paths, acquisition=acq, tasks=tasks, session=session,
                         runtime=runtime, source=Path(path).resolve(), raw=raw)


# --------------------------------------------------------------------------- events config
@dataclass(frozen=True)
class StatusConfig:
    """Status-channel decoding parameters."""

    channel: str
    event_mask: int
    join_bit: int
    shortest_event: int
    stuck_code_min_samples: int
    status_only_marker_codes: tuple[int, ...]
    sample_onset_tolerance: int


@dataclass(frozen=True)
class RTWindow:
    """Closed response window [lo_ms, hi_ms]."""

    name: str
    lo_ms: float
    hi_ms: float


@dataclass(frozen=True)
class HistoryRule:
    """History-homogeneous stratum: previous target outcome, lag of the last press, no join since the target."""

    prev_outcome: str
    press_lag_min_ms: float
    press_lag_max_ms: float
    last_press_is_prev_response: bool
    no_join_since_prev_target: bool


@dataclass(frozen=True)
class TaskEventsConfig:
    """Event conventions of one task."""

    name: str
    role: str
    stimulus_codes: dict[int, str]
    trial_codes: tuple[int, ...]
    target_codes: tuple[int, ...]
    response_codes: tuple[int, ...]
    response_bound_codes: tuple[int, ...]
    marker_codes: dict[int, str]
    block_start_codes: tuple[int, ...]
    rt_windows: dict[str, RTWindow]
    primary_window: str
    lookbacks_ms: tuple[int, ...]
    join_padding_ms: float
    join_post_ms: float
    history: HistoryRule
    misattribution_fast_rt_ms: float
    analysis_subsets: dict
    rt_targets_only: bool = False
    soa_strata_ms: tuple[float, ...] = ()
    misattribution_applies: bool = True
    truncated_post_s: float | None = None
    prestim_windows: dict[str, tuple[float, float]] = field(default_factory=dict)
    prestim_edge_margin_s: float = 1.65
    press_clear_ms: tuple[int, ...] = ()
    join_near_lookback_ms: int | None = None
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def post_stim_required_s(self) -> float:
        """Data needed after onset (``truncated_post_s``; default: the primary window's hi)."""
        return self.truncated_post_s if self.truncated_post_s is not None else self.primary.hi_ms / 1000.0

    @property
    def primary(self) -> RTWindow:
        return self.rt_windows[self.primary_window]

    @property
    def max_lookback_ms(self) -> int:
        return max(self.lookbacks_ms)

    @property
    def join_lookback_ms(self) -> int:
        """Lookback used by ``join_near_window`` (``join_near_lookback_ms``; default: the longest lookback)."""
        return self.join_near_lookback_ms if self.join_near_lookback_ms is not None else self.max_lookback_ms


@dataclass(frozen=True)
class EventsConfig:
    """Validated ``configs/events.yaml``."""

    status: StatusConfig
    tasks: dict[str, TaskEventsConfig]
    response_rule: str
    source: Path
    raw: dict = field(repr=False, compare=False)

    def task(self, name: str) -> TaskEventsConfig:
        if name not in self.tasks:
            raise ConfigError(f"task {name!r} not configured in {self.source}")
        return self.tasks[name]


_TASK_KNOWN_KEYS = {
    "role", "status", "stimulus_codes", "trial_codes", "target_codes", "response_codes",
    "response_bound_codes", "marker_codes", "block_start_codes", "rt_windows_ms", "primary_window",
    "lookbacks_ms", "join_padding_ms", "join_post_ms", "history_homogeneous", "misattribution",
    "analysis_subsets", "rt_targets_only", "soa_strata_ms", "truncated_post_s",
    "prestim_windows_ms", "prestim_edge_margin_s", "press_clear_ms", "join_near_lookback_ms",
}
#: Keys of one analysis subset. ``extends`` names a subset whose requirements are inherited (resolved at load time,
#: so ``require_true`` / ``require_false`` always hold the full lists); ``outcome_independent`` subsets may not use
#: any column that depends on the response to the current stimulus (the predictor set).
SUBSET_KEYS = frozenset({"require_true", "require_false", "extends", "outcome_independent", "description"})
#: Columns (or prefixes) that depend on the response to the current stimulus. An ``outcome_independent`` subset may
#: not name them. Previous-trial and press-history columns are allowed (they precede the onset).
OUTCOME_COLUMN_PREFIXES: tuple[str, ...] = (
    "responded", "rt_", "late_hit", "response_after_next_contour", "anticipation_before_kept_press", "flag_rt_",
    "false_alarm", "raw_first_press", "n_valid_window_presses", "outcome",
)


def is_outcome_column(name: str) -> bool:
    """Whether ``name`` depends on the response to the current stimulus (see :data:`OUTCOME_COLUMN_PREFIXES`).

    A deny-list, kept as a second line of defence. The guard on ``outcome_independent`` subsets is the allow-list
    :func:`outcome_independence_basis` (fail closed: a new column with a new name is refused until it is classified).
    """
    return any(name == p or name.startswith(p) for p in OUTCOME_COLUMN_PREFIXES)


#: Allow-list of the columns an ``outcome_independent`` subset may name, each with the reason it does not depend on
#: the response to the current stimulus. Exact names; see also the patterns below.
OUTCOME_INDEPENDENT_COLUMNS: dict[str, str] = {
    "excluded_known_issue": "known-issues registry action (file-level fault registry)",
    "stim_unverified": "Status confirmation of the stimulus onset",
    "post_stim_truncated": "recording length after onset",
    "is_target": "stimulus code",
    "last_in_run": "position of the stimulus in the run",
    "first_in_run": "position of the stimulus in the run",
    "first_in_block": "position of the stimulus in its block (markers)",
    "join_near_window": "bit-16 joins around onset",
    "post_gap": "SOA before onset, joins",
    "prestim_has_unlogged_stim": "gap before onset (dropped trigger)",
    "next_is_dropped_trigger": "gap after onset to the next logged stimulus (stimulus timing, no press)",
    "in_aborted_segment": "stimulus gaps (pause/restart), no press",
    "first_after_block_marker": "Status markers",
    "before_first_block_marker": "Status markers",
    "first_in_segment": "SOA, joins and run position",
    "first_after_break": "Status markers / block structure",
    "post_pause_first_flip": "IC pause markers",
    "after_foreign_cut": "foreign-paradigm trigger segment",
    "beyond_ic_cap": "IC contour-sequence counts (stimulus stream)",
    "seq180_complete": "IC contour-sequence counts (stimulus stream)",
}
#: Column families on the allow-list (fnmatch patterns): lookback contamination and press clearance use presses
#: strictly *before* onset; edge validity uses joins and file edges.
OUTCOME_INDEPENDENT_PATTERNS: dict[str, str] = {
    "clean_all_lookback_*": "presses, stimuli and markers strictly before onset",
    "clean_lookback_*": "presses and stimuli strictly before onset",
    "press_in_lookback_*": "presses strictly before onset",
    "stim_in_lookback_*": "stimuli strictly before onset",
    "marker_in_lookback_*": "markers strictly before onset",
    "press_clear_*": "presses strictly before onset",
    "edge_valid_*": "bit-16 joins and file edges around the prestimulus window",
}
#: Behaviour-informed columns allowed in outcome-independent sets by design.
BEHAVIOUR_INFORMED_EXCEPTIONS: dict[str, str] = {
    "low_press_rate_segment": (
        "AVSRT segment rule: computed from the any-press rate of the whole segment, the current trial's own press "
        "included (behaviour-informed). Every row it flags is also excluded by a known-issue fault, so the rule "
        "removes no row that the known issues keep; the predictor set without it is "
        "sens_keep_low_press_rate_predictor_set"),
}


def outcome_independence_basis(name: str) -> str | None:
    """Why column ``name`` may appear in an ``outcome_independent`` subset, or ``None`` if it may not.

    Behaviour-informed exceptions (:data:`BEHAVIOUR_INFORMED_EXCEPTIONS`) return their reason prefixed with
    ``"behaviour-informed exception: "``. Columns matched by the deny-list :func:`is_outcome_column` never qualify.
    """
    import fnmatch

    if is_outcome_column(name):
        return None
    if name in BEHAVIOUR_INFORMED_EXCEPTIONS:
        return "behaviour-informed exception: " + BEHAVIOUR_INFORMED_EXCEPTIONS[name]
    if name in OUTCOME_INDEPENDENT_COLUMNS:
        return OUTCOME_INDEPENDENT_COLUMNS[name]
    for pat, why in OUTCOME_INDEPENDENT_PATTERNS.items():
        if fnmatch.fnmatchcase(name, pat):
            return why
    return None


def _resolve_subsets(subsets: Mapping[str, Any], where: str) -> dict[str, dict]:
    """Validate the analysis subsets and resolve ``extends`` (parents first, duplicates removed).

    Returns ``{name: {"require_true": [...], "require_false": [...], "extends": str | None,
    "outcome_independent": bool, "description": str}}``.
    """
    raw: dict[str, Mapping[str, Any]] = {}
    for sn, sd in subsets.items():
        if not isinstance(sd, Mapping) or set(sd) - SUBSET_KEYS:
            bad = sorted(set(sd) - SUBSET_KEYS) if isinstance(sd, Mapping) else sd
            raise ConfigError(f"{where}.analysis_subsets.{sn}: only {sorted(SUBSET_KEYS)} allowed (got {bad})")
        for k in ("require_true", "require_false"):
            _str_list(sd.get(k) or [], f"{where}.analysis_subsets.{sn}.{k}")
        if "extends" in sd and sd["extends"] is not None:
            _str(sd["extends"], f"{where}.analysis_subsets.{sn}.extends")
        if "outcome_independent" in sd:
            _bool(sd["outcome_independent"], f"{where}.analysis_subsets.{sn}.outcome_independent")
        if "description" in sd:
            _str(sd["description"], f"{where}.analysis_subsets.{sn}.description")
        raw[str(sn)] = sd
    out: dict[str, dict] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> dict:
        if name in out:
            return out[name]
        if name in chain:
            raise ConfigError(f"{where}.analysis_subsets: 'extends' cycle {' -> '.join(chain + (name,))}")
        if name not in raw:
            raise ConfigError(f"{where}.analysis_subsets.{chain[-1]}.extends: unknown subset {name!r}")
        sd = raw[name]
        parent = sd.get("extends")
        rt: list[str] = []
        rf: list[str] = []
        if parent:
            p = resolve(str(parent), chain + (name,))
            rt, rf = list(p["require_true"]), list(p["require_false"])
        for c in sd.get("require_true") or []:
            if c not in rt:
                rt.append(str(c))
        for c in sd.get("require_false") or []:
            if c not in rf:
                rf.append(str(c))
        both = set(rt) & set(rf)
        if both:
            raise ConfigError(f"{where}.analysis_subsets.{name}: {sorted(both)} both required True and False")
        oi = bool(sd.get("outcome_independent", False))
        if oi:
            dep = [c for c in rt + rf if is_outcome_column(c)]
            if dep:
                raise ConfigError(f"{where}.analysis_subsets.{name}: outcome_independent subset uses outcome "
                                  f"column(s) {dep}")
            unknown = [c for c in rt + rf if outcome_independence_basis(c) is None]
            if unknown:
                raise ConfigError(f"{where}.analysis_subsets.{name}: outcome_independent subset uses column(s) "
                                  f"{unknown} that are not on the outcome-independence allow-list "
                                  "(utils.config.OUTCOME_INDEPENDENT_COLUMNS / _PATTERNS); classify "
                                  "them there, with the reason, if they do not depend on the current response")
        out[name] = {"require_true": rt, "require_false": rf, "extends": str(parent) if parent else None,
                     "outcome_independent": oi, "description": str(sd.get("description", "")).strip()}
        return out[name]

    for sn in raw:
        resolve(sn, ())
    return {k: out[k] for k in raw}


def _parse_prestim_windows(v: Any, where: str) -> dict[str, tuple[float, float]]:
    if v is None:
        return {}
    if not isinstance(v, Mapping):
        raise ConfigError(f"{where}: expected a mapping name -> [start_ms, stop_ms]")
    out: dict[str, tuple[float, float]] = {}
    for wn, x in v.items():
        w = f"{where}.{wn}"
        if not str(wn).replace("_", "").isalnum():
            raise ConfigError(f"{w}: window name must be alphanumeric/underscore")
        if not (isinstance(x, list) and len(x) == 2):
            raise ConfigError(f"{w}: expected [start_ms, stop_ms]")
        a, b = _num(x[0], w), _num(x[1], w)
        if not a < b <= 0:
            raise ConfigError(f"{w}: need start < stop <= 0 (ms relative to onset), got {x}")
        out[str(wn)] = (a, b)
    return out
#: Builder-specific blocks allowed per task (validated in depth by the builder's parameter class at load time).
TASK_EXTRA_KEYS: dict[str, frozenset[str]] = {
    "AVSRT": frozenset({"avsrt"}),
    "FAST": frozenset({"block_transition_press_max_s", "block_structure", "nonperformance"}),
    "IC": frozenset({"ic", "nonperformance"}),
}


def _parse_task(name: str, d: Mapping[str, Any], where: str) -> TaskEventsConfig:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where}: expected a mapping")
    _known_keys(d, _TASK_KNOWN_KEYS | set(TASK_EXTRA_KEYS.get(name, ())), where)
    stim = _code_map(_req(d, "stimulus_codes", where), f"{where}.stimulus_codes")
    trial = _int_list(_req(d, "trial_codes", where), f"{where}.trial_codes")
    target = _int_list(d.get("target_codes", list(trial)), f"{where}.target_codes")
    resp = _int_list(_req(d, "response_codes", where), f"{where}.response_codes")
    bound = _int_list(_req(d, "response_bound_codes", where), f"{where}.response_bound_codes")
    mk_raw = d.get("marker_codes") or {}
    markers = _code_map(mk_raw, f"{where}.marker_codes") if mk_raw else {}
    bstart = _int_list(d.get("block_start_codes", []), f"{where}.block_start_codes")
    if not trial:
        raise ConfigError(f"{where}.trial_codes must not be empty")
    if not resp:
        raise ConfigError(f"{where}.response_codes must not be empty")
    for nm, codes in (("trial_codes", trial), ("response_bound_codes", bound)):
        bad = set(codes) - set(stim)
        if bad:
            raise ConfigError(f"{where}.{nm} not in stimulus_codes: {sorted(bad)}")
    if set(target) - set(trial):
        raise ConfigError(f"{where}.target_codes must be a subset of trial_codes")
    if set(trial) - set(bound):
        raise ConfigError(
            f"{where}.response_bound_codes must include every trial code (otherwise one press could be matched "
            f"to two trials); missing {sorted(set(trial) - set(bound))}")
    if set(resp) & set(stim):
        raise ConfigError(f"{where}: response_codes overlap stimulus_codes")
    if set(bstart) - set(markers):
        raise ConfigError(f"{where}.block_start_codes must be listed in marker_codes")

    wins_raw = _req(d, "rt_windows_ms", where)
    if not isinstance(wins_raw, Mapping) or not wins_raw:
        raise ConfigError(f"{where}.rt_windows_ms must be a non-empty mapping")
    wins = {}
    for wn, v in wins_raw.items():
        if not (isinstance(v, list) and len(v) == 2):
            raise ConfigError(f"{where}.rt_windows_ms.{wn} must be [lo, hi]")
        lo, hi = _num(v[0], f"{where}.rt_windows_ms.{wn}"), _num(v[1], f"{where}.rt_windows_ms.{wn}")
        if not 0 <= lo < hi:
            raise ConfigError(f"{where}.rt_windows_ms.{wn}: need 0 <= lo < hi, got {v}")
        if not str(wn).replace("_", "").isalnum():
            raise ConfigError(f"{where}.rt_windows_ms: window name {wn!r} must be alphanumeric/underscore")
        wins[str(wn)] = RTWindow(str(wn), lo, hi)
    prim = _str(_req(d, "primary_window", where), f"{where}.primary_window")
    if prim not in wins:
        raise ConfigError(f"{where}.primary_window {prim!r} not in rt_windows_ms")

    looks = _int_list(_req(d, "lookbacks_ms", where), f"{where}.lookbacks_ms")
    if not looks or any(x <= 0 for x in looks) or len(set(looks)) != len(looks):
        raise ConfigError(f"{where}.lookbacks_ms must be distinct positive integers")

    hw = f"{where}.history_homogeneous"
    h = _req(d, "history_homogeneous", where)
    if not isinstance(h, Mapping):
        raise ConfigError(f"{hw}: expected a mapping")
    _known_keys(h, {"prev_outcome", "press_lag_ms", "last_press_is_prev_response", "no_join_since_prev_target"}, hw)
    lag = _req(h, "press_lag_ms", hw)
    if not (isinstance(lag, list) and len(lag) == 2):
        raise ConfigError(f"{hw}.press_lag_ms must be [lo, hi]")
    lag_lo, lag_hi = _num(lag[0], f"{hw}.press_lag_ms"), _num(lag[1], f"{hw}.press_lag_ms")
    if not 0 <= lag_lo < lag_hi:
        raise ConfigError(f"{hw}.press_lag_ms must be [lo, hi] with 0 <= lo < hi")
    po = _str(_req(h, "prev_outcome", hw), f"{hw}.prev_outcome")
    if po not in ("hit", "miss", "any"):
        raise ConfigError(f"{hw}.prev_outcome must be hit, miss or any")
    hist = HistoryRule(prev_outcome=po, press_lag_min_ms=lag_lo, press_lag_max_ms=lag_hi,
                       last_press_is_prev_response=_bool(h.get("last_press_is_prev_response", True),
                                                         f"{hw}.last_press_is_prev_response"),
                       no_join_since_prev_target=_bool(h.get("no_join_since_prev_target", True),
                                                       f"{hw}.no_join_since_prev_target"))
    mis = d.get("misattribution") or {}
    if not isinstance(mis, Mapping):
        raise ConfigError(f"{where}.misattribution must be a mapping")
    _known_keys(mis, {"fast_rt_ms", "applies"}, f"{where}.misattribution")
    subsets = d.get("analysis_subsets") or {}
    if not isinstance(subsets, Mapping):
        raise ConfigError(f"{where}.analysis_subsets must be a mapping")
    resolved = _resolve_subsets(subsets, where)
    pw = _parse_prestim_windows(d.get("prestim_windows_ms"), f"{where}.prestim_windows_ms")
    margin = _num(d.get("prestim_edge_margin_s", 1.65), f"{where}.prestim_edge_margin_s")
    if margin < 0:
        raise ConfigError(f"{where}.prestim_edge_margin_s must be >= 0")
    pc = _int_list(d.get("press_clear_ms", []) or [], f"{where}.press_clear_ms")
    if any(x <= 0 for x in pc) or len(set(pc)) != len(pc):
        raise ConfigError(f"{where}.press_clear_ms must be distinct positive integers")
    jnl = d.get("join_near_lookback_ms")
    if jnl is not None and _int(jnl, f"{where}.join_near_lookback_ms") <= 0:
        raise ConfigError(f"{where}.join_near_lookback_ms must be a positive integer")
    rto = _bool(d.get("rt_targets_only", False), f"{where}.rt_targets_only")
    strata = d.get("soa_strata_ms", []) or []
    if not isinstance(strata, list):
        raise ConfigError(f"{where}.soa_strata_ms must be a list of increasing edges (ms)")
    strata_t = tuple(_num(x, f"{where}.soa_strata_ms") for x in strata)
    if strata_t and (len(strata_t) < 2 or any(b <= a for a, b in zip(strata_t, strata_t[1:]))):
        raise ConfigError(f"{where}.soa_strata_ms must hold at least two strictly increasing edges")
    tps = d.get("truncated_post_s")
    if tps is not None and _num(tps, f"{where}.truncated_post_s") <= 0:
        raise ConfigError(f"{where}.truncated_post_s must be > 0")
    return TaskEventsConfig(
        name=name, role=str(d.get("role", "primary")), stimulus_codes=stim, trial_codes=trial,
        target_codes=target, response_codes=resp, response_bound_codes=bound, marker_codes=markers,
        block_start_codes=bstart, rt_windows=wins, primary_window=prim, lookbacks_ms=looks,
        join_padding_ms=_num(d.get("join_padding_ms", 1000), f"{where}.join_padding_ms"),
        join_post_ms=_num(d.get("join_post_ms", 1500), f"{where}.join_post_ms"),
        history=hist, misattribution_fast_rt_ms=_num(mis.get("fast_rt_ms", 200), f"{where}.misattribution.fast_rt_ms"),
        analysis_subsets=resolved, rt_targets_only=rto, soa_strata_ms=strata_t,
        misattribution_applies=_bool(mis.get("applies", True), f"{where}.misattribution.applies"),
        truncated_post_s=float(tps) if tps is not None else None,
        prestim_windows=pw, prestim_edge_margin_s=float(margin), press_clear_ms=pc,
        join_near_lookback_ms=int(jnl) if jnl is not None else None,
        extra={k: v for k, v in d.items() if k not in _TASK_KNOWN_KEYS},
    )


def load_events_config(
    path: str | os.PathLike[str] | None = None, root: str | os.PathLike[str] | None = None,
    validate_builders: bool = True,
) -> EventsConfig:
    """Load ``configs/events.yaml``.

    The schema is strict: unknown keys (task keys, builder blocks, history and
    status keys) and wrong types raise :class:`ConfigError`. With
    ``validate_builders``, every task that has a trial builder is also checked
    by that builder at load time (its parameter block, and that every
    analysis-subset flag is a boolean column of its table).

    Parameters
    ----------
    path
        Explicit path; default ``<root>/configs/events.yaml``.
    root
        Project root (default: discovered).
    validate_builders
        Run the builder-level checks (:func:`trials.registry.validate_task_config`).

    Returns
    -------
    EventsConfig
    """
    if path is None:
        root_p = Path(root) if root is not None else find_project_root()
        path = root_p / "configs" / "events.yaml"
    raw = read_yaml(path)
    w = str(path)
    s = _req(raw, "status", w)
    sw = f"{w}:status"
    if not isinstance(s, Mapping):
        raise ConfigError(f"{sw}: expected a mapping")
    _known_keys(s, {"channel", "event_mask", "join_bit", "shortest_event", "stuck_code_min_samples",
                    "status_only_marker_codes", "sample_onset_tolerance"}, sw)
    mask = _int(_req(s, "event_mask", sw), f"{sw}.event_mask")
    if not 0 < mask < (1 << 24):
        raise ConfigError(f"{sw}.event_mask must be a positive integer below 2**24")
    jb = _int(_req(s, "join_bit", sw), f"{sw}.join_bit")
    if not 0 <= jb < 24 or (mask >> jb) & 1:
        raise ConfigError(f"{sw}.join_bit must be a bit outside event_mask")
    status = StatusConfig(
        channel=_str(_req(s, "channel", sw), f"{sw}.channel"), event_mask=mask, join_bit=jb,
        shortest_event=_int(s.get("shortest_event", 1), f"{sw}.shortest_event"),
        stuck_code_min_samples=_int(s.get("stuck_code_min_samples", 512), f"{sw}.stuck_code_min_samples"),
        status_only_marker_codes=_int_list(s.get("status_only_marker_codes", []), f"{sw}.status_only_marker_codes"),
        sample_onset_tolerance=_int(s.get("sample_onset_tolerance", 0), f"{sw}.sample_onset_tolerance"),
    )
    tasks_raw = _req(raw, "tasks", w)
    if not isinstance(tasks_raw, Mapping) or not tasks_raw:
        raise ConfigError(f"{w}: tasks must be a non-empty mapping")
    _known_keys(raw, {"status", "tasks", "response_rule"}, w)
    tasks = {str(k): _parse_task(str(k), v, f"{w}:tasks.{k}") for k, v in tasks_raw.items()}
    cfg = EventsConfig(status=status, tasks=tasks, response_rule=str(raw.get("response_rule", "")),
                       source=Path(path).resolve(), raw=raw)
    if validate_builders:
        # lazy import: the builders import this module, which is fully initialised by now
        from trials.registry import validate_task_config

        for name, tc in tasks.items():
            validate_task_config(tc, status, where=f"{w}:tasks.{name}")
    return cfg
