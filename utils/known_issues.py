"""Registry of known-bad / irregular recordings (loader only).

Each issue targets a subject, optionally one task and one run, and carries a
list of machine-readable actions from a fixed vocabulary (:data:`ACTION_SPECS`).
Task builders and preprocessing steps query the registry; they never hard-code
subject IDs. The registry file itself lists participant IDs and is not part of
this repository.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import ConfigError, read_yaml

#: Allowed action types -> (required params, optional params).
ACTION_SPECS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "exclude_recording": (frozenset(), frozenset({"scopes"})),
    "exclude_from_session_order": (frozenset(), frozenset()),
    "exclude_from_iaf": (frozenset(), frozenset()),
    "recode_event": (frozenset({"sample", "from_value", "to_value"}), frozenset()),
    "exclude_sample_range": (frozenset({"start_sample", "end_sample"}), frozenset()),
    "exclude_time_range": (frozenset({"start_s", "end_s"}), frozenset()),
    "exclude_events_before_first_marker": (frozenset({"marker_codes"}), frozenset()),
    "crop_signal_after_s": (frozenset({"t_s"}), frozenset()),
    "crop_to_task_events": (frozenset(), frozenset({"pad_s"})),
    "set_channel_status": (frozenset({"channels", "status"}), frozenset()),
    "no_exg": (frozenset(), frozenset()),
    "exclude_partial_blocks": (frozenset({"nominal_block_size"}), frozenset()),
    "flag_only": (frozenset({"flag"}), frozenset()),
    "set_value_missing": (frozenset({"column"}), frozenset()),
    "allow_unclassified_codes": (frozenset({"codes"}), frozenset()),
}
EXCLUSION_SCOPES = frozenset({"trials", "preprocessing", "iaf", "session_order"})
DECISIONS = frozenset({"confirmed", "provisional"})
_ID_RE = re.compile(r"^KI-\d{3,}$")
_SUB_RE = re.compile(r"^sub-[A-Za-z0-9]+$")


@dataclass(frozen=True)
class IssueAction:
    """One machine-readable action of a known issue."""

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @property
    def scopes(self) -> frozenset[str]:
        """Exclusion scopes (only meaningful for ``exclude_recording``)."""
        return frozenset(self.params.get("scopes") or EXCLUSION_SCOPES)


@dataclass(frozen=True)
class KnownIssue:
    """A registry entry."""

    id: str
    subject: str
    task: str | None
    run: int | None
    category: str
    description: str
    evidence: str
    decision: str
    actions: tuple[IssueAction, ...]

    def applies_to(self, subject: str, task: str | None = None, run: int | None = None) -> bool:
        """Whether the issue targets this subject / task / run (``None`` task/run = wildcard)."""
        if self.subject != subject:
            return False
        if self.task is not None and task is not None and self.task != task:
            return False
        if self.task is not None and task is None:
            return False
        if self.run is not None and run is not None and self.run != run:
            return False
        return True


def _parse_action(d: Any, where: str) -> IssueAction:
    if not isinstance(d, Mapping) or "type" not in d:
        raise ConfigError(f"{where}: each action must be a mapping with a 'type'")
    typ = str(d["type"])
    if typ not in ACTION_SPECS:
        raise ConfigError(f"{where}: unknown action type {typ!r}; allowed: {sorted(ACTION_SPECS)}")
    req, opt = ACTION_SPECS[typ]
    params = {k: v for k, v in d.items() if k != "type"}
    missing = req - set(params)
    extra = set(params) - req - opt
    if missing:
        raise ConfigError(f"{where}: action {typ} missing params {sorted(missing)}")
    if extra:
        raise ConfigError(f"{where}: action {typ} has unknown params {sorted(extra)}")
    if typ == "exclude_recording" and "scopes" in params:
        bad = set(params["scopes"]) - EXCLUSION_SCOPES
        if bad:
            raise ConfigError(f"{where}: unknown exclusion scopes {sorted(bad)}")
    if typ == "recode_event":
        for k in ("sample", "from_value", "to_value"):
            if not isinstance(params[k], int):
                raise ConfigError(f"{where}: recode_event.{k} must be an integer")
    if typ == "exclude_sample_range":
        a, b = params["start_sample"], params["end_sample"]
        if a is not None and b is not None and a > b:
            raise ConfigError(f"{where}: exclude_sample_range start > end")
    if typ == "allow_unclassified_codes":
        c = params["codes"]
        if not isinstance(c, list) or not c or not all(isinstance(x, int) and not isinstance(x, bool) for x in c):
            raise ConfigError(f"{where}: allow_unclassified_codes.codes must be a non-empty list of integers")
    if typ == "exclude_time_range":
        a, b = params["start_s"], params["end_s"]
        if a is not None and b is not None and a > b:
            raise ConfigError(f"{where}: exclude_time_range start > end")
    return IssueAction(typ, params)


def _parse_issue(d: Any, where: str) -> KnownIssue:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where}: issue must be a mapping")
    for k in ("id", "subject", "category", "description", "decision", "actions"):
        if k not in d:
            raise ConfigError(f"{where}: missing key {k!r}")
    iid = str(d["id"])
    if not _ID_RE.match(iid):
        raise ConfigError(f"{where}: id {iid!r} must look like KI-001")
    sub = str(d["subject"])
    if not _SUB_RE.match(sub):
        raise ConfigError(f"{where}: subject {sub!r} must look like sub-XXXX")
    run = d.get("run")
    if run is not None and (not isinstance(run, int) or run < 1):
        raise ConfigError(f"{where}: run must be a positive integer or null")
    dec = str(d["decision"])
    if dec not in DECISIONS:
        raise ConfigError(f"{where}: decision must be one of {sorted(DECISIONS)}")
    acts = d["actions"]
    if not isinstance(acts, list) or not acts:
        raise ConfigError(f"{where}: actions must be a non-empty list")
    return KnownIssue(
        id=iid, subject=sub, task=(str(d["task"]) if d.get("task") is not None else None), run=run,
        category=str(d["category"]), description=" ".join(str(d["description"]).split()),
        evidence=str(d.get("evidence", "")), decision=dec,
        actions=tuple(_parse_action(a, f"{where}.actions[{i}]") for i, a in enumerate(acts)),
    )


class KnownIssuesRegistry:
    """Queryable collection of :class:`KnownIssue`.

    Parameters
    ----------
    issues
        Parsed issues.
    """

    def __init__(self, issues: Iterable[KnownIssue]) -> None:
        self.issues = tuple(issues)
        ids = [i.id for i in self.issues]
        dup = {x for x in ids if ids.count(x) > 1}
        if dup:
            raise ConfigError(f"duplicate known-issue ids: {sorted(dup)}")

    @classmethod
    def load(cls, path: str | os.PathLike[str], valid_tasks: Iterable[str] | None = None) -> KnownIssuesRegistry:
        """Load and validate a registry YAML.

        Parameters
        ----------
        path
            Registry YAML (top-level ``issues`` list).
        valid_tasks
            If given, every non-null ``task`` must be one of these.
        """
        raw = read_yaml(path)
        items = raw.get("issues")
        if not isinstance(items, list):
            raise ConfigError(f"{path}: top-level 'issues' list required")
        issues = [_parse_issue(d, f"{path}:issues[{i}]") for i, d in enumerate(items)]
        if valid_tasks is not None:
            vt = set(valid_tasks)
            bad = sorted({i.task for i in issues if i.task is not None and i.task not in vt})
            if bad:
                raise ConfigError(f"{path}: unknown tasks {bad}")
        return cls(issues)

    def __len__(self) -> int:
        return len(self.issues)

    def __iter__(self) -> Iterator[KnownIssue]:
        return iter(self.issues)

    def get(self, issue_id: str) -> KnownIssue:
        for i in self.issues:
            if i.id == issue_id:
                return i
        raise KeyError(issue_id)

    def for_recording(self, subject: str, task: str, run: int) -> list[KnownIssue]:
        """Issues that apply to one recording (subject-wide and task-wide entries included)."""
        return [i for i in self.issues if i.applies_to(subject, task, run)]

    def for_subject(self, subject: str) -> list[KnownIssue]:
        """Every issue of a subject."""
        return [i for i in self.issues if i.subject == subject]

    def actions(self, subject: str, task: str, run: int, type: str | None = None) -> list[tuple[KnownIssue, IssueAction]]:
        """(issue, action) pairs for a recording, optionally filtered by action type."""
        return [(i, a) for i in self.for_recording(subject, task, run) for a in i.actions
                if type is None or a.type == type]

    def is_excluded(self, subject: str, task: str, run: int, scope: str) -> bool:
        """Whether an ``exclude_recording`` action (or a scope-specific action) excludes a recording."""
        if scope not in EXCLUSION_SCOPES:
            raise ValueError(f"unknown scope {scope!r}")
        for _, a in self.actions(subject, task, run):
            if a.type == "exclude_recording" and scope in a.scopes:
                return True
            if scope == "session_order" and a.type == "exclude_from_session_order":
                return True
            if scope == "iaf" and a.type == "exclude_from_iaf":
                return True
        return False

    def to_frame(self) -> pd.DataFrame:
        """One row per (issue, action)."""
        rows = []
        for i in self.issues:
            for a in i.actions:
                rows.append({"id": i.id, "subject": i.subject, "task": i.task, "run": i.run,
                             "category": i.category, "decision": i.decision, "action": a.type,
                             "params": dict(a.params)})
        return pd.DataFrame(rows)


def load_known_issues(root: str | os.PathLike[str], valid_tasks: Iterable[str] | None = None) -> KnownIssuesRegistry:
    """Load ``<root>/configs/known_issues.yaml``."""
    return KnownIssuesRegistry.load(Path(root) / "configs" / "known_issues.yaml", valid_tasks)
