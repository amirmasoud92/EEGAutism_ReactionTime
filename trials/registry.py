"""Task name -> trial builder, and load-time validation of a task's event config."""

from __future__ import annotations

from trials.avsrt import AvsrtTrialBuilder
from trials.base import TaskTrialBuilder
from trials.fast import FastTrialBuilder
from trials.ic import IcTrialBuilder
from utils.config import ConfigError, StatusConfig, TaskEventsConfig

#: Task name -> trial builder class (tasks with a validated builder).
BUILDERS: dict[str, type[TaskTrialBuilder]] = {
    "AVSRT": AvsrtTrialBuilder,
    "FAST": FastTrialBuilder,
    "IC": IcTrialBuilder,
}

_BOOL_DTYPES = ("bool", "boolean")


def validate_task_config(task_cfg: TaskEventsConfig, status_cfg: StatusConfig, where: str = "") -> None:
    """Check a task config against its builder (no-op for tasks without a builder).

    - The builder accepts its parameter block (unknown keys or wrong types in
      ``avsrt`` / ``block_structure`` / ``ic`` raise).
    - The ``nonperformance`` block (FAST, IC) is valid
      (:class:`trials.child_qc.NonperformanceRule`).
    - Every flag named in ``analysis_subsets`` is a column of the task's
      child-level trial table with a boolean dtype (a float column such as
      ``rt_ms`` would otherwise be cast to "non-zero").

    Raises
    ------
    utils.config.ConfigError
    """
    cls = BUILDERS.get(task_cfg.name)
    if cls is None:
        return
    where = where or f"tasks.{task_cfg.name}"
    try:
        builder = cls(task_cfg, status_cfg, None)
        schema = builder.child_schema()
    except ConfigError:
        raise
    except (ValueError, TypeError, KeyError) as e:
        raise ConfigError(f"{where}: {e}") from e
    try:
        from trials.child_qc import NonperformanceRule

        NonperformanceRule.from_task_cfg(task_cfg)
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}") from e
    dtypes = {c.name: c.dtype for c in schema.columns}
    for sn, sd in task_cfg.analysis_subsets.items():
        for k in ("require_true", "require_false"):
            for col in sd.get(k) or []:
                if col not in dtypes:
                    raise ConfigError(f"{where}.analysis_subsets.{sn}.{k}: {col!r} is not a column of the "
                                      f"{task_cfg.name} trial table")
                if dtypes[col] not in _BOOL_DTYPES:
                    raise ConfigError(f"{where}.analysis_subsets.{sn}.{k}: {col!r} has dtype {dtypes[col]!r}; "
                                      f"subset flags must be boolean columns")
