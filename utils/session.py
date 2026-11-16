"""Recording start times, day assignment and time in session.

- Start time per BDF comes from the header (``mne.io.read_raw_bdf(preload=False)``
  -> ``info['meas_date']``); no data are loaded. Tasks in ``tasks.undated``
  (motor, BrainVision without timestamps) get no time.
- Per child, recordings are grouped by calendar date. The main day is the date
  with the most recordings counted by ``session.main_day_basis`` (default: all
  dated, non-excluded, non-motor recordings); ties go to the earliest date.
- ``time_in_session_h`` is the start time relative to the first counted
  recording of the same child and date.

:func:`relative_session_table` drops every absolute-time column.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.bids_index import Recording, RecordingIndex
from utils.config import ProjectConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)
ABSOLUTE_COLUMNS = ("meas_date", "date", "end_time")


def read_meas_date(path: str | Path) -> tuple[datetime | None, float | None]:
    """Recording start (UTC, tz-aware) and duration (s) from a BDF header.

    Returns ``(None, None)`` on failure (logged).
    """
    import mne

    try:
        raw = mne.io.read_raw_bdf(path, preload=False, verbose="error")
    except Exception as e:  # noqa: BLE001 - header problems are reported, not fatal
        log.warning("cannot read header of %s: %r", Path(path).name, e)
        return None, None
    md = raw.info["meas_date"]
    dur = raw.n_times / raw.info["sfreq"]
    return (md.astimezone(timezone.utc) if md is not None else None), float(dur)


def _header_row(rec: Recording, undated: set[str]) -> dict:
    row = {"subject": rec.subject, "task": rec.task, "run": rec.run, "fmt": rec.fmt}
    if rec.task in undated or rec.fmt != "bdf":
        row.update(meas_date=None, duration_s=np.nan)
    else:
        md, dur = read_meas_date(rec.data_path)
        row.update(meas_date=md, duration_s=dur if dur is not None else np.nan)
    return row


def read_start_times(index: RecordingIndex, undated: set[str], n_jobs: int = 1) -> pd.DataFrame:
    """Header start times for every recording of ``index`` (parallel over files)."""
    recs = list(index)
    if n_jobs == 1 or len(recs) < 8:
        rows = [_header_row(r, undated) for r in recs]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(_header_row)(r, undated) for r in recs)
    df = pd.DataFrame(rows)
    df["meas_date"] = pd.to_datetime(df["meas_date"], utc=True)
    return df


def assign_days(df: pd.DataFrame, task_recordings: tuple[str, ...] | list[str] | None = None,
                basis: str = "non_motor", undated: tuple[str, ...] = ("motor",)) -> pd.DataFrame:
    """Add day index, main-day flag and time in session.

    Parameters
    ----------
    df
        Columns ``subject``, ``task``, ``run``, ``meas_date`` (tz-aware or
        naive datetimes; NaT = undated), ``duration_s`` and optional boolean
        ``exclude_from_order``.
    task_recordings
        Tasks counted when ``basis == "tasks_only"``.
    basis
        ``"non_motor"`` (count every dated non-undated recording) or
        ``"tasks_only"``.
    undated
        Task names never counted.

    Returns
    -------
    pandas.DataFrame
        Input plus ``date``, ``day_index`` (1-based chronological), ``n_days``,
        ``is_main_day``, ``time_in_session_h``, ``order_in_day``, ``end_time``.
    """
    out = df.copy()
    if "exclude_from_order" not in out:
        out["exclude_from_order"] = False
    out["meas_date"] = pd.to_datetime(out["meas_date"])
    out["date"] = out["meas_date"].dt.date
    out["end_time"] = out["meas_date"] + pd.to_timedelta(out["duration_s"], unit="s")
    counted = out["meas_date"].notna() & ~out["exclude_from_order"].astype(bool) & ~out["task"].isin(list(undated))
    if basis == "tasks_only":
        counted &= out["task"].isin(list(task_recordings or ()))
    elif basis != "non_motor":
        raise ValueError(f"unknown basis {basis!r}")
    out["_counted"] = counted
    out["day_index"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["n_days"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["is_main_day"] = False
    out["time_in_session_h"] = np.nan
    out["order_in_day"] = pd.array([pd.NA] * len(out), dtype="Int64")
    for sub, g in out.groupby("subject", sort=True):
        dated = g[g["meas_date"].notna()]
        days = sorted(dated["date"].unique())
        if not days:
            continue
        dmap = {d: i + 1 for i, d in enumerate(days)}
        out.loc[dated.index, "day_index"] = dated["date"].map(dmap).astype("int64").to_numpy()
        out.loc[g.index, "n_days"] = len(days)
        cg = g[g["_counted"]]
        if len(cg):
            counts = cg.groupby("date").size()
            best = max(counts.items(), key=lambda kv: (kv[1], -dmap[kv[0]]))[0]
            out.loc[g.index, "is_main_day"] = (g["date"] == best).to_numpy()
            first = cg.groupby("date")["meas_date"].min()
            for d, t0 in first.items():
                idx = g.index[(g["date"] == d).to_numpy()]
                out.loc[idx, "time_in_session_h"] = ((out.loc[idx, "meas_date"] - t0).dt.total_seconds() / 3600.0).to_numpy()
                cidx = cg.index[(cg["date"] == d).to_numpy()]
                order = out.loc[cidx].sort_values(["meas_date", "task", "run"]).index
                out.loc[order, "order_in_day"] = np.arange(1, len(order) + 1)
    return out.drop(columns=["_counted"])


def build_session_table(cfg: ProjectConfig, index: RecordingIndex, registry: KnownIssuesRegistry | None = None,
                        n_jobs: int | None = None) -> pd.DataFrame:
    """Session table for every recording (absolute dates included)."""
    st = read_start_times(index, set(cfg.tasks.undated), n_jobs or cfg.runtime.n_jobs)
    if registry is not None:
        st["exclude_from_order"] = [registry.is_excluded(s, t, r, "session_order")
                                    for s, t, r in zip(st["subject"], st["task"], st["run"])]
    else:
        st["exclude_from_order"] = False
    return assign_days(st, cfg.session.task_recordings, cfg.session.main_day_basis, cfg.tasks.undated)


def relative_session_table(df: pd.DataFrame) -> pd.DataFrame:
    """Drop absolute dates/times."""
    return df.drop(columns=[c for c in ABSOLUTE_COLUMNS if c in df.columns])
