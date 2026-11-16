"""events.tsv reading and integrity checks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("onset", "sample", "value")


class EventsIntegrityError(ValueError):
    """Raised when an events table violates an integrity rule (order, sample/onset agreement)."""


def read_events_tsv(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a BIDS events.tsv (handles the UTF-8 BOM present in ds006780).

    Returns
    -------
    pandas.DataFrame
        Columns ``onset`` (float), ``sample`` (int64), ``value`` (int64),
        ``trial_type`` (string, if present) and ``duration`` (if present), in
        file order.

    Raises
    ------
    EventsIntegrityError
        If the file cannot be parsed, lacks a required column, or a required
        column holds a missing, non-numeric or (``sample``, ``value``)
        non-integer value; the message names the file, the column and the row.
    """
    name = Path(path).name
    try:
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", keep_default_na=False, na_values=["n/a"])
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        raise EventsIntegrityError(f"{name}: cannot parse as a tab-separated events file: {e}") from e
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise EventsIntegrityError(f"{name}: missing columns {missing}")

    def num(col: str, dtype: str) -> pd.Series:
        s = df[col]
        x = pd.to_numeric(s, errors="coerce")
        bad = x.isna()
        if bad.any():
            i = int(np.flatnonzero(bad.to_numpy())[0])
            raise EventsIntegrityError(f"{name}: column {col!r} has {int(bad.sum())} missing or non-numeric value(s); "
                                       f"first at data row {i + 1}: {s.iloc[i]!r}")
        if dtype == "int64" and not np.all(np.mod(x.to_numpy(float), 1) == 0):
            i = int(np.flatnonzero(np.mod(x.to_numpy(float), 1) != 0)[0])
            raise EventsIntegrityError(f"{name}: column {col!r} must hold integers; data row {i + 1}: {s.iloc[i]!r}")
        return x.astype(dtype)

    out = pd.DataFrame({"onset": num("onset", "float64"), "sample": num("sample", "int64"),
                        "value": num("value", "int64")})
    if "trial_type" in df.columns:
        out["trial_type"] = df["trial_type"].astype("string")
    if "duration" in df.columns:
        out["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    return out


def validate_events(events: pd.DataFrame, sfreq: float, sample_tolerance: int = 0) -> None:
    """Check ordering and onset/sample agreement.

    Parameters
    ----------
    events
        Table with ``onset`` (s) and ``sample`` columns, in file order.
    sfreq
        Sampling rate.
    sample_tolerance
        Maximum allowed ``|sample - round(onset * sfreq)|``.

    Raises
    ------
    EventsIntegrityError
        If onsets or samples are non-monotonic (decreasing) or disagree.
    """
    if len(events) == 0:
        return
    s = events["sample"].to_numpy()
    on = events["onset"].to_numpy(dtype=float)
    if np.any(np.diff(on) < 0) or np.any(np.diff(s) < 0):
        i = int(np.flatnonzero((np.diff(on) < 0) | (np.diff(s) < 0))[0])
        raise EventsIntegrityError(f"non-monotonic onsets at rows {i}->{i + 1}")
    if np.any(s < 0):
        raise EventsIntegrityError("negative sample indices")
    d = np.abs(s - np.round(on * sfreq).astype(np.int64))
    if np.any(d > sample_tolerance):
        raise EventsIntegrityError(f"sample != round(onset*sfreq) for {int(np.sum(d > sample_tolerance))} rows "
                                   f"(max |diff| {int(d.max())} samples)")
