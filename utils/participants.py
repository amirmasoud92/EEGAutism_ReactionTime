"""Participant covariates from ``participants.tsv``.

The covariates (default age, sex, handedness, fsiq) and their ``_dup`` copies
are read from ``participants.tsv``.

Coalescing: ``value = original if present else _dup``; ``<col>_source`` records
``orig`` / ``dup`` / ``both_equal`` / ``both_differ`` (original kept) /
``missing``. Missing tokens (``n/a``, ``NAN``, ``-``, empty) become NA.
Subjects that have a directory but no row are returned with
``has_participants_row = False`` and NA covariates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from utils.config import ProjectConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)
SOURCE_SUFFIX = "_source"
#: Covariates returned by default.
COVARIATES: tuple[str, ...] = ("age", "sex", "handedness", "fsiq")
#: Suffix of the duplicate covariate columns in ``participants.tsv``; values are coalesced with the original.
DUP_SUFFIX = "_dup"
COALESCE_DUP = True
#: Cell values read as missing.
MISSING_TOKENS: tuple[str, ...] = ("n/a", "NAN", "nan", "NaN", "-", "")
#: Covariates converted to numbers.
NUMERIC_COVARIATES: tuple[str, ...] = ("age", "fsiq")


def normalise_participant_id(pid: str | int) -> str:
    """Participant label with the BIDS ``sub-`` prefix (added if missing)."""
    s = str(pid).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def _clean(s: pd.Series, missing: set[str]) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s.isin(missing) | s.isna(), pd.NA)


def coalesce(orig: pd.Series, dup: pd.Series | None) -> tuple[pd.Series, pd.Series]:
    """Coalesce an original column with its ``_dup`` copy.

    Parameters
    ----------
    orig, dup
        Cleaned string series (NA = missing). ``dup`` may be ``None``.

    Returns
    -------
    (values, source)
        Coalesced values and the source flag per row.
    """
    if dup is None:
        src = pd.Series(np.where(orig.notna(), "orig", "missing"), index=orig.index, dtype="string")
        return orig, src
    o, d = orig.notna(), dup.notna()
    equal = o & d & (orig.fillna("") == dup.fillna(""))
    src = np.select([o & d & equal, o & d & ~equal, o & ~d, ~o & d], ["both_equal", "both_differ", "orig", "dup"],
                    default="missing")
    vals = orig.where(o, dup)
    return vals, pd.Series(src, index=orig.index, dtype="string")


def load_participants(
    cfg: ProjectConfig,
    columns: Iterable[str] | None = None,
    subjects: Iterable[str] | None = None,
    registry: KnownIssuesRegistry | None = None,
) -> pd.DataFrame:
    """Load participant covariates.

    Parameters
    ----------
    cfg
        Project config (``paths.bids_root``).
    columns
        Covariates to return (default: :data:`COVARIATES`).
    subjects
        If given (e.g. ``RecordingIndex.subjects``), the result has exactly one
        row per subject, including rowless subjects.
    registry
        Known-issues registry; ``set_value_missing`` actions are applied.

    Returns
    -------
    pandas.DataFrame
        ``participant_id``, ``has_participants_row``, the covariates and one
        ``<col>_source`` column per covariate.
    """
    cols = list(columns) if columns is not None else list(COVARIATES)
    path = cfg.paths.bids_root / "participants.tsv"
    header = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8-sig").columns.tolist()
    missing_cols = [c for c in cols if c not in header]
    if missing_cols:
        raise KeyError(f"participants.tsv lacks columns {missing_cols}")
    use = ["participant_id"] + cols + ([c + DUP_SUFFIX for c in cols if c + DUP_SUFFIX in header]
                                       if COALESCE_DUP else [])
    raw = pd.read_csv(path, sep="\t", dtype=str, usecols=use, keep_default_na=False, encoding="utf-8-sig")
    miss = set(MISSING_TOKENS)
    out = pd.DataFrame({"participant_id": raw["participant_id"].map(normalise_participant_id)})
    if out["participant_id"].duplicated().any():
        raise ValueError("participants.tsv has duplicated participant_id values")
    for c in cols:
        dup = _clean(raw[c + DUP_SUFFIX], miss) if (COALESCE_DUP and c + DUP_SUFFIX in raw) else None
        vals, src = coalesce(_clean(raw[c], miss), dup)
        if c in NUMERIC_COVARIATES:
            vals = pd.to_numeric(vals, errors="coerce").astype("Float64")
        out[c] = vals
        out[c + SOURCE_SUFFIX] = src
    out["has_participants_row"] = True
    if registry is not None:
        for iss in registry:
            for a in iss.actions:
                if a.type == "set_value_missing" and a.get("column") in cols:
                    hit = out["participant_id"] == iss.subject
                    if hit.any():
                        out.loc[hit, a.get("column")] = pd.NA
                        out.loc[hit, a.get("column") + SOURCE_SUFFIX] = f"set_missing:{iss.id}"
    if subjects is not None:
        subs = pd.DataFrame({"participant_id": sorted({normalise_participant_id(s) for s in subjects})})
        extra = set(out["participant_id"]) - set(subs["participant_id"])
        out = subs.merge(out, on="participant_id", how="left")
        out["has_participants_row"] = out["has_participants_row"].eq(True)
        for c in cols:
            out[c + SOURCE_SUFFIX] = out[c + SOURCE_SUFFIX].fillna("no_row")
        if extra:
            log.info("%d participants rows have no subject directory in the given subject list", len(extra))
    out = out.sort_values("participant_id").reset_index(drop=True)
    return out
