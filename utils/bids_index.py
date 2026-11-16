"""Index of every EEG recording in the BIDS tree.

:class:`RecordingIndex` enumerates ``sub-*/eeg/*_eeg.(bdf|vhdr)`` files with
their sidecars. It reads directory listings only (no data, no participants
file), so it never touches diagnostic information.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from utils.known_issues import KnownIssuesRegistry

FNAME_RE = re.compile(
    r"^(?P<subject>sub-[A-Za-z0-9]+)_task-(?P<task>[A-Za-z0-9]+)(?:_run-(?P<run>\d+))?_eeg\.(?P<ext>bdf|vhdr)$"
)


@dataclass(frozen=True)
class Recording:
    """One EEG recording and its BIDS sidecars.

    Attributes
    ----------
    subject, task, run
        BIDS entities (``run`` defaults to 1 when the entity is absent).
    data_path
        ``*_eeg.bdf`` or ``*_eeg.vhdr``.
    fmt
        ``"bdf"`` or ``"vhdr"``.
    events_path, events_json_path, channels_path, eeg_json_path
        Sidecar paths, or ``None`` if the file does not exist.
    size_bytes
        Size of the data file (``.eeg`` payload for BrainVision).
    """

    subject: str
    task: str
    run: int
    data_path: Path
    fmt: str
    events_path: Path | None
    events_json_path: Path | None
    channels_path: Path | None
    eeg_json_path: Path | None
    size_bytes: int

    @property
    def has_events(self) -> bool:
        return self.events_path is not None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.subject, self.task, self.run)

    @property
    def stem(self) -> str:
        return self.data_path.name[: self.data_path.name.rfind("_eeg.")]


def _sidecar(eeg_dir: Path, names: set[str], name: str) -> Path | None:
    return eeg_dir / name if name in names else None


class RecordingIndex:
    """All recordings of a BIDS dataset.

    Parameters
    ----------
    recordings
        Recordings (normally from :meth:`from_bids`).
    bids_root
        Dataset root.
    """

    def __init__(self, recordings: Iterable[Recording], bids_root: Path) -> None:
        recs = sorted(recordings, key=lambda r: (r.subject, r.task, r.run))
        keys = [r.key for r in recs]
        if len(set(keys)) != len(keys):
            dup = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate recordings (subject, task, run): {dup[:5]}")
        self.recordings: tuple[Recording, ...] = tuple(recs)
        self.bids_root = Path(bids_root)
        self._by_key = {r.key: r for r in recs}

    @classmethod
    def from_bids(cls, bids_root: str | os.PathLike[str]) -> RecordingIndex:
        """Scan ``bids_root`` for EEG recordings.

        Parameters
        ----------
        bids_root
            Dataset root containing ``sub-*`` directories.
        """
        root = Path(bids_root)
        if not root.is_dir():
            raise FileNotFoundError(f"BIDS root not found: {root}")
        recs: list[Recording] = []
        for sub_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
            eeg = sub_dir / "eeg"
            if not eeg.is_dir():
                continue
            names = set(os.listdir(eeg))
            for f in sorted(names):
                m = FNAME_RE.match(f)
                if not m or m["subject"] != sub_dir.name:
                    continue
                stem = f[: f.rfind("_eeg.")]
                data = eeg / f
                size_src = data if m["ext"] == "bdf" else eeg / f.replace(".vhdr", ".eeg")
                recs.append(Recording(
                    subject=m["subject"], task=m["task"], run=int(m["run"]) if m["run"] else 1,
                    data_path=data, fmt=m["ext"],
                    events_path=_sidecar(eeg, names, f"{stem}_events.tsv"),
                    events_json_path=_sidecar(eeg, names, f"{stem}_events.json"),
                    channels_path=_sidecar(eeg, names, f"{stem}_channels.tsv"),
                    eeg_json_path=_sidecar(eeg, names, f"{stem}_eeg.json"),
                    size_bytes=size_src.stat().st_size if size_src.exists() else -1,
                ))
        return cls(recs, root)

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.recordings)

    def __iter__(self) -> Iterator[Recording]:
        return iter(self.recordings)

    def get(self, subject: str, task: str, run: int = 1) -> Recording:
        """Look up one recording; raises ``KeyError`` if absent."""
        return self._by_key[(subject, task, int(run))]

    def filter(self, task: str | Iterable[str] | None = None, subject: str | Iterable[str] | None = None,
               has_events: bool | None = None) -> RecordingIndex:
        """Subset by task(s), subject(s) and/or events availability."""
        tasks = {task} if isinstance(task, str) else (set(task) if task is not None else None)
        subs = {subject} if isinstance(subject, str) else (set(subject) if subject is not None else None)
        out = [r for r in self.recordings
               if (tasks is None or r.task in tasks) and (subs is None or r.subject in subs)
               and (has_events is None or r.has_events == has_events)]
        return RecordingIndex(out, self.bids_root)

    @property
    def subjects(self) -> list[str]:
        return sorted({r.subject for r in self.recordings})

    @property
    def tasks(self) -> list[str]:
        return sorted({r.task for r in self.recordings})

    def runs(self, subject: str, task: str) -> list[Recording]:
        """All runs of one subject and task, sorted by run number."""
        return [r for r in self.recordings if r.subject == subject and r.task == task]

    def runs_by_subject(self, task: str) -> dict[str, list[Recording]]:
        """Mapping subject -> runs of ``task`` (multi-run handling)."""
        out: dict[str, list[Recording]] = {}
        for r in self.recordings:
            if r.task == task:
                out.setdefault(r.subject, []).append(r)
        return out

    def to_frame(self, registry: KnownIssuesRegistry | None = None) -> pd.DataFrame:
        """Tabular view: one row per recording.

        Adds ``n_runs`` / ``is_multi_run`` per subject and task and, when a
        registry is given, ``known_issue_ids`` plus one ``excluded_<scope>``
        boolean per exclusion scope.
        """
        rows = []
        for r in self.recordings:
            d = asdict(r)
            for k in ("data_path", "events_path", "events_json_path", "channels_path", "eeg_json_path"):
                d[k] = str(d[k]) if d[k] is not None else None
            d["has_events"] = r.has_events
            rows.append(d)
        df = pd.DataFrame(rows, columns=[
            "subject", "task", "run", "data_path", "fmt", "events_path", "events_json_path",
            "channels_path", "eeg_json_path", "size_bytes", "has_events"])
        if len(df):
            df["n_runs"] = df.groupby(["subject", "task"])["run"].transform("size").astype("int64")
            df["is_multi_run"] = df["n_runs"] > 1
            df["run_rank"] = df.groupby(["subject", "task"])["run"].rank(method="first").astype("int64")
        if registry is not None:
            ids, excl = [], {s: [] for s in ("trials", "preprocessing", "iaf", "session_order")}
            for r in self.recordings:
                ids.append(";".join(i.id for i in registry.for_recording(r.subject, r.task, r.run)))
                for s in excl:
                    excl[s].append(registry.is_excluded(r.subject, r.task, r.run, s))
            df["known_issue_ids"] = ids
            for s, v in excl.items():
                df[f"excluded_{s}"] = v
        return df
