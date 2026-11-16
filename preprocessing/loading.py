"""Load one BioSemi BDF recording for preprocessing.

Loading rules:

- ``mne.io.read_raw_bdf(preload=True)``; MNE types every BDF signal as EEG, so the
  types are set explicitly: EXG1/EXG2 ``eog`` (outer canthi), EXG3/EXG4 ``misc``
  (mastoids), the Status channel ``stim`` (kept, so every derivative carries the
  trigger channel sample-aligned);
- EXG5-8 are dropped. Before dropping, each is checked for the "unconnected"
  signature (median raw value below the -250 mV rail, or a 1-40 Hz SD below
  2 uV); a channel that fails the check is still dropped, and the QC says so;
- montage ``biosemi64`` on the 64 EEG channels (``on_missing='raise'``).

Known issues (known-issues registry) that act at this stage:

- ``exclude_recording`` with scope ``preprocessing``: :func:`preprocessing_exclusions`
  (the pipeline skips the recording);
- ``no_exg``: the file has 65 signals; loading works without EXG channels and
  records ``has_exg = False`` (the ocular module uses its Fp-only fallback);
- ``set_channel_status`` (status ``bad``): the channels go to :attr:`LoadedRecording.known_bads`;
- truncation rules: ``crop_signal_after_s`` and ``crop_to_task_events`` become
  *unusable spans* (:func:`known_issue_spans`). Nothing is cropped: the sample
  axis of every derivative equals the BDF's (same ``n_times``, ``first_samp``
  0), so trial-table event samples index the preprocessed data directly;
- a BDF holding fewer bytes than its header promises raises
  :class:`~trials.status.BdfTruncatedError` before MNE reads it.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preprocessing.config import PreprocessingConfig
from preprocessing.segments import as_spans
from trials.status import BdfHeader, DecodedStatus, decode_status_file, read_bdf_header
from utils.bids_index import Recording
from utils.config import StatusConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)


class RecordingExcludedError(RuntimeError):
    """Raised when the known-issues registry excludes a recording from preprocessing."""


@dataclass
class LoadedRecording:
    """A BDF recording loaded and typed for preprocessing.

    Attributes
    ----------
    raw
        Preloaded float64 :class:`mne.io.Raw`: 64 EEG (montage set), EXG1-4 as
        eog/misc when present, Status as stim; EXG5-8 dropped. Volts.
    header
        Parsed BDF header.
    decoded
        Decoded Status channel (trigger onsets, joins).
    has_exg
        Whether EXG channels exist in the file.
    dropped
        Per dropped channel: ``median_v``, ``sd_uv`` (1-40 Hz), ``unconnected``.
    known_bads
        Channels marked bad by ``set_channel_status`` known issues (present channels only).
    unusable_spans, unusable_reasons
        Known-issue spans (``[start, stop)`` samples) and their labels.
    issue_ids, flags, warnings
        Known-issue ids that apply, ``flag_only`` flags, and loading warnings.
    """

    raw: object
    header: BdfHeader
    decoded: DecodedStatus
    has_exg: bool
    eeg_names: list[str]
    eog_names: list[str]
    misc_names: list[str]
    dropped: dict[str, dict]
    known_bads: list[str]
    unusable_spans: np.ndarray
    unusable_reasons: tuple[str, ...]
    issue_ids: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    path: Path | None = None

    @property
    def sfreq(self) -> float:
        return float(self.raw.info["sfreq"])  # type: ignore[attr-defined]

    @property
    def n_times(self) -> int:
        return int(self.raw.n_times)  # type: ignore[attr-defined]

    def qc(self) -> dict:
        """JSON-ready loading summary."""
        return {"has_exg": self.has_exg, "n_eeg": len(self.eeg_names), "eog": self.eog_names, "misc": self.misc_names,
                "dropped": self.dropped, "known_bads": self.known_bads, "issue_ids": self.issue_ids,
                "flags": self.flags, "warnings": self.warnings, "n_records": self.header.n_records,
                "joins_off_record_boundary": int(self.decoded.joins_off_record_boundary),
                "join_bit_high_at_start": bool(self.decoded.join_bit_high_at_start)}


# --------------------------------------------------------------------------- known issues
def preprocessing_exclusions(rec: Recording, registry: KnownIssuesRegistry | None) -> list[str]:
    """Ids of known issues that exclude ``rec`` from preprocessing (empty list if none)."""
    if registry is None:
        return []
    return sorted({i.id for i, a in registry.actions(rec.subject, rec.task, rec.run, "exclude_recording")
                   if "preprocessing" in a.scopes})


def known_issue_spans(rec: Recording, registry: KnownIssuesRegistry | None, n_times: int,
                      sfreq: float) -> tuple[np.ndarray, tuple[str, ...]]:
    """Unusable spans from ``crop_signal_after_s`` and ``crop_to_task_events`` actions.

    ``crop_to_task_events`` keeps ``[first event - pad_s, last event + pad_s]`` of
    the recording's events.tsv and marks the rest unusable.
    """
    spans: list[tuple[int, int]] = []
    reasons: list[str] = []
    if registry is None:
        return as_spans(None), ()
    for iss, act in registry.actions(rec.subject, rec.task, rec.run):
        if act.type == "crop_signal_after_s":
            a = int(np.floor(float(act.get("t_s")) * sfreq))
            if a < n_times:
                spans.append((max(a, 0), n_times))
                reasons.append(f"{iss.id}:crop_signal_after_s")
        elif act.type == "crop_to_task_events":
            if rec.events_path is None:
                log.warning("%s: crop_to_task_events without an events file; not applied", rec.stem)
                continue
            from trials.io import read_events_tsv

            ev = read_events_tsv(rec.events_path)
            if not len(ev):
                continue
            pad = int(round(float(act.get("pad_s", 0.0) or 0.0) * sfreq))
            lo = max(int(ev["sample"].min()) - pad, 0)
            hi = min(int(ev["sample"].max()) + pad + 1, n_times)
            if lo > 0:
                spans.append((0, lo))
                reasons.append(f"{iss.id}:crop_to_task_events")
            if hi < n_times:
                spans.append((hi, n_times))
                reasons.append(f"{iss.id}:crop_to_task_events")
    return as_spans(spans), tuple(reasons)


def known_bad_channels(rec: Recording, registry: KnownIssuesRegistry | None) -> list[str]:
    """Channels with a ``set_channel_status: bad`` action for ``rec``."""
    if registry is None:
        return []
    out: list[str] = []
    for _, act in registry.actions(rec.subject, rec.task, rec.run, "set_channel_status"):
        if str(act.get("status")).lower() == "bad":
            out.extend(str(c) for c in act.get("channels") or [])
    return sorted(set(out))


# --------------------------------------------------------------------------- unconnected check
def _join_segments(n_times: int, joins: np.ndarray) -> np.ndarray:
    edges = np.concatenate([[0], np.asarray(joins, np.int64), [n_times]])
    s = np.stack([edges[:-1], edges[1:]], axis=1)
    return s[s[:, 1] > s[:, 0]]


def unconnected_check(data: np.ndarray, sfreq: float, joins: np.ndarray, rail_v: float,
                      max_sd_uv: float) -> list[dict]:
    """Per channel of ``data`` (volts): raw median, 1-40 Hz SD (filtered per join segment), unconnected verdict.

    The SD is the median of the per-segment SDs after trimming one filter
    half-length at both segment ends (segments shorter than three half-lengths
    are skipped).
    """
    from mne.filter import create_filter, filter_data

    h = create_filter(None, sfreq, 1.0, 40.0, fir_design="firwin", fir_window="hamming", phase="zero", verbose=False)
    half = len(h) // 2
    out = []
    segs = _join_segments(data.shape[1], joins)
    sds = np.full((data.shape[0], len(segs)), np.nan)
    for k, (a, b) in enumerate(segs):
        if b - a < 3 * half:
            continue
        x = filter_data(data[:, a:b].astype(np.float64), sfreq, 1.0, 40.0, fir_design="firwin",
                        fir_window="hamming", phase="zero", verbose=False)
        sds[:, k] = x[:, half:-half].std(axis=1) * 1e6
    for i in range(data.shape[0]):
        med = float(np.median(data[i]))
        sd = float(np.nanmedian(sds[i])) if np.isfinite(sds[i]).any() else float("nan")
        unconn = bool(med < rail_v or (np.isfinite(sd) and sd < max_sd_uv))
        out.append({"median_v": med, "sd_uv": sd, "unconnected": unconn})
    return out


# --------------------------------------------------------------------------- loading
def read_bdf(path: str | os.PathLike[str]):
    """``mne.io.read_raw_bdf(preload=True)`` with all channels typed EEG except Status (stim)."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mne.io.read_raw_bdf(path, preload=True, infer_types=False, verbose="error")


def load_bdf(rec: Recording, cfg: PreprocessingConfig, status_cfg: StatusConfig,
             registry: KnownIssuesRegistry | None = None, *, n_eeg_expected: int = 64) -> LoadedRecording:
    """Load, type and check one recording (see the module docstring).

    Parameters
    ----------
    rec
        Recording from the :class:`~utils.bids_index.RecordingIndex`.
    cfg
        Preprocessing config.
    status_cfg
        Status decoding parameters (``configs/events.yaml``).
    registry
        Known-issues registry.
    n_eeg_expected
        Number of EEG channels the file must have.

    Raises
    ------
    RecordingExcludedError
        If a known issue excludes the recording from preprocessing.
    trials.status.BdfTruncatedError
        If the file is truncated.
    ValueError
        If the channel set or the sample count is not as expected.
    """
    excl = preprocessing_exclusions(rec, registry)
    if excl:
        raise RecordingExcludedError(f"{rec.stem}: excluded from preprocessing by {', '.join(excl)}")
    if rec.fmt != "bdf":
        raise ValueError(f"{rec.stem}: only BDF recordings are preprocessed (got {rec.fmt})")
    hdr = read_bdf_header(rec.data_path)          # raises BdfTruncatedError on a short file
    raw = read_bdf(rec.data_path)
    lc = cfg.loading
    st_name = next((c for c in raw.ch_names if c.lower().startswith(lc.status_channel.lower())), None)
    if st_name is None:
        raise ValueError(f"{rec.stem}: no {lc.status_channel} channel")
    sfreq = float(raw.info["sfreq"])
    rec_samples = int(round(hdr.record_duration_s * sfreq))
    if raw.n_times != hdr.n_records * rec_samples or raw.first_samp != 0:
        raise ValueError(f"{rec.stem}: MNE read {raw.n_times} samples (first_samp {raw.first_samp}); the header "
                         f"promises {hdr.n_records} records x {rec_samples}")

    # Status decoding as for the trial tables (memory-mapped Status bytes of the same file)
    dec = decode_status_file(rec.data_path, status_cfg)
    if dec.n_times != raw.n_times:
        raise ValueError(f"{rec.stem}: Status has {dec.n_times} samples, MNE read {raw.n_times}")

    exg_all = set(lc.eog_channels) | set(lc.misc_channels) | set(lc.drop_channels)
    present = set(raw.ch_names)
    eeg = [c for c in raw.ch_names if c not in exg_all and c != st_name]
    if len(eeg) != n_eeg_expected:
        raise ValueError(f"{rec.stem}: {len(eeg)} EEG channels, expected {n_eeg_expected} (non-EXG, non-Status)")
    issue_ids = sorted({i.id for i in (registry.for_recording(rec.subject, rec.task, rec.run) if registry else [])})
    flags = sorted({str(a.get("flag")) for _, a in (registry.actions(rec.subject, rec.task, rec.run, "flag_only")
                                                    if registry else [])})
    warn: list[str] = []
    no_exg_issue = bool(registry and registry.actions(rec.subject, rec.task, rec.run, "no_exg"))
    has_exg = bool(exg_all & present)
    if no_exg_issue and has_exg:
        warn.append("known issue says no_exg but EXG channels are present")
    if not has_exg and not no_exg_issue:
        warn.append("no EXG channels in the file and no no_exg known issue")
    missing_exg = sorted((set(lc.eog_channels) | set(lc.misc_channels)) - present) if has_exg else []
    if missing_exg:
        warn.append(f"EXG channels missing: {missing_exg}")

    # unconnected check, then drop
    drop = [c for c in lc.drop_channels if c in present]
    dropped: dict[str, dict] = {}
    if drop:
        res = unconnected_check(raw.get_data(picks=drop), sfreq, dec.joins, lc.unconnected_rail_v,
                                lc.unconnected_max_sd_uv)
        dropped = dict(zip(drop, res))
        conn = [c for c, r in dropped.items() if not r["unconnected"]]
        if conn:
            warn.append(f"dropped channels without the unconnected signature: {conn}")
        raw.drop_channels(drop)
    types = {c: "eog" for c in lc.eog_channels if c in present}
    types.update({c: "misc" for c in lc.misc_channels if c in present})
    types[st_name] = "stim"
    raw.set_channel_types(types, on_unit_change="ignore", verbose="error")
    raw.set_montage(lc.montage, on_missing="raise", verbose="error")

    known_bads = [c for c in known_bad_channels(rec, registry) if c in raw.ch_names]
    ignored = sorted(set(known_bad_channels(rec, registry)) - set(known_bads))
    if ignored:
        log.info("%s: known-bad channels not in the data (dropped or absent): %s", rec.stem, ignored)
    un_spans, un_reasons = known_issue_spans(rec, registry, raw.n_times, sfreq)
    return LoadedRecording(
        raw=raw, header=hdr, decoded=dec, has_exg=has_exg, eeg_names=eeg,
        eog_names=[c for c in lc.eog_channels if c in raw.ch_names],
        misc_names=[c for c in lc.misc_channels if c in raw.ch_names], dropped=dropped, known_bads=known_bads,
        unusable_spans=un_spans, unusable_reasons=un_reasons,
        issue_ids=issue_ids, flags=flags, warnings=warn, path=Path(rec.data_path))
