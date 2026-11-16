"""Preprocessing of one recording: order of operations and stored outputs.

Order of operations (:meth:`PreprocessingPipeline.process`):

1. :func:`~preprocessing.loading.load_bdf` - BDF, channel types, EXG5-8
   check and drop, montage, Status decoding (joins), known issues.
2. :func:`~preprocessing.segments.build_segment_info` on the raw EEG -
   rail / flat / unusable spans, valid pieces shorter than the FIR kernel
   (``BAD_short``) and the valid spans; annotations (``EDGE boundary``,
   ``BAD_rail``, ``BAD_flat``, ``BAD_unusable``, ``BAD_short``); per-channel
   saturation fractions. A recording without any valid span gets a QC record with
   ``status: unusable`` (and its segments file), nothing else.
3. Analysis copy: EEG + EXG1-4, 1-40 Hz FIR per valid span (notch only if
   ``line_noise.analysis_strategy == 'notch'``); bad-span samples set to 0.
4. 1-100 Hz copy (EEG), per valid span; notched detection copy ->
   :func:`~preprocessing.bad_channels.find_bad_channels` (PyPREP within a robust
   average reference).
5. EXG QC and ocular copy on the *unfiltered* loaded Raw (EXG5-8 already dropped):
   :func:`~preprocessing.exg_qc.exg_qc` (pass/fail per EXG1-4; failed channels are
   also marked in the analysis copy's ``info['bads']``), then
   :func:`~preprocessing.ocular.build_ocular_copy` with that QC result and the
   hard-fault bad EEG channels.
6. ICA on the 1-100 Hz copy of the good channels
   (:func:`~preprocessing.ica.run_ica`: Picard, ICLabel, P(eye) >= 0.8), then the
   re-selection on the fitted solution (:func:`~preprocessing.ica_selection.run_selection`:
   components with posterior alpha are spared; fit-rejected ``BAD_icafit`` spans).
7. Outputs (:mod:`preprocessing.io`): analysis copy, segments, ICA, ICLabel table,
   ICA re-selection, ocular copy and a QC record. The cleaned data are rebuilt from
   these parts by :func:`preprocessing.clean.load_clean`.

The pipeline never reads participants.tsv; it processes whatever recording it is given.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preprocessing import bad_channels as bad_mod
from preprocessing import exg_qc as exg_mod
from preprocessing import filtering as filt_mod
from preprocessing import ica as ica_mod
from preprocessing import ica_selection as sel_mod
from preprocessing import io as io_mod
from preprocessing import loading as load_mod
from preprocessing import ocular as oc_mod
from preprocessing import segments as seg_mod
from preprocessing.config import PreprocessingConfig
from utils import config_ocular as oc_cfg_mod
from utils.bids_index import Recording
from utils.config import EventsConfig, ProjectConfig
from utils.known_issues import KnownIssuesRegistry

log = logging.getLogger(__name__)

#: QC flags that make a recording incomplete
INCOMPLETE_FLAGS = ("badch_detection_incomplete",)


def load_ocular_cfg(prep_cfg: PreprocessingConfig, root: Path) -> oc_cfg_mod.OcularConfig:
    """``<root>/configs/ocular.yaml``, with the ``ocular:`` stub of ``preprocessing.yaml`` checked against it."""
    oc = oc_cfg_mod.load_ocular_config(root=root)
    oc_cfg_mod.check_preprocessing_stub(prep_cfg.ocular or {}, oc)
    return oc


def qc_flags(qc: dict) -> list[str]:
    """Flags derived from a QC record."""
    flags = list(qc.get("qc_flags") or [])
    if (qc.get("bad_channels") or {}).get("errors") and "badch_detection_incomplete" not in flags:
        flags.append("badch_detection_incomplete")
    return flags


@dataclass
class RecordingResult:
    """Outcome of one recording."""

    subject: str
    task: str
    run: int
    stem: str
    status: str             # done | incomplete | unusable
    message: str = ""
    outputs: dict = field(default_factory=dict)
    qc: dict | None = None


class PreprocessingPipeline:
    """Preprocess recordings (see the module docstring).

    Parameters
    ----------
    project_cfg, prep_cfg, events_cfg
        Loaded configs (project configuration, ``preprocessing.yaml``, ``events.yaml``).
    registry
        Known-issues registry.
    ocular_cfg
        ``configs/ocular.yaml`` (default: loaded from the project root and checked against the
        ``ocular:`` stub of ``preprocessing.yaml``).
    """

    def __init__(self, project_cfg: ProjectConfig, prep_cfg: PreprocessingConfig, events_cfg: EventsConfig,
                 registry: KnownIssuesRegistry | None = None,
                 ocular_cfg: oc_cfg_mod.OcularConfig | None = None) -> None:
        self.project_cfg = project_cfg
        self.cfg = prep_cfg
        self.events_cfg = events_cfg
        self.registry = registry
        self.ocular_cfg = ocular_cfg if ocular_cfg is not None else load_ocular_cfg(prep_cfg, project_cfg.root)
        self.sel_cfg = sel_mod.parse_selection_config(prep_cfg.ica_selection)

    # ------------------------------------------------------------------ paths
    @property
    def root(self) -> Path:
        return io_mod.preproc_root(self.project_cfg.paths.derivatives, self.cfg.storage_subdir)

    def outputs(self, rec: Recording) -> dict[str, Path]:
        return io_mod.recording_outputs(self.root, rec)

    # ------------------------------------------------------------------ shared steps
    def _min_valid_samples(self, sf: float) -> int:
        kernels = [len(filt_mod.fir_kernel(self.cfg.filter_analysis, sf)),
                   len(filt_mod.fir_kernel(self.cfg.filter_highpass_copy, sf))]
        return seg_mod.min_valid_samples(self.cfg.segments, kernels, sf)

    def _load_and_segment(self, rec: Recording):
        """Steps 1-2: loaded recording, its segment info (raw annotated) and the saturation fractions."""
        cfg = self.cfg
        loaded = load_mod.load_bdf(rec, cfg, self.events_cfg.status, self.registry,
                                   n_eeg_expected=self.project_cfg.acquisition.n_eeg_channels)
        raw = loaded.raw
        sf = loaded.sfreq
        eeg_idx = [raw.ch_names.index(c) for c in loaded.eeg_names]
        eeg_raw = raw._data[eeg_idx]
        seg = seg_mod.build_segment_info(eeg_raw, sf, loaded.decoded.joins, cfg.segments,
                                         unusable_spans=loaded.unusable_spans,
                                         unusable_reasons=loaded.unusable_reasons, first_samp=raw.first_samp,
                                         min_valid_samples=self._min_valid_samples(sf))
        seg_mod.annotate_raw(raw, seg, cfg.segments.join_annotation)
        sat = bad_mod.saturation_fractions(eeg_raw, seg, loaded.eeg_names, cfg.bad_channels.saturation_threshold_v)
        return loaded, seg, sat

    def _highpass_copy(self, raw, seg: seg_mod.SegmentInfo, eeg: Sequence[str]):
        """The 1-100 Hz per-span copy of the EEG channels (bad spans zeroed)."""
        hp = raw.copy().pick(list(eeg))
        filt_mod.filter_raw_per_segment(hp, seg, self.cfg.filter_highpass_copy, picks=list(eeg))
        if self.cfg.segments.zero_skip_spans:
            seg_mod.zero_spans(hp, seg.bad_spans)
        return hp

    def _ocular(self, raw, seg, rec, exg_qc_res: dict, bads_reasons: dict, known_bads: Sequence[str]):
        """Ocular copy and its QC block; returns (ocular_raw or None, ocular_qc)."""
        # Only hard faults remove an EEG channel from the ocular derivation (registry, saturation, NaN/flat): the
        # statistical PyPREP criteria flag Fp / F7 / F8 for ocular activity itself
        hard = {"manual", "saturation", "nan", "flat"}
        oc_bads = sorted({c for c, rs in bads_reasons.items() if hard & set(rs)} | set(known_bads))
        ocfg = self.ocular_cfg
        try:
            ocular_raw = oc_mod.build_ocular_copy(raw, seg, ocfg, exg_qc_result=exg_qc_res, bad_channels=oc_bads)
            meta = oc_mod.ocular_meta(ocular_raw)
            ocular_qc = {"status": "ok", "n_channels": len(ocular_raw.ch_names),
                         "sfreq": float(ocular_raw.info["sfreq"]), "n_times": int(ocular_raw.n_times),
                         "first_samp": int(ocular_raw.first_samp), "decim": int(meta["decim"]),
                         "sources": meta["sources"], "n_pieces": len(meta["pieces"]),
                         "n_dropped_short_pieces": len(meta["dropped_short_pieces"]),
                         "valid_fraction": float(np.mean(ocular_raw.get_data(picks=[oc_mod.VALID_CH])[0] > 0.5))}
            return ocular_raw, ocular_qc
        except Exception as e:  # noqa: BLE001 - the core outputs are still written; QC says incomplete
            log.error("%s: ocular copy failed: %s", rec.stem, e)
            return None, {"status": "error", "reason": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _write(path: Path, writer) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)
        return path

    def _unusable(self, rec: Recording, paths: dict, seg, loaded) -> RecordingResult:
        """No valid span left: segments file + a QC record that says why (``status: unusable``)."""
        reason = "no valid span left (whole recording rail / flat / unusable / short)"
        self._write(paths["segments"], lambda p: seg.to_json(p))
        qc = {"recording": {"subject": rec.subject, "task": rec.task, "run": rec.run, "stem": rec.stem},
              "complete": False, "status": "unusable", "unusable_reason": reason, "qc_flags": ["unusable"],
              "config_version": self.cfg.version,
              "sample_axis": {"n_times": seg.n_times, "first_samp": seg.first_samp, "sfreq": seg.sfreq},
              "loading": loaded.qc(), "segments": seg.summary(), "outputs": ["qc", "segments"]}
        self._write(paths["qc"], lambda p: io_mod.save_json(qc, p))
        return RecordingResult(subject=rec.subject, task=rec.task, run=rec.run, stem=rec.stem, status="unusable",
                               message=reason, outputs={"segments": paths["segments"], "qc": paths["qc"]}, qc=qc)

    # ------------------------------------------------------------------ one recording
    def process(self, rec: Recording) -> RecordingResult:
        """Run every step for ``rec`` and write the outputs (raises on error)."""
        cfg = self.cfg
        paths = self.outputs(rec)

        # 1-2. load, segments
        loaded, seg, sat = self._load_and_segment(rec)
        raw = loaded.raw
        sf = loaded.sfreq
        eeg = loaded.eeg_names
        exg = loaded.eog_names + loaded.misc_names
        if seg.n_segments == 0:
            return self._unusable(rec, paths, seg, loaded)

        # 3. analysis copy (1-40 Hz)
        ana = raw.copy()
        filt_mod.filter_raw_per_segment(ana, seg, cfg.filter_analysis, picks=eeg + exg)
        if cfg.line_noise.analysis_strategy == "notch":
            filt_mod.notch_raw_per_segment(ana, seg, cfg.line_noise.notch_freqs, picks=eeg + exg, check=False)
        if cfg.segments.zero_skip_spans:
            seg_mod.zero_spans(ana, seg.bad_spans)

        # 4. 1-100 Hz copy, bad channels
        hp = self._highpass_copy(raw, seg, eeg)
        if cfg.bad_channels.notch and cfg.line_noise.notch_freqs:
            det = hp.copy()
            filt_mod.notch_raw_per_segment(det, seg, cfg.line_noise.notch_freqs, picks=eeg, check=False)
        else:
            det = hp
        ddata = bad_mod.detection_data(det, seg, cfg.bad_channels, eeg)
        del det
        manual = loaded.known_bads if cfg.bad_channels.manual_bads_from_known_issues else []
        bads = bad_mod.find_bad_channels(ddata, sf / cfg.bad_channels.decimate, eeg, cfg.bad_channels,
                                         seed=cfg.random_seed, manual=[c for c in manual if c in eeg],
                                         saturation_frac=sat, montage=cfg.loading.montage)
        del ddata

        # 5. EXG QC + ocular copy (on the unfiltered Raw)
        exg_qc_res: dict | None = None
        try:
            exg_qc_res = exg_mod.exg_qc(raw, seg, self.ocular_cfg)
            ocular_raw, ocular_qc = self._ocular(raw, seg, rec, exg_qc_res, bads.reasons, loaded.known_bads)
        except Exception as e:  # noqa: BLE001 - the core outputs are still written; QC says incomplete
            ocular_raw, ocular_qc = None, {"status": "error", "reason": f"{type(e).__name__}: {e}"}
            log.error("%s: EXG QC failed: %s", rec.stem, e)
        # EXG channels that are present but fail QC: marked bad in the analysis copy (they are never EEG)
        exg_bads = sorted(ch for ch, r in (exg_qc_res or {}).items() if r.get("present") and not r.get("pass"))
        del raw, loaded.raw

        # 6. ICA + re-selection on the fitted solution
        good = [c for c in eeg if c not in bads.bads]
        ica_res = None
        ica_qc: dict
        sel: dict | None = None
        if cfg.ica.enabled:
            try:
                if cfg.line_noise.ica_notch and cfg.line_noise.notch_freqs:
                    filt_mod.notch_raw_per_segment(hp, seg, cfg.line_noise.notch_freqs, picks=eeg)
                ica_res = ica_mod.run_ica(hp, seg, good, cfg.ica, cfg.random_seed, cfg.loading.montage,
                                          n_threads=cfg.threads_per_worker)
                ica_qc = {"status": "ok", **ica_res.qc}
            except Exception as e:  # noqa: BLE001
                ica_qc = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
                log.error("%s: ICA failed: %s", rec.stem, e)
            if ica_res is not None:
                try:
                    sel = sel_mod.run_selection(hp, ica_res.fit, ica_res.ica, ica_res.excluded, self.sel_cfg)
                except Exception as e:  # noqa: BLE001
                    sel = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
                    log.error("%s: ICA re-selection failed: %s", rec.stem, e)
                ica_res.fit = None
        else:
            ica_qc = {"status": "disabled"}
        del hp

        # analysis copy: bad channels (EEG from the detector; non-EEG from the registry and the EXG QC)
        non_eeg_bads = sorted({c for c in [*loaded.known_bads, *exg_bads] if c not in eeg and c in ana.ch_names})
        with ana.info._unlock():
            ana.info["bads"] = list(bads.bads) + non_eeg_bads

        # 7. outputs
        fmt = cfg.storage_fmt
        written: dict[str, Path] = {}
        written["preica"] = self._write(paths["preica"], lambda p: io_mod.save_raw_fif(ana, p, fmt))
        written["segments"] = self._write(paths["segments"], lambda p: seg.to_json(p))
        if ica_res is not None:
            written["ica"] = self._write(paths["ica"], lambda p: io_mod.save_ica(ica_res.ica, p))
            written["iclabel"] = self._write(paths["iclabel"], lambda p: ica_res.table().to_csv(
                p, sep="\t", index=False, lineterminator="\n"))
            if sel is not None and sel.get("status") == "ok":
                sel = {**sel, "recording": {"subject": rec.subject, "task": rec.task, "run": rec.run,
                                            "stem": rec.stem}}
                written["icasel"] = self._write(paths["icasel"], lambda p: io_mod.save_json(sel, p))
        if ocular_raw is not None:
            written["ocular"] = self._write(paths["ocular"],
                                            lambda p: oc_mod.save_ocular_copy(ocular_raw, p, overwrite=True))

        flags = ["badch_detection_incomplete"] if bads.errors else []
        sel_ok = ica_res is None or (sel is not None and sel.get("status") == "ok")
        if not sel_ok:
            flags.append("ica_selection_failed")
        complete = (ocular_qc["status"] != "error" and ica_qc["status"] in ("ok", "disabled")
                    and sel_ok and not any(f in INCOMPLETE_FLAGS for f in flags))
        qc = {
            "recording": {"subject": rec.subject, "task": rec.task, "run": rec.run, "stem": rec.stem},
            "complete": bool(complete),
            "status": "done" if complete else "incomplete",
            "qc_flags": flags,
            "config_version": cfg.version,
            "sample_axis": {"n_times": seg.n_times, "first_samp": seg.first_samp, "sfreq": sf,
                            "bdf_n_records": loaded.header.n_records,
                            "bdf_record_samples": int(round(loaded.header.record_duration_s * sf))},
            "loading": loaded.qc(),
            "segments": seg.summary(),
            "filters": {"analysis": cfg.filter_analysis.mne_kwargs(), "highpass_copy": cfg.filter_highpass_copy.mne_kwargs(),
                        "line_noise_strategy": cfg.line_noise.analysis_strategy},
            "bad_channels": bads.qc(),
            "roi_bads": [c for c in cfg.artefacts.roi if c in bads.bads],
            "reference": {"method": cfg.reference_method, "n_reference_channels": len(good)},
            "interpolation_origin": cfg.interpolation_origin,
            "ica": ica_qc,
            "ocular": ocular_qc,
            "exg_qc": exg_qc_res,
            "exg_bads": exg_bads,
            "non_eeg_bads": non_eeg_bads,
            "outputs": sorted([*written, "qc"]),
        }
        written["qc"] = self._write(paths["qc"], lambda p: io_mod.save_json(qc, p))
        status = "done" if complete else "incomplete"
        return RecordingResult(subject=rec.subject, task=rec.task, run=rec.run, stem=rec.stem, status=status,
                               outputs=written, qc=qc,
                               message="" if complete else ";".join(flags) or "see QC record")
