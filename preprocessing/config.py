"""Load and validate ``configs/preprocessing.yaml``.

The schema is strict for the sections this package owns (unknown keys and wrong
types raise :class:`~utils.config.ConfigError`). Two sections are
passed through as raw mappings: ``ocular`` (a pointer to ``configs/ocular.yaml``,
checked against it by :func:`utils.config_ocular.check_preprocessing_stub`;
it defines no output) and ``ica_selection`` (validated by
:func:`preprocessing.ica_selection.parse_selection_config`).

The returned :class:`PreprocessingConfig` is an immutable dataclass. It also
supports ``cfg["section"]`` / ``cfg.get("section")``, returning the raw parsed
mapping of a section (the ocular module may use either ``cfg.ocular`` or
``cfg["ocular"]``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.config import ConfigError, find_project_root, read_yaml

PREPROCESSING_REL = Path("configs") / "preprocessing.yaml"
#: Top-level sections the loader validates (``ocular`` and ``ica_selection`` are passed through).
CORE_SECTIONS = ("version", "loading", "segments", "filters", "line_noise", "bad_channels", "reference",
                 "interpolation", "ica", "artefacts", "storage", "runtime")
PASS_THROUGH_SECTIONS = ("ocular", "ica_selection")
BAD_CHANNEL_CRITERIA = ("nan_flat", "deviation", "hf_noise", "correlation", "snr", "psd", "ransac")
ICLABEL_CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other")


# --------------------------------------------------------------------------- helpers
def _map(d: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where}: expected a mapping")
    return d


def _keys(d: Mapping[str, Any], allowed: set[str], where: str, required: set[str] | None = None) -> None:
    bad = set(d) - allowed
    if bad:
        raise ConfigError(f"{where}: unknown keys {sorted(map(str, bad))}; allowed: {sorted(allowed)}")
    miss = (required if required is not None else allowed) - set(d)
    if miss:
        raise ConfigError(f"{where}: missing keys {sorted(miss)}")


def _num(v: Any, where: str, *, allow_none: bool = False, positive: bool = False) -> float | None:
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {v!r}")
    if positive and v <= 0:
        raise ConfigError(f"{where}: must be > 0, got {v!r}")
    return float(v)


def _int(v: Any, where: str, *, minimum: int | None = None, allow_none: bool = False) -> int | None:
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{where}: expected an integer, got {v!r}")
    if minimum is not None and v < minimum:
        raise ConfigError(f"{where}: must be >= {minimum}, got {v}")
    return int(v)


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        raise ConfigError(f"{where}: expected true or false, got {v!r}")
    return v


def _strs(v: Any, where: str) -> tuple[str, ...]:
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise ConfigError(f"{where}: expected a list of non-empty strings")
    return tuple(v)


def _str(v: Any, where: str, choices: tuple[str, ...] | None = None) -> str:
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{where}: expected a non-empty string, got {v!r}")
    if choices is not None and v not in choices:
        raise ConfigError(f"{where}: must be one of {list(choices)}, got {v!r}")
    return v


def _auto_or_num(v: Any, where: str) -> float | str:
    if v == "auto":
        return "auto"
    return _num(v, where, positive=True)  # type: ignore[return-value]


# --------------------------------------------------------------------------- sections
@dataclass(frozen=True)
class FilterSpec:
    """Zero-phase FIR band-pass parameters (passed to MNE ``filter``)."""

    l_freq: float | None
    h_freq: float | None
    l_trans_bandwidth: float | str = "auto"
    h_trans_bandwidth: float | str = "auto"
    filter_length: str | int = "auto"
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    phase: str = "zero"
    pad: str = "reflect_limited"

    def mne_kwargs(self) -> dict:
        """Keyword arguments of :meth:`mne.io.Raw.filter` / :func:`mne.filter.filter_data`."""
        return {"l_freq": self.l_freq, "h_freq": self.h_freq, "l_trans_bandwidth": self.l_trans_bandwidth,
                "h_trans_bandwidth": self.h_trans_bandwidth, "filter_length": self.filter_length,
                "fir_window": self.fir_window, "fir_design": self.fir_design, "phase": self.phase,
                "method": "fir", "pad": self.pad}


def _filter(d: Any, where: str) -> FilterSpec:
    d = _map(d, where)
    _keys(d, {"l_freq", "h_freq", "l_trans_bandwidth", "h_trans_bandwidth", "filter_length", "fir_window",
              "fir_design", "phase", "pad"}, where, required={"l_freq", "h_freq"})
    lf = _num(d["l_freq"], f"{where}.l_freq", allow_none=True)
    hf = _num(d["h_freq"], f"{where}.h_freq", allow_none=True)
    if lf is not None and hf is not None and not 0 < lf < hf:
        raise ConfigError(f"{where}: need 0 < l_freq < h_freq")
    fl = d.get("filter_length", "auto")
    if fl != "auto":
        fl = _int(fl, f"{where}.filter_length", minimum=3)
    phase = _str(d.get("phase", "zero"), f"{where}.phase", ("zero", "zero-double"))
    return FilterSpec(l_freq=lf, h_freq=hf, l_trans_bandwidth=_auto_or_num(d.get("l_trans_bandwidth", "auto"),
                                                                           f"{where}.l_trans_bandwidth"),
                      h_trans_bandwidth=_auto_or_num(d.get("h_trans_bandwidth", "auto"), f"{where}.h_trans_bandwidth"),
                      filter_length=fl, fir_window=_str(d.get("fir_window", "hamming"), f"{where}.fir_window"),
                      fir_design=_str(d.get("fir_design", "firwin"), f"{where}.fir_design", ("firwin", "firwin2")),
                      phase=phase, pad=_str(d.get("pad", "reflect_limited"), f"{where}.pad"))


@dataclass(frozen=True)
class LoadingConfig:
    """``loading`` section: channel typing / dropping, montage, Status channel, unconnected check."""

    eog_channels: tuple[str, ...]
    misc_channels: tuple[str, ...]
    drop_channels: tuple[str, ...]
    montage: str
    status_channel: str
    unconnected_rail_v: float
    unconnected_max_sd_uv: float


@dataclass(frozen=True)
class SpanRule:
    """Rule turning a per-sample mask into padded, merged spans."""

    pad_s: float
    merge_gap_s: float
    min_duration_s: float


@dataclass(frozen=True)
class SegmentsConfig:
    """``segments`` section: join annotation, rail / flat span rules, minimum valid-span length."""

    join_annotation: str
    rail_threshold_v: float
    rail: SpanRule
    flat_max_abs_diff_uv: float
    flat: SpanRule
    zero_skip_spans: bool
    min_valid_span_s: float | str = 0.0     # "auto" = the longest per-span FIR kernel; 0 = no minimum


@dataclass(frozen=True)
class LineNoiseConfig:
    analysis_strategy: str
    notch_freqs: tuple[float, ...]
    ica_notch: bool
    report_freq: float


@dataclass(frozen=True)
class RansacConfig:
    n_samples: int
    sample_prop: float
    corr_thresh: float
    frac_bad: float
    corr_window_secs: float
    channel_wise: bool


@dataclass(frozen=True)
class BadChannelsConfig:
    method: str
    criteria: tuple[str, ...]
    deviation_threshold: float
    hf_noise_threshold: float
    correlation_secs: float
    correlation_threshold: float
    correlation_frac_bad: float
    ransac: RansacConfig
    saturation_threshold_v: float
    saturation_max_frac: float | None
    notch: bool
    decimate: int
    initial_reference: str
    max_iterations: int
    max_bad_fraction_warn: float
    manual_bads_from_known_issues: bool
    ransac_pass: str = "every"     # 'every' robust-reference pass, or only the 'final' one
    correlation_min_bad_windows: int = 1   # correlation / dropout also need at least this many bad windows


@dataclass(frozen=True)
class ICAConfig:
    enabled: bool
    method: str
    fit_params: dict
    max_iter: int
    n_components: str | int
    max_components: int | None
    min_samples_factor: float
    decimate: int
    fit_reject_ptp_uv: float | None
    fit_reject_chunk_s: float
    iclabel_rule: str
    iclabel_exclude: dict[str, float]
    apply_in_primary: bool


@dataclass(frozen=True)
class ArtefactConfig:
    roi: tuple[str, ...]
    roi_ptp_uv: float
    qc_chunk_s: float
    frontal_proxy: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Validated ``configs/preprocessing.yaml``."""

    version: str
    loading: LoadingConfig
    segments: SegmentsConfig
    filter_analysis: FilterSpec
    filter_highpass_copy: FilterSpec
    line_noise: LineNoiseConfig
    bad_channels: BadChannelsConfig
    reference_method: str
    interpolation_method: str
    interpolation_origin: str
    ica: ICAConfig
    artefacts: ArtefactConfig
    ocular: dict
    storage_subdir: str
    storage_fmt: str
    n_jobs: int
    random_seed: int
    source: Path
    raw: dict = field(repr=False, compare=False)
    threads_per_worker: int = 1
    ica_selection: dict = field(default_factory=dict)

    def __getitem__(self, section: str) -> Any:
        """Raw parsed mapping of one top-level section (e.g. ``cfg["ocular"]``)."""
        return self.raw[section]

    def get(self, section: str, default: Any = None) -> Any:
        return self.raw.get(section, default)


def _span_rule(d: Mapping[str, Any], where: str, extra: set[str]) -> SpanRule:
    _keys(d, {"pad_s", "merge_gap_s", "min_duration_s"} | extra, where, required=set(extra))
    vals = {k: _num(d.get(k, 0.0), f"{where}.{k}") for k in ("pad_s", "merge_gap_s", "min_duration_s")}
    if any(v < 0 for v in vals.values()):  # type: ignore[operator]
        raise ConfigError(f"{where}: pad_s, merge_gap_s and min_duration_s must be >= 0")
    return SpanRule(**vals)  # type: ignore[arg-type]


def parse_preprocessing_config(raw: Mapping[str, Any], source: str | os.PathLike[str] = "<memory>") -> PreprocessingConfig:
    """Validate a parsed ``preprocessing.yaml`` mapping."""
    w = str(source)
    raw = _map(raw, w)
    _keys(raw, set(CORE_SECTIONS) | set(PASS_THROUGH_SECTIONS), w, required=set(CORE_SECTIONS))

    lo = _map(raw["loading"], f"{w}:loading")
    _keys(lo, {"eog_channels", "misc_channels", "drop_channels", "montage", "status_channel", "unconnected_check"},
          f"{w}:loading")
    uc = _map(lo["unconnected_check"], f"{w}:loading.unconnected_check")
    _keys(uc, {"rail_v", "max_sd_uv"}, f"{w}:loading.unconnected_check")
    loading = LoadingConfig(
        eog_channels=_strs(lo["eog_channels"], f"{w}:loading.eog_channels"),
        misc_channels=_strs(lo["misc_channels"], f"{w}:loading.misc_channels"),
        drop_channels=_strs(lo["drop_channels"], f"{w}:loading.drop_channels"),
        montage=_str(lo["montage"], f"{w}:loading.montage"),
        status_channel=_str(lo["status_channel"], f"{w}:loading.status_channel"),
        unconnected_rail_v=_num(uc["rail_v"], f"{w}:loading.unconnected_check.rail_v"),  # type: ignore[arg-type]
        unconnected_max_sd_uv=_num(uc["max_sd_uv"], f"{w}:loading.unconnected_check.max_sd_uv", positive=True),  # type: ignore[arg-type]
    )
    groups = [set(loading.eog_channels), set(loading.misc_channels), set(loading.drop_channels)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ConfigError(f"{w}:loading: eog, misc and drop channel lists must be disjoint")

    sg = _map(raw["segments"], f"{w}:segments")
    _keys(sg, {"join_annotation", "rail", "flat", "zero_skip_spans", "min_valid_span_s"}, f"{w}:segments",
          required={"join_annotation", "rail", "flat", "zero_skip_spans"})
    mvs = sg.get("min_valid_span_s", 0.0)
    if mvs != "auto":
        mvs = _num(mvs, f"{w}:segments.min_valid_span_s")
        if mvs < 0:  # type: ignore[operator]
            raise ConfigError(f"{w}:segments.min_valid_span_s must be 'auto' or >= 0")
    rail = _map(sg["rail"], f"{w}:segments.rail")
    flat = _map(sg["flat"], f"{w}:segments.flat")
    ja = _str(sg["join_annotation"], f"{w}:segments.join_annotation")
    if not ja.lower().startswith("edge"):
        raise ConfigError(f"{w}:segments.join_annotation must start with 'EDGE' (MNE splits filters at 'edge' "
                          f"annotations)")
    segments = SegmentsConfig(
        join_annotation=ja, rail_threshold_v=_num(rail.get("threshold_v"), f"{w}:segments.rail.threshold_v"),  # type: ignore[arg-type]
        rail=_span_rule(rail, f"{w}:segments.rail", {"threshold_v"}),
        flat_max_abs_diff_uv=_num(flat.get("max_abs_diff_uv"), f"{w}:segments.flat.max_abs_diff_uv", positive=True),  # type: ignore[arg-type]
        flat=_span_rule(flat, f"{w}:segments.flat", {"max_abs_diff_uv"}),
        zero_skip_spans=_bool(sg["zero_skip_spans"], f"{w}:segments.zero_skip_spans"),
        min_valid_span_s=mvs,  # type: ignore[arg-type]
    )

    fl = _map(raw["filters"], f"{w}:filters")
    _keys(fl, {"analysis", "highpass_copy"}, f"{w}:filters")
    fa = _filter(fl["analysis"], f"{w}:filters.analysis")
    fh = _filter(fl["highpass_copy"], f"{w}:filters.highpass_copy")
    if fa.l_freq is None or fh.l_freq is None:
        raise ConfigError(f"{w}:filters: both copies must be high-passed (the BDF is DC-coupled)")

    ln = _map(raw["line_noise"], f"{w}:line_noise")
    _keys(ln, {"analysis_strategy", "notch_freqs", "ica_notch", "report_freq"}, f"{w}:line_noise")
    nf = ln["notch_freqs"]
    if not isinstance(nf, list):
        raise ConfigError(f"{w}:line_noise.notch_freqs must be a list")
    line = LineNoiseConfig(
        analysis_strategy=_str(ln["analysis_strategy"], f"{w}:line_noise.analysis_strategy", ("lowpass_40", "notch")),
        notch_freqs=tuple(_num(x, f"{w}:line_noise.notch_freqs", positive=True) for x in nf),  # type: ignore[misc]
        ica_notch=_bool(ln["ica_notch"], f"{w}:line_noise.ica_notch"),
        report_freq=_num(ln["report_freq"], f"{w}:line_noise.report_freq", positive=True),  # type: ignore[arg-type]
    )

    bc = _map(raw["bad_channels"], f"{w}:bad_channels")
    bw = f"{w}:bad_channels"
    _keys(bc, {"method", "criteria", "deviation_threshold", "hf_noise_threshold", "correlation_secs",
               "correlation_threshold", "correlation_frac_bad", "ransac", "saturation", "notch", "decimate",
               "robust_reference", "max_bad_fraction_warn", "manual_bads_from_known_issues", "ransac_pass",
               "correlation_min_bad_windows"}, bw)
    crit = _strs(bc["criteria"], f"{bw}.criteria")
    bad = set(crit) - set(BAD_CHANNEL_CRITERIA)
    if bad:
        raise ConfigError(f"{bw}.criteria: unknown {sorted(bad)}; allowed {list(BAD_CHANNEL_CRITERIA)}")
    if "snr" in crit and not {"hf_noise", "correlation"} <= set(crit):
        raise ConfigError(f"{bw}.criteria: 'snr' combines hf_noise and correlation; list both")
    rs = _map(bc["ransac"], f"{bw}.ransac")
    _keys(rs, {"n_samples", "sample_prop", "corr_thresh", "frac_bad", "corr_window_secs", "channel_wise"},
          f"{bw}.ransac")
    sat = _map(bc["saturation"], f"{bw}.saturation")
    _keys(sat, {"threshold_v", "max_frac"}, f"{bw}.saturation")
    rr = _map(bc["robust_reference"], f"{bw}.robust_reference")
    _keys(rr, {"initial", "max_iterations"}, f"{bw}.robust_reference")
    badch = BadChannelsConfig(
        method=_str(bc["method"], f"{bw}.method", ("pyprep_noisychannels",)), criteria=crit,
        deviation_threshold=_num(bc["deviation_threshold"], f"{bw}.deviation_threshold", positive=True),  # type: ignore[arg-type]
        hf_noise_threshold=_num(bc["hf_noise_threshold"], f"{bw}.hf_noise_threshold", positive=True),  # type: ignore[arg-type]
        correlation_secs=_num(bc["correlation_secs"], f"{bw}.correlation_secs", positive=True),  # type: ignore[arg-type]
        correlation_threshold=_num(bc["correlation_threshold"], f"{bw}.correlation_threshold"),  # type: ignore[arg-type]
        correlation_frac_bad=_num(bc["correlation_frac_bad"], f"{bw}.correlation_frac_bad"),  # type: ignore[arg-type]
        ransac=RansacConfig(
            n_samples=_int(rs["n_samples"], f"{bw}.ransac.n_samples", minimum=1),  # type: ignore[arg-type]
            sample_prop=_num(rs["sample_prop"], f"{bw}.ransac.sample_prop", positive=True),  # type: ignore[arg-type]
            corr_thresh=_num(rs["corr_thresh"], f"{bw}.ransac.corr_thresh"),  # type: ignore[arg-type]
            frac_bad=_num(rs["frac_bad"], f"{bw}.ransac.frac_bad"),  # type: ignore[arg-type]
            corr_window_secs=_num(rs["corr_window_secs"], f"{bw}.ransac.corr_window_secs", positive=True),  # type: ignore[arg-type]
            channel_wise=_bool(rs["channel_wise"], f"{bw}.ransac.channel_wise")),
        saturation_threshold_v=_num(sat["threshold_v"], f"{bw}.saturation.threshold_v", positive=True),  # type: ignore[arg-type]
        saturation_max_frac=_num(sat["max_frac"], f"{bw}.saturation.max_frac", allow_none=True),
        notch=_bool(bc["notch"], f"{bw}.notch"), decimate=_int(bc["decimate"], f"{bw}.decimate", minimum=1),  # type: ignore[arg-type]
        initial_reference=_str(rr["initial"], f"{bw}.robust_reference.initial", ("median", "average")),
        max_iterations=_int(rr["max_iterations"], f"{bw}.robust_reference.max_iterations", minimum=1),  # type: ignore[arg-type]
        max_bad_fraction_warn=_num(bc["max_bad_fraction_warn"], f"{bw}.max_bad_fraction_warn", positive=True),  # type: ignore[arg-type]
        manual_bads_from_known_issues=_bool(bc["manual_bads_from_known_issues"], f"{bw}.manual_bads_from_known_issues"),
        ransac_pass=_str(bc.get("ransac_pass", "every"), f"{bw}.ransac_pass", ("every", "final")),
        correlation_min_bad_windows=_int(bc.get("correlation_min_bad_windows", 1),  # type: ignore[arg-type]
                                         f"{bw}.correlation_min_bad_windows", minimum=1),
    )

    ref = _map(raw["reference"], f"{w}:reference")
    _keys(ref, {"method"}, f"{w}:reference")
    interp = _map(raw["interpolation"], f"{w}:interpolation")
    _keys(interp, {"method", "origin"}, f"{w}:interpolation")

    ic = _map(raw["ica"], f"{w}:ica")
    iw = f"{w}:ica"
    _keys(ic, {"enabled", "method", "fit_params", "max_iter", "n_components", "max_components", "min_samples_factor",
               "decimate", "fit_reject_ptp_uv", "fit_reject_chunk_s", "iclabel", "apply_in_primary"}, iw)
    lab = _map(ic["iclabel"], f"{iw}.iclabel")
    _keys(lab, {"rule", "exclude"}, f"{iw}.iclabel")
    excl = _map(lab["exclude"], f"{iw}.iclabel.exclude")
    unknown = set(excl) - set(ICLABEL_CLASSES)
    if unknown:
        raise ConfigError(f"{iw}.iclabel.exclude: unknown classes {sorted(unknown)}; allowed {list(ICLABEL_CLASSES)}")
    thr = {str(k): _num(v, f"{iw}.iclabel.exclude.{k}") for k, v in excl.items()}
    if any(not 0 < v <= 1 for v in thr.values()):  # type: ignore[operator]
        raise ConfigError(f"{iw}.iclabel.exclude: probabilities must be in (0, 1]")
    ncomp = ic["n_components"]
    if ncomp != "auto":
        ncomp = _int(ncomp, f"{iw}.n_components", minimum=2)
    fp = _map(ic["fit_params"], f"{iw}.fit_params")
    ica = ICAConfig(
        enabled=_bool(ic["enabled"], f"{iw}.enabled"), method=_str(ic["method"], f"{iw}.method", ("picard", "infomax")),
        fit_params=dict(fp), max_iter=_int(ic["max_iter"], f"{iw}.max_iter", minimum=1),  # type: ignore[arg-type]
        n_components=ncomp, max_components=_int(ic["max_components"], f"{iw}.max_components", minimum=2, allow_none=True),
        min_samples_factor=_num(ic["min_samples_factor"], f"{iw}.min_samples_factor", positive=True),  # type: ignore[arg-type]
        decimate=_int(ic["decimate"], f"{iw}.decimate", minimum=1),  # type: ignore[arg-type]
        fit_reject_ptp_uv=_num(ic["fit_reject_ptp_uv"], f"{iw}.fit_reject_ptp_uv", allow_none=True),
        fit_reject_chunk_s=_num(ic["fit_reject_chunk_s"], f"{iw}.fit_reject_chunk_s", positive=True),  # type: ignore[arg-type]
        iclabel_rule=_str(lab["rule"], f"{iw}.iclabel.rule", ("threshold", "argmax_and_threshold")),
        iclabel_exclude=thr,  # type: ignore[arg-type]
        apply_in_primary=_bool(ic["apply_in_primary"], f"{iw}.apply_in_primary"),
    )

    ar = _map(raw["artefacts"], f"{w}:artefacts")
    _keys(ar, {"roi", "roi_ptp_uv", "qc_chunk_s", "frontal_proxy"}, f"{w}:artefacts")
    art = ArtefactConfig(roi=_strs(ar["roi"], f"{w}:artefacts.roi"),
                         roi_ptp_uv=_num(ar["roi_ptp_uv"], f"{w}:artefacts.roi_ptp_uv", positive=True),  # type: ignore[arg-type]
                         qc_chunk_s=_num(ar["qc_chunk_s"], f"{w}:artefacts.qc_chunk_s", positive=True),  # type: ignore[arg-type]
                         frontal_proxy=_strs(ar["frontal_proxy"], f"{w}:artefacts.frontal_proxy"))

    st = _map(raw["storage"], f"{w}:storage")
    _keys(st, {"subdir", "fmt"}, f"{w}:storage")
    rt = _map(raw["runtime"], f"{w}:runtime")
    _keys(rt, {"n_jobs", "random_seed", "threads_per_worker"}, f"{w}:runtime", required={"n_jobs", "random_seed"})
    oc = raw.get("ocular") or {}
    if not isinstance(oc, Mapping):
        raise ConfigError(f"{w}:ocular must be a mapping")
    sel = raw.get("ica_selection") or {}
    if not isinstance(sel, Mapping):
        raise ConfigError(f"{w}:ica_selection must be a mapping")
    return PreprocessingConfig(
        version=_str(str(raw["version"]), f"{w}:version"), loading=loading, segments=segments, filter_analysis=fa,
        filter_highpass_copy=fh, line_noise=line, bad_channels=badch,
        reference_method=_str(ref["method"], f"{w}:reference.method", ("robust_average",)),
        interpolation_method=_str(interp["method"], f"{w}:interpolation.method", ("spline",)),
        interpolation_origin=_str(str(interp["origin"]), f"{w}:interpolation.origin"),
        ica=ica, artefacts=art, ocular=dict(oc),
        storage_subdir=_str(st["subdir"], f"{w}:storage.subdir"),
        storage_fmt=_str(st["fmt"], f"{w}:storage.fmt", ("single", "double")),
        n_jobs=_int(rt["n_jobs"], f"{w}:runtime.n_jobs", minimum=1),  # type: ignore[arg-type]
        random_seed=_int(rt["random_seed"], f"{w}:runtime.random_seed", minimum=0),  # type: ignore[arg-type]
        source=Path(str(source)), raw=dict(raw),
        threads_per_worker=_int(rt.get("threads_per_worker", 1), f"{w}:runtime.threads_per_worker",  # type: ignore[arg-type]
                                minimum=1),
        ica_selection=dict(sel),
    )


def load_preprocessing_config(path: str | os.PathLike[str] | None = None,
                              root: str | os.PathLike[str] | None = None) -> PreprocessingConfig:
    """Load ``<root>/configs/preprocessing.yaml`` (root discovered if not given)."""
    if path is None:
        root_p = Path(root) if root is not None else find_project_root()
        path = root_p / PREPROCESSING_REL
    p = Path(path)
    return parse_preprocessing_config(read_yaml(p), source=p.resolve())
