"""Load and validate ``configs/features.yaml`` (trial-level alpha features).

The loader returns an immutable :class:`FeaturesConfig`.

Cross-checks (:func:`check_consistency`): the windows must equal ``configs/events.yaml`` ``prestim_windows_ms`` and the
reported windows, the edge margin must equal the events config's ``prestim_edge_margin_s``, and the artefact
threshold must equal ``configs/preprocessing.yaml`` ``artefacts.roi_ptp_uv`` (the preprocessing config is the source,
this file names the rule).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.config import (
    ConfigError,
    _bool,
    _int,
    _known_keys,
    _num,
    _req,
    _str,
    _str_list,
    find_project_root,
    read_yaml,
)

#: Prestimulus windows ([start, stop) ms; primary and window-length variants); the config must reproduce them.
SPEC_WINDOWS_MS: dict[str, dict[str, tuple[float, float]]] = {
    "AVSRT": {"primary": (-850.0, -50.0), "s_w500": (-550.0, -50.0), "s_w1000": (-1050.0, -50.0)},
    "FAST": {"primary": (-550.0, -50.0), "s_w400": (-450.0, -50.0)},
    "IC": {"primary": (-550.0, -50.0), "s_w400": (-450.0, -50.0)},
}
#: Primary posterior ROI.
SPEC_PRIMARY_ROI: tuple[str, ...] = ("PO7", "PO8", "PO3", "PO4", "POz", "O1", "O2", "Oz")


def _pair(v: Any, where: str) -> tuple[float, float]:
    if not isinstance(v, list) or len(v) != 2:
        raise ConfigError(f"{where}: expected [lo, hi]")
    lo, hi = (_num(x, where) for x in v)
    if not hi > lo:
        raise ConfigError(f"{where}: hi must exceed lo")
    return lo, hi


@dataclass(frozen=True)
class ROIConfig:
    name: str
    channels: tuple[str, ...]
    min_channels: int


@dataclass(frozen=True)
class DpssConfig:
    bandwidth_hz: float
    low_bias: bool
    adaptive: bool
    normalization: str
    remove_dc: bool
    raise_to_mne_minimum: bool


@dataclass(frozen=True)
class WelchConfig:
    window: str
    nperseg: int
    overlap_frac: float
    detrend: str


@dataclass(frozen=True)
class AlphaConfig:
    data_scale: float
    windows_ms: dict[str, dict[str, tuple[float, float]]]
    edge_margin_s: float
    use_edge_flags: bool
    check_segments: bool
    bad_annotation_prefix: str
    artefact_rule: str
    artefact_ptp_uv: float
    artefact_keep_equal: bool
    rois: dict[str, ROIConfig]
    nfft: int
    dpss: DpssConfig
    mt4_bandwidth_hz: float
    welch: WelchConfig
    iaf_half_width_hz: float
    band_tol_hz: float
    fixed_bands: dict[str, tuple[float, float]]
    split_time_tasks: tuple[str, ...]
    erp_tasks: tuple[str, ...]
    erp_epoch_s: float
    erp_min_epochs: int
    erp_leave_out_until: str
    pc1_tasks: tuple[str, ...]
    pc1_window_ms: tuple[float, float]
    context_lengths_s: tuple[float, ...]
    context_min_valid_frac: float
    context_filter: dict[str, str]
    spectra_fmax_hz: float

    def roi(self, name: str) -> ROIConfig:
        if name not in self.rois:
            raise KeyError(f"no ROI {name!r} in configs/features.yaml (have {sorted(self.rois)})")
        return self.rois[name]

    def all_channels(self) -> list[str]:
        """Union of every ROI's channels, first-seen order."""
        out: list[str] = []
        for r in self.rois.values():
            out.extend(c for c in r.channels if c not in out)
        return out


MERGE_MODES = ("preceding", "greedy")
#: Averaging scopes of the harmonised-history variant; the same tuple as ``feature_extraction.decomposition.HARM_SCOPES``.
HARM_SCOPES = ("run", "segment")


@dataclass(frozen=True)
class DecompositionConfig:
    min_block_windows: int
    #: task -> block-merge mode (``preceding`` or ``greedy``; FAST uses ``greedy``)
    merge_modes: dict[str, str]
    sd_ddof: int
    blocks: dict[str, tuple[str, ...]]
    harm_half_window_s: float
    #: averaging scope of the harmonised-history variant: ``run`` (across joins) or ``segment`` (within a valid span)
    harm_scope: str
    hist_n_prev: int
    hist_min_prev: int
    context_lengths_s: tuple[float, ...]
    reliability_min_windows: int

    def merge_mode(self, task: str) -> str:
        """Block-merge mode of ``task``."""
        if task not in self.merge_modes:
            raise KeyError(f"no decomposition.merge_mode for task {task!r}")
        return self.merge_modes[task]


@dataclass(frozen=True)
class FeaturesConfig:
    """Validated ``configs/features.yaml``."""

    version: str
    alpha: AlphaConfig
    decomposition: DecompositionConfig
    source: Path | None
    raw: dict = field(repr=False, compare=False)


ALLOWED_LEAVE_OUT = ("max_rt_window_end_next_stimulus",)


def parse_features_config(raw: Mapping[str, Any], where: str = "features config",
                          source: Path | None = None) -> FeaturesConfig:
    """Validate a parsed ``features.yaml`` mapping (raises :class:`ConfigError`)."""
    w = where
    _known_keys(raw, {"version", "alpha", "decomposition"}, w)
    version = _str(_req(raw, "version", w), f"{w}.version")
    a = _req(raw, "alpha", w)
    aw = f"{w}:alpha"
    _known_keys(a, {"units", "windows", "artefact", "rois", "spectrum", "bands", "split_time", "s_erp", "pc1",
                    "context", "spectra_fmax_hz"}, aw)
    u = _req(a, "units", aw)
    _known_keys(u, {"data_scale"}, f"{aw}.units")
    scale = _num(_req(u, "data_scale", f"{aw}.units"), f"{aw}.units.data_scale")
    if scale <= 0:
        raise ConfigError(f"{aw}.units.data_scale must be > 0")

    wi = _req(a, "windows", aw)
    ww = f"{aw}.windows"
    _known_keys(wi, {"expected_ms", "edge_margin_s", "use_edge_flags", "check_segments", "bad_annotation_prefix"}, ww)
    exp = _req(wi, "expected_ms", ww)
    if not isinstance(exp, Mapping) or not exp:
        raise ConfigError(f"{ww}.expected_ms: expected a mapping task -> {{window: [start, stop]}}")
    windows: dict[str, dict[str, tuple[float, float]]] = {}
    for task, wd in exp.items():
        if not isinstance(wd, Mapping) or "primary" not in wd:
            raise ConfigError(f"{ww}.expected_ms.{task}: needs a 'primary' window")
        windows[str(task)] = {}
        for wn, v in wd.items():
            lo, hi = _pair(v, f"{ww}.expected_ms.{task}.{wn}")
            if hi > 0:
                raise ConfigError(f"{ww}.expected_ms.{task}.{wn}: a prestimulus window must end before onset")
            windows[str(task)][str(wn)] = (lo, hi)
    margin = _num(_req(wi, "edge_margin_s", ww), f"{ww}.edge_margin_s")
    if margin < 0:
        raise ConfigError(f"{ww}.edge_margin_s must be >= 0")

    ar = _req(a, "artefact", aw)
    arw = f"{aw}.artefact"
    _known_keys(ar, {"rule", "ptp_uv", "keep_equal"}, arw)
    ptp = _num(_req(ar, "ptp_uv", arw), f"{arw}.ptp_uv")
    rule = _str(_req(ar, "rule", arw), f"{arw}.rule")
    if rule != f"roi{int(ptp) if float(ptp).is_integer() else ptp}":
        raise ConfigError(f"{arw}.rule {rule!r} does not name the threshold {ptp} uV (expected 'roi{ptp:g}')")

    ro = _req(a, "rois", aw)
    if not isinstance(ro, Mapping) or "primary" not in ro:
        raise ConfigError(f"{aw}.rois: needs a 'primary' ROI")
    rois: dict[str, ROIConfig] = {}
    for name, rd in ro.items():
        rw = f"{aw}.rois.{name}"
        _known_keys(rd, {"channels", "min_channels"}, rw)
        ch = _str_list(_req(rd, "channels", rw), f"{rw}.channels")
        mn = _int(_req(rd, "min_channels", rw), f"{rw}.min_channels")
        if len(set(ch)) != len(ch) or not 1 <= mn <= len(ch):
            raise ConfigError(f"{rw}: duplicate channels or min_channels outside 1..{len(ch)}")
        rois[str(name)] = ROIConfig(str(name), ch, mn)
    if tuple(rois["primary"].channels) != SPEC_PRIMARY_ROI or rois["primary"].min_channels != 5:
        raise ConfigError(f"{aw}.rois.primary must be the ROI {list(SPEC_PRIMARY_ROI)} with 5 channels")
    for need in ("ptheta", "mu", "mu_left", "mu_right", "half_left", "half_right"):
        if need not in rois:
            raise ConfigError(f"{aw}.rois: missing ROI {need!r}")
    halves = set(rois["half_left"].channels) | set(rois["half_right"].channels)
    if halves != set(rois["primary"].channels) or set(rois["half_left"].channels) & set(rois["half_right"].channels):
        raise ConfigError(f"{aw}.rois: half_left and half_right must partition the primary ROI")

    s = _req(a, "spectrum", aw)
    sw = f"{aw}.spectrum"
    _known_keys(s, {"nfft", "dpss", "mt4", "welch"}, sw)
    nfft = _int(_req(s, "nfft", sw), f"{sw}.nfft")
    d = _req(s, "dpss", sw)
    _known_keys(d, {"bandwidth_hz", "low_bias", "adaptive", "normalization", "remove_dc", "raise_to_mne_minimum"},
                f"{sw}.dpss")
    dpss = DpssConfig(_num(_req(d, "bandwidth_hz", sw), f"{sw}.dpss.bandwidth_hz"),
                      _bool(_req(d, "low_bias", sw), f"{sw}.dpss.low_bias"),
                      _bool(_req(d, "adaptive", sw), f"{sw}.dpss.adaptive"),
                      _str(_req(d, "normalization", sw), f"{sw}.dpss.normalization"),
                      _bool(_req(d, "remove_dc", sw), f"{sw}.dpss.remove_dc"),
                      _bool(_req(d, "raise_to_mne_minimum", sw), f"{sw}.dpss.raise_to_mne_minimum"))
    if dpss.normalization != "full" or dpss.adaptive or not dpss.low_bias:
        raise ConfigError(f"{sw}.dpss: normalization 'full', adaptive false and low_bias true are required")
    m4 = _req(s, "mt4", sw)
    _known_keys(m4, {"bandwidth_hz"}, f"{sw}.mt4")
    mt4 = _num(_req(m4, "bandwidth_hz", f"{sw}.mt4"), f"{sw}.mt4.bandwidth_hz")
    we = _req(s, "welch", sw)
    _known_keys(we, {"window", "nperseg", "overlap_frac", "detrend"}, f"{sw}.welch")
    welch = WelchConfig(_str(_req(we, "window", sw), f"{sw}.welch.window"),
                        _int(_req(we, "nperseg", sw), f"{sw}.welch.nperseg"),
                        _num(_req(we, "overlap_frac", sw), f"{sw}.welch.overlap_frac"),
                        _str(_req(we, "detrend", sw), f"{sw}.welch.detrend"))
    if not 0 <= welch.overlap_frac < 1:
        raise ConfigError(f"{sw}.welch.overlap_frac must be in [0, 1)")

    b = _req(a, "bands", aw)
    bw = f"{aw}.bands"
    _known_keys(b, {"iaf_half_width_hz", "inclusive_tol_hz", "fixed"}, bw)
    fixed = {str(k): _pair(v, f"{bw}.fixed.{k}") for k, v in _req(b, "fixed", bw).items()}
    for need in ("s_band", "e9", "ap_2p5_5", "ap_15_25"):
        if need not in fixed:
            raise ConfigError(f"{bw}.fixed: missing band {need!r}")

    st = _req(a, "split_time", aw)
    _known_keys(st, {"tasks"}, f"{aw}.split_time")
    e = _req(a, "s_erp", aw)
    ew = f"{aw}.s_erp"
    _known_keys(e, {"tasks", "epoch_s", "min_epochs", "leave_out_until"}, ew)
    lo_until = _str(_req(e, "leave_out_until", ew), f"{ew}.leave_out_until")
    if lo_until not in ALLOWED_LEAVE_OUT:
        raise ConfigError(f"{ew}.leave_out_until must be one of {ALLOWED_LEAVE_OUT}")
    p = _req(a, "pc1", aw)
    _known_keys(p, {"tasks", "post_window_ms"}, f"{aw}.pc1")
    c = _req(a, "context", aw)
    cw = f"{aw}.context"
    _known_keys(c, {"lengths_s", "min_valid_frac", "filter"}, cw)
    lengths = tuple(_num(x, f"{cw}.lengths_s") for x in _req(c, "lengths_s", cw))
    if not lengths or 10.0 not in lengths:
        raise ConfigError(f"{cw}.lengths_s must include the 10-s context")
    filt = _req(c, "filter", cw)
    _known_keys(filt, {"fir_design", "fir_window", "phase"}, f"{cw}.filter")

    alpha = AlphaConfig(
        data_scale=scale, windows_ms=windows, edge_margin_s=margin,
        use_edge_flags=_bool(_req(wi, "use_edge_flags", ww), f"{ww}.use_edge_flags"),
        check_segments=_bool(_req(wi, "check_segments", ww), f"{ww}.check_segments"),
        bad_annotation_prefix=_str(_req(wi, "bad_annotation_prefix", ww), f"{ww}.bad_annotation_prefix"),
        artefact_rule=rule, artefact_ptp_uv=ptp,
        artefact_keep_equal=_bool(_req(ar, "keep_equal", arw), f"{arw}.keep_equal"),
        rois=rois, nfft=nfft, dpss=dpss, mt4_bandwidth_hz=mt4, welch=welch,
        iaf_half_width_hz=_num(_req(b, "iaf_half_width_hz", bw), f"{bw}.iaf_half_width_hz"),
        band_tol_hz=_num(_req(b, "inclusive_tol_hz", bw), f"{bw}.inclusive_tol_hz"),
        fixed_bands=fixed,
        split_time_tasks=_str_list(_req(st, "tasks", f"{aw}.split_time"), f"{aw}.split_time.tasks"),
        erp_tasks=_str_list(_req(e, "tasks", ew), f"{ew}.tasks"),
        erp_epoch_s=_num(_req(e, "epoch_s", ew), f"{ew}.epoch_s"),
        erp_min_epochs=_int(_req(e, "min_epochs", ew), f"{ew}.min_epochs"),
        erp_leave_out_until=lo_until,
        pc1_tasks=_str_list(_req(p, "tasks", f"{aw}.pc1"), f"{aw}.pc1.tasks"),
        pc1_window_ms=_pair(_req(p, "post_window_ms", f"{aw}.pc1"), f"{aw}.pc1.post_window_ms"),
        context_lengths_s=lengths,
        context_min_valid_frac=_num(_req(c, "min_valid_frac", cw), f"{cw}.min_valid_frac"),
        context_filter={str(k): _str(v, f"{cw}.filter.{k}") for k, v in filt.items()},
        spectra_fmax_hz=_num(_req(a, "spectra_fmax_hz", aw), f"{aw}.spectra_fmax_hz"),
    )

    dd = _req(raw, "decomposition", w)
    dw = f"{w}:decomposition"
    _known_keys(dd, {"min_block_windows", "merge_mode", "sd_ddof", "blocks", "harm_half_window_s", "harm_scope",
                     "hist_n_prev", "hist_min_prev", "context_lengths_s", "reliability_min_windows"}, dw)
    blocks = {str(k): _str_list(v, f"{dw}.blocks.{k}") for k, v in _req(dd, "blocks", dw).items()}
    mm = _req(dd, "merge_mode", dw)
    if isinstance(mm, Mapping):
        modes = {str(k): _str(v, f"{dw}.merge_mode.{k}") for k, v in mm.items()}
        if set(modes) != set(blocks):
            raise ConfigError(f"{dw}.merge_mode needs exactly the tasks of {dw}.blocks ({sorted(blocks)})")
    else:
        modes = {k: _str(mm, f"{dw}.merge_mode") for k in blocks}
    bad = {k: v for k, v in modes.items() if v not in MERGE_MODES}
    if bad:
        raise ConfigError(f"{dw}.merge_mode must be one of {MERGE_MODES} (got {bad})")
    dec = DecompositionConfig(
        min_block_windows=_int(_req(dd, "min_block_windows", dw), f"{dw}.min_block_windows"),
        merge_modes=modes,
        sd_ddof=_int(_req(dd, "sd_ddof", dw), f"{dw}.sd_ddof"),
        blocks=blocks,
        harm_half_window_s=_num(_req(dd, "harm_half_window_s", dw), f"{dw}.harm_half_window_s"),
        harm_scope=_str(_req(dd, "harm_scope", dw), f"{dw}.harm_scope"),
        hist_n_prev=_int(_req(dd, "hist_n_prev", dw), f"{dw}.hist_n_prev"),
        hist_min_prev=_int(_req(dd, "hist_min_prev", dw), f"{dw}.hist_min_prev"),
        context_lengths_s=tuple(_num(x, f"{dw}.context_lengths_s") for x in _req(dd, "context_lengths_s", dw)),
        reliability_min_windows=_int(_req(dd, "reliability_min_windows", dw), f"{dw}.reliability_min_windows"),
    )
    if dec.harm_scope not in HARM_SCOPES:
        raise ConfigError(f"{dw}.harm_scope must be one of {HARM_SCOPES} (got {dec.harm_scope!r})")
    if not 1 <= dec.hist_min_prev <= dec.hist_n_prev:
        raise ConfigError(f"{dw}: need 1 <= hist_min_prev <= hist_n_prev")
    if set(dec.context_lengths_s) != set(alpha.context_lengths_s):
        raise ConfigError(f"{dw}.context_lengths_s must equal alpha.context.lengths_s")
    return FeaturesConfig(version=version, alpha=alpha, decomposition=dec, source=source, raw=dict(raw))


def load_features_config(path: str | os.PathLike[str] | None = None,
                         root: str | os.PathLike[str] | None = None) -> FeaturesConfig:
    """Load ``configs/features.yaml`` (default: under :func:`find_project_root`)."""
    if path is None:
        root_p = Path(root) if root is not None else find_project_root()
        path = root_p / "configs" / "features.yaml"
    p = Path(path)
    return parse_features_config(read_yaml(p), where=str(p), source=p.resolve())


def check_consistency(fcfg: FeaturesConfig, events_cfg=None, preprocessing_cfg=None) -> None:
    """Raise :class:`ConfigError` if ``features.yaml`` disagrees with the reported windows, the events or
    preprocessing config."""
    a = fcfg.alpha
    for task, spec in SPEC_WINDOWS_MS.items():
        got = a.windows_ms.get(task)
        if got != spec:
            raise ConfigError(f"features.yaml windows for {task} {got} != {spec}")
    if events_cfg is not None:
        for task, want in a.windows_ms.items():
            tc = events_cfg.task(task)
            have = {k: (float(v[0]), float(v[1])) for k, v in dict(tc.prestim_windows).items()}
            if have != want:
                raise ConfigError(f"events.yaml {task} prestim_windows_ms {have} != features.yaml {want}")
            m = getattr(tc, "prestim_edge_margin_s", None)
            if m is not None and not math.isclose(float(m), a.edge_margin_s):
                raise ConfigError(f"events.yaml {task} prestim_edge_margin_s {m} != features.yaml {a.edge_margin_s}")
    if preprocessing_cfg is not None:
        thr = float(preprocessing_cfg.artefacts.roi_ptp_uv)
        if not math.isclose(thr, a.artefact_ptp_uv):
            raise ConfigError(f"preprocessing.yaml artefacts.roi_ptp_uv {thr} != features.yaml artefact.ptp_uv "
                              f"{a.artefact_ptp_uv}")
        roi = tuple(getattr(preprocessing_cfg.artefacts, "roi", ()) or ())
        if roi and set(roi) != set(a.rois["primary"].channels):
            raise ConfigError("preprocessing.yaml artefacts.roi differs from the features.yaml primary ROI")
