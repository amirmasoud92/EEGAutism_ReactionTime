"""Load and validate ``configs/ocular.yaml`` (ocular covariates and EXG QC).

The loader returns an immutable :class:`OcularConfig`.

Sections and who uses them:

- ``channels``, ``exg_qc``: preprocessing (EXG QC, ocular-copy sources);
- ``ocular_copy``: the stored non-high-passed ocular copy;
- ``indices``, ``blinks``, ``saccades``: index computation, blink and saccade detection on the stored copy
  (:mod:`preprocessing.ocular_indices`). ``saccades`` is optional.
"""

from __future__ import annotations

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

#: Prestimulus alpha windows ([start, stop) ms).
SPEC_WINDOWS_MS: dict[str, tuple[float, float]] = {"AVSRT": (-850.0, -50.0), "FAST": (-550.0, -50.0),
                                                   "IC": (-550.0, -50.0)}


def _pair(v: Any, where: str, n: int = 2) -> tuple[float, ...]:
    if not isinstance(v, list) or len(v) != n:
        raise ConfigError(f"{where}: expected a list of {n} numbers")
    return tuple(_num(x, where) for x in v)


@dataclass(frozen=True)
class OcularChannels:
    fp: tuple[str, ...]
    heog_pair: tuple[str, str]
    mastoids: tuple[str, str]
    heog_fallback_pair: tuple[str, str]
    ref_fallback: tuple[str, str]
    allow_single_fp: bool

    @property
    def all_inputs(self) -> tuple[str, ...]:
        """Every raw channel the ocular copy may use, in a fixed order, without duplicates."""
        return tuple(dict.fromkeys((*self.fp, *self.heog_pair, *self.mastoids, *self.heog_fallback_pair,
                                    *self.ref_fallback)))


@dataclass(frozen=True)
class OcularCopyConfig:
    lowpass_slow_hz: float
    lowpass_fast_hz: float
    fir_design: str
    fir_window: str
    decim: int
    store_lp15_constituents: bool


@dataclass(frozen=True)
class IndicesConfig:
    windows_ms: dict[str, tuple[float, float]]
    closure_window_ms: tuple[float, float]
    running_median_half_s: float
    running_median_min_coverage: float
    logsd_detrend: str
    log_base: float


@dataclass(frozen=True)
class BlinkConfig:
    band_hz: tuple[float, float]
    polarity: str
    k_mad: float
    min_prominence_uv: float
    max_threshold_uv: float
    sigma_iterations: int
    width_s: tuple[float, float]
    min_separation_s: float
    local_wlen_s: float
    local_max_width_s: float
    extent_pad_fwhm: float


@dataclass(frozen=True)
class SaccadeConfig:
    """``saccades`` section: the HEOG step rule (:func:`preprocessing.ocular_indices.detect_saccades`).

    ``rule = "step"``: a sample ``t`` is supra-threshold if ``max_{1<=l<=k} |x[t+l] - x[t]| > step_uv`` with ``k``
    the largest lag within ``within_ms`` (default ``step_uv`` 30, ``within_ms`` 50).
    ``rule = "step_like"``: ``|mean(x[t, t+h)) - mean(x[t-h, t))| > step_uv`` with ``h = within_ms / 2`` (the
    moving-window step function of ERPLAB-style HEOG screening). Consecutive supra-threshold samples form one
    event; events closer than ``refractory_ms`` are merged. With ``exclude_blinks``, an event whose extent overlaps
    a blink extent (the ``blinks:`` detector on the VEOG proxy) widened by ``blink_pad_ms`` on each side is dropped,
    because blinks leak into the outer-canthus HEOG.
    """

    rule: str
    signal: str
    step_uv: float
    within_ms: float
    refractory_ms: float = 0.0
    exclude_blinks: bool = False
    blink_pad_ms: float = 0.0


@dataclass(frozen=True)
class ExgQcConfig:
    channels: tuple[str, ...]
    band_hz: tuple[float, float]
    chunk_s: float
    sat_threshold_mv: float
    bad_chunk_sd_uv: float
    bad_chunk_absmean_mv: float
    bad_chunk_step_mv: float | None
    max_frac_sat: float
    max_frac_bad_chunks: float
    min_median_chunk_sd_uv: float
    max_median_chunk_sd_uv: float
    min_bad_chunks: int = 1           # frac_bad_chunks fails only if it also rests on >= this many chunks


@dataclass(frozen=True)
class OcularConfig:
    """Validated ``configs/ocular.yaml``."""

    channels: OcularChannels
    ocular_copy: OcularCopyConfig
    indices: IndicesConfig
    blinks: BlinkConfig
    exg_qc: ExgQcConfig
    source: Path | None
    raw: dict = field(repr=False, compare=False)
    saccades: SaccadeConfig | None = None


def parse_ocular_config(raw: Mapping[str, Any], where: str = "ocular config",
                        source: Path | None = None) -> OcularConfig:
    """Validate a parsed ``ocular.yaml`` mapping.

    Raises
    ------
    ConfigError
        On missing or unknown keys, wrong types or inconsistent values.
    """
    w = where
    _known_keys(raw, {"channels", "ocular_copy", "indices", "blinks", "exg_qc", "saccades"}, w)

    c = _req(raw, "channels", w)
    cw = f"{w}:channels"
    _known_keys(c, {"fp", "heog_pair", "mastoids", "heog_fallback_pair", "ref_fallback", "allow_single_fp"}, cw)

    def two(key: str) -> tuple[str, str]:
        v = _str_list(_req(c, key, cw), f"{cw}.{key}")
        if len(v) != 2:
            raise ConfigError(f"{cw}.{key}: expected exactly two channel names")
        return (v[0], v[1])

    fp = _str_list(_req(c, "fp", cw), f"{cw}.fp")
    if not fp:
        raise ConfigError(f"{cw}.fp: at least one channel")
    channels = OcularChannels(fp=fp, heog_pair=two("heog_pair"), mastoids=two("mastoids"),
                              heog_fallback_pair=two("heog_fallback_pair"), ref_fallback=two("ref_fallback"),
                              allow_single_fp=_bool(c.get("allow_single_fp", True), f"{cw}.allow_single_fp"))

    o = _req(raw, "ocular_copy", w)
    ow = f"{w}:ocular_copy"
    _known_keys(o, {"lowpass_slow_hz", "lowpass_fast_hz", "fir_design", "fir_window", "decim",
                    "store_lp15_constituents"}, ow)
    oc = OcularCopyConfig(
        lowpass_slow_hz=_num(_req(o, "lowpass_slow_hz", ow), f"{ow}.lowpass_slow_hz"),
        lowpass_fast_hz=_num(_req(o, "lowpass_fast_hz", ow), f"{ow}.lowpass_fast_hz"),
        fir_design=_str(o.get("fir_design", "firwin"), f"{ow}.fir_design"),
        fir_window=_str(o.get("fir_window", "hamming"), f"{ow}.fir_window"),
        decim=_int(_req(o, "decim", ow), f"{ow}.decim"),
        store_lp15_constituents=_bool(o.get("store_lp15_constituents", True), f"{ow}.store_lp15_constituents"))
    if not 0 < oc.lowpass_slow_hz <= oc.lowpass_fast_hz:
        raise ConfigError(f"{ow}: need 0 < lowpass_slow_hz <= lowpass_fast_hz")
    if oc.decim < 1:
        raise ConfigError(f"{ow}.decim must be >= 1")

    i = _req(raw, "indices", w)
    iw = f"{w}:indices"
    _known_keys(i, {"windows_ms", "closure_window_ms", "running_median_half_s", "running_median_min_coverage",
                    "logsd_detrend", "log_base"}, iw)
    wins_raw = _req(i, "windows_ms", iw)
    if not isinstance(wins_raw, Mapping) or not wins_raw:
        raise ConfigError(f"{iw}.windows_ms: expected a mapping task -> [start, stop]")
    wins = {str(k): _pair(v, f"{iw}.windows_ms.{k}") for k, v in wins_raw.items()}
    for k, (a, b) in wins.items():
        if not a < b <= 0:
            raise ConfigError(f"{iw}.windows_ms.{k}: need start < stop <= 0 (prestimulus)")
    cl = _pair(_req(i, "closure_window_ms", iw), f"{iw}.closure_window_ms")
    if not cl[0] < cl[1] <= 0:
        raise ConfigError(f"{iw}.closure_window_ms: need start < stop <= 0")
    detrend = _str(i.get("logsd_detrend", "linear"), f"{iw}.logsd_detrend")
    if detrend not in ("linear", "constant"):
        raise ConfigError(f"{iw}.logsd_detrend: 'linear' or 'constant'")
    cov = _num(i.get("running_median_min_coverage", 0.5), f"{iw}.running_median_min_coverage")
    if not 0 < cov <= 1:
        raise ConfigError(f"{iw}.running_median_min_coverage must be in (0, 1]")
    base = _num(i.get("log_base", 10), f"{iw}.log_base")
    if base <= 1:
        raise ConfigError(f"{iw}.log_base must be > 1")
    ind = IndicesConfig(windows_ms=wins, closure_window_ms=(cl[0], cl[1]),
                        running_median_half_s=_num(_req(i, "running_median_half_s", iw),
                                                   f"{iw}.running_median_half_s"),
                        running_median_min_coverage=cov, logsd_detrend=detrend, log_base=base)

    b = _req(raw, "blinks", w)
    bw = f"{w}:blinks"
    _known_keys(b, {"band_hz", "polarity", "k_mad", "min_prominence_uv", "max_threshold_uv", "sigma_iterations",
                    "width_s", "min_separation_s", "local_wlen_s", "local_max_width_s", "extent_pad_fwhm"}, bw)
    pol = _str(b.get("polarity", "positive"), f"{bw}.polarity")
    if pol not in ("positive", "negative"):
        raise ConfigError(f"{bw}.polarity: 'positive' or 'negative'")
    bl = BlinkConfig(
        band_hz=_pair(_req(b, "band_hz", bw), f"{bw}.band_hz"), polarity=pol,
        k_mad=_num(_req(b, "k_mad", bw), f"{bw}.k_mad"),
        min_prominence_uv=_num(_req(b, "min_prominence_uv", bw), f"{bw}.min_prominence_uv"),
        max_threshold_uv=_num(_req(b, "max_threshold_uv", bw), f"{bw}.max_threshold_uv"),
        sigma_iterations=_int(b.get("sigma_iterations", 3), f"{bw}.sigma_iterations"),
        width_s=_pair(_req(b, "width_s", bw), f"{bw}.width_s"),
        min_separation_s=_num(_req(b, "min_separation_s", bw), f"{bw}.min_separation_s"),
        local_wlen_s=_num(b.get("local_wlen_s", 1.0), f"{bw}.local_wlen_s"),
        local_max_width_s=_num(b.get("local_max_width_s", 0.5), f"{bw}.local_max_width_s"),
        extent_pad_fwhm=_num(b.get("extent_pad_fwhm", 0.5), f"{bw}.extent_pad_fwhm"))
    if bl.min_prominence_uv > bl.max_threshold_uv:
        raise ConfigError(f"{bw}: min_prominence_uv > max_threshold_uv")
    if bl.local_wlen_s <= 0 or bl.local_max_width_s <= 0 or bl.extent_pad_fwhm < 0:
        raise ConfigError(f"{bw}: local_wlen_s, local_max_width_s > 0 and extent_pad_fwhm >= 0 required")
    if not 0 < bl.band_hz[0] < bl.band_hz[1]:
        raise ConfigError(f"{bw}.band_hz: need 0 < low < high")
    if bl.band_hz[1] > oc.lowpass_fast_hz:
        raise ConfigError(f"{bw}.band_hz upper edge must not exceed ocular_copy.lowpass_fast_hz")

    e = _req(raw, "exg_qc", w)
    ew = f"{w}:exg_qc"
    _known_keys(e, {"channels", "band_hz", "chunk_s", "sat_threshold_mv", "bad_chunk_sd_uv", "bad_chunk_absmean_mv",
                    "bad_chunk_step_mv", "max_frac_sat", "max_frac_bad_chunks", "min_median_chunk_sd_uv",
                    "max_median_chunk_sd_uv", "min_bad_chunks"}, ew)
    step = e.get("bad_chunk_step_mv")
    ex = ExgQcConfig(
        channels=_str_list(_req(e, "channels", ew), f"{ew}.channels"),
        band_hz=_pair(_req(e, "band_hz", ew), f"{ew}.band_hz"),
        chunk_s=_num(_req(e, "chunk_s", ew), f"{ew}.chunk_s"),
        sat_threshold_mv=_num(_req(e, "sat_threshold_mv", ew), f"{ew}.sat_threshold_mv"),
        bad_chunk_sd_uv=_num(_req(e, "bad_chunk_sd_uv", ew), f"{ew}.bad_chunk_sd_uv"),
        bad_chunk_absmean_mv=_num(_req(e, "bad_chunk_absmean_mv", ew), f"{ew}.bad_chunk_absmean_mv"),
        bad_chunk_step_mv=None if step is None else _num(step, f"{ew}.bad_chunk_step_mv"),
        max_frac_sat=_num(_req(e, "max_frac_sat", ew), f"{ew}.max_frac_sat"),
        max_frac_bad_chunks=_num(_req(e, "max_frac_bad_chunks", ew), f"{ew}.max_frac_bad_chunks"),
        min_median_chunk_sd_uv=_num(_req(e, "min_median_chunk_sd_uv", ew), f"{ew}.min_median_chunk_sd_uv"),
        max_median_chunk_sd_uv=_num(_req(e, "max_median_chunk_sd_uv", ew), f"{ew}.max_median_chunk_sd_uv"),
        min_bad_chunks=int(_num(e.get("min_bad_chunks", 1), f"{ew}.min_bad_chunks")))
    if ex.min_bad_chunks < 1:
        raise ConfigError(f"{ew}.min_bad_chunks must be >= 1")
    if ex.chunk_s <= 0 or not 0 < ex.band_hz[0] < ex.band_hz[1]:
        raise ConfigError(f"{ew}: chunk_s > 0 and 0 < band low < band high required")
    if ex.min_median_chunk_sd_uv >= ex.max_median_chunk_sd_uv:
        raise ConfigError(f"{ew}: min_median_chunk_sd_uv must be < max_median_chunk_sd_uv")

    sac = None
    if raw.get("saccades") is not None:
        s = raw["saccades"]
        sw = f"{w}:saccades"
        if not isinstance(s, Mapping):
            raise ConfigError(f"{sw}: expected a mapping")
        _known_keys(s, {"rule", "signal", "step_uv", "within_ms", "refractory_ms", "exclude_blinks", "blink_pad_ms"},
                    sw)
        rule = _str(s.get("rule", "step"), f"{sw}.rule")
        if rule not in ("step", "step_like"):
            raise ConfigError(f"{sw}.rule: 'step' or 'step_like'")
        sig = _str(s.get("signal", "HEOG_lp15"), f"{sw}.signal")
        if sig not in ("HEOG_lp15", "HEOG"):
            raise ConfigError(f"{sw}.signal: 'HEOG_lp15' (15-Hz copy) or 'HEOG' (5-Hz copy)")
        sac = SaccadeConfig(rule=rule, signal=sig, step_uv=_num(_req(s, "step_uv", sw), f"{sw}.step_uv"),
                            within_ms=_num(_req(s, "within_ms", sw), f"{sw}.within_ms"),
                            refractory_ms=_num(s.get("refractory_ms", 0.0), f"{sw}.refractory_ms"),
                            exclude_blinks=_bool(s.get("exclude_blinks", False), f"{sw}.exclude_blinks"),
                            blink_pad_ms=_num(s.get("blink_pad_ms", 0.0), f"{sw}.blink_pad_ms"))
        if sac.step_uv <= 0 or sac.within_ms <= 0 or sac.refractory_ms < 0 or sac.blink_pad_ms < 0:
            raise ConfigError(f"{sw}: step_uv > 0, within_ms > 0, refractory_ms >= 0 and blink_pad_ms >= 0 required")

    return OcularConfig(channels=channels, ocular_copy=oc, indices=ind, blinks=bl, exg_qc=ex, source=source,
                        raw=dict(raw), saccades=sac)


def load_ocular_config(path: str | os.PathLike[str] | None = None,
                       root: str | os.PathLike[str] | None = None) -> OcularConfig:
    """Load ``configs/ocular.yaml`` (default: under :func:`find_project_root`)."""
    if path is None:
        root_p = Path(root) if root is not None else find_project_root()
        path = root_p / "configs" / "ocular.yaml"
    p = Path(path)
    return parse_ocular_config(read_yaml(p), where=str(p), source=p.resolve())


#: Keys of the ``ocular:`` stub in ``configs/preprocessing.yaml`` -> accessor on OcularConfig (consistency check).
_STUB_EXG_KEYS: dict[str, Any] = {
    "channels": lambda c: list(c.exg_qc.channels),
    "sat_threshold_v": lambda c: c.exg_qc.sat_threshold_mv / 1e3,
    "frac_sat_max": lambda c: c.exg_qc.max_frac_sat,
    "chunk_s": lambda c: c.exg_qc.chunk_s,
    "chunk_sd_max_uv": lambda c: c.exg_qc.bad_chunk_sd_uv,
    "frac_bad_chunks_max": lambda c: c.exg_qc.max_frac_bad_chunks,
    "sd_min_uv": lambda c: c.exg_qc.min_median_chunk_sd_uv,
    "sd_max_uv": lambda c: c.exg_qc.max_median_chunk_sd_uv,
}


def check_preprocessing_stub(stub: Mapping[str, Any], cfg: OcularConfig) -> None:
    """Raise :class:`ConfigError` if the ``ocular:`` section of ``preprocessing.yaml`` disagrees with ``ocular.yaml``.

    ``configs/ocular.yaml`` is the source of truth for this module; the stub
    may repeat a subset of its values (channel list, EXG QC thresholds), which
    must then be equal.
    """
    w = "preprocessing.yaml:ocular"
    if "channels" in stub:
        got = set(_str_list(stub["channels"], f"{w}.channels"))
        if got != set(cfg.channels.all_inputs):
            raise ConfigError(f"{w}.channels {sorted(got)} != ocular.yaml inputs {sorted(cfg.channels.all_inputs)}")
    e = stub.get("exg_qc") or {}
    if not isinstance(e, Mapping):
        raise ConfigError(f"{w}.exg_qc must be a mapping")
    for k, v in e.items():
        if k not in _STUB_EXG_KEYS:
            raise ConfigError(f"{w}.exg_qc.{k}: unknown key (ocular.yaml owns the EXG QC schema)")
        want = _STUB_EXG_KEYS[k](cfg)
        same = (list(v) == want) if isinstance(want, list) else (
            isinstance(v, (int, float)) and abs(float(v) - float(want)) <= 1e-12 * max(1.0, abs(float(want))))
        if not same:
            raise ConfigError(f"{w}.exg_qc.{k} = {v!r} disagrees with configs/ocular.yaml ({want!r})")


def as_ocular_config(cfg: Any) -> OcularConfig:
    """Coerce what callers pass as ``cfg`` into an :class:`OcularConfig`.

    Accepted:

    - an :class:`OcularConfig`;
    - a full ``ocular.yaml`` mapping (sections ``channels``, ``ocular_copy``,
      ``indices``, ``blinks``, ``exg_qc``);
    - ``None``: the project's ``configs/ocular.yaml``;
    - a mapping with ``config_file`` (path to an ``ocular.yaml``);
    - the ``preprocessing.yaml`` config object (with an ``ocular`` attribute) or
      its ``ocular:`` stub mapping: ``configs/ocular.yaml`` is loaded and the
      stub is checked against it (:func:`check_preprocessing_stub`).
    """
    if cfg is None:
        return load_ocular_config()
    if isinstance(cfg, OcularConfig):
        return cfg
    if not isinstance(cfg, Mapping) and hasattr(cfg, "ocular"):
        root = getattr(getattr(cfg, "source", None), "parent", None)
        stub = getattr(cfg, "ocular") or {}
        full = load_ocular_config(root=root.parent if root is not None and (root.parent / "configs").is_dir()
                                  else None)
        check_preprocessing_stub(stub, full)
        return full
    if isinstance(cfg, Mapping):
        if "config_file" in cfg:
            return load_ocular_config(cfg["config_file"])
        if {"ocular_copy", "indices", "blinks"} & set(cfg):
            return parse_ocular_config(cfg)
        if set(cfg) <= {"channels", "exg_qc"}:  # the preprocessing.yaml stub
            full = load_ocular_config()
            check_preprocessing_stub(cfg, full)
            return full
        return parse_ocular_config(cfg)
    raise TypeError(f"cfg must be an OcularConfig, a mapping or None, not {type(cfg).__name__}")
