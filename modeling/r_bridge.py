"""Python -> R bridge of the model engine (R 4.4.2 / lme4 1.1.37 / lmerTest / glmmTMB through ``Rscript`` on TSV
files).

:func:`prepare` writes the model-input TSV (:func:`modeling.model_data.write_model_input`) and the JSON model spec
(:func:`modeling.model_specs.build_spec`); :func:`run_spec` runs the family's script (``R/fit_lmm.R`` or
``R/fit_glmm.R``) and reads its JSON result; :func:`fit` does both. The ``Rscript`` executable is taken from the
``RSCRIPT`` environment variable (default ``Rscript`` on the PATH).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from modeling import model_specs

__all__ = ["RSCRIPT", "R_DIR", "FAMILY_SCRIPTS", "EngineError", "prepare", "run_spec", "fit", "target"]

RSCRIPT = os.environ.get("RSCRIPT", "Rscript")
R_DIR = Path(__file__).resolve().parent / "R"
FAMILY_SCRIPTS = {"lmm": "fit_lmm.R", "glmm": "fit_glmm.R"}


class EngineError(RuntimeError):
    """The R engine failed (non-zero exit or no result)."""


def prepare(md, out_dir: str | os.PathLike[str], label: str, **spec_kw) -> tuple[Path, dict]:
    """Write ``<out_dir>/<label>_input.tsv`` and ``<label>_spec.json``; return ``(spec_path, spec)``."""
    from modeling.model_data import write_model_input

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsv = write_model_input(md, out / f"{label}_input.tsv")
    export = {k: spec_kw.pop(k) for k in ("export_cr2_vcov", "export_fixef_vcov") if k in spec_kw}
    spec = model_specs.build_spec(md, label=label, input_tsv=tsv.as_posix(),
                                  output_json=(out / f"{label}_result.json").as_posix(), **spec_kw)
    spec.update(export)
    spec_path = out / f"{label}_spec.json"
    spec_path.write_text(model_specs.spec_json(spec), encoding="utf-8", newline="\n")
    return spec_path, spec


def run_spec(spec_path: str | os.PathLike[str], *, timeout: float = 6 * 3600, rscript: str = RSCRIPT) -> dict:
    """Run the R engine on a spec JSON; return the result dict."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    script = R_DIR / FAMILY_SCRIPTS[spec["family"]]
    out = Path(spec["output_json"])
    if out.exists():
        out.unlink()
    t0 = time.time()
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    r = subprocess.run([rscript, "--vanilla", str(script), str(spec_path)], capture_output=True, text=True,
                       timeout=timeout, env=env)
    wall = time.time() - t0
    if not out.is_file():
        raise EngineError(f"{script.name} exited {r.returncode} without a result:\n{r.stderr[-3000:]}")
    res = json.loads(out.read_text(encoding="utf-8"))
    if r.returncode != 0 and "error" not in res:
        res["error"] = f"exit {r.returncode}: {r.stderr[-1500:]}"
    res["wall_seconds"] = wall
    return res


def fit(md, out_dir: str | os.PathLike[str], label: str, *, timeout: float = 6 * 3600, **spec_kw) -> dict:
    """:func:`prepare` + :func:`run_spec`. Raises :class:`EngineError` if R returned an error result."""
    spec_path, _ = prepare(md, out_dir, label, **spec_kw)
    res = run_spec(spec_path, timeout=timeout)
    if "error" in res:
        raise EngineError(f"{label}: {res['error']}")
    return res


def target(res: Mapping, name: str, *, cr2: bool = False) -> dict:
    """The model-based (or CR2) inference row of a target in a result."""
    if cr2:
        return res["cr2"]["targets"][name]
    return res["targets"][name]
