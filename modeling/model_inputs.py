"""Ocular covariates of the RT and miss models.

The models (M1-M4, their group forms, M1-hist and every model inheriting the M1 fixed covariates) carry the ocular
covariates ``heog_z`` and ``vfp_z``; ``closure_z`` and ``gaze_dev_z`` enter the descriptive S-OCU analyses only.

Missing-covariate rule. The feature layer (:func:`feature_extraction.ocular_trial.apply_missing_rule`, called by
:func:`feature_extraction.trial_features.ocular_z`) sets a missing ``heog_z`` / ``vfp_z`` to 0 on the model rows and
marks the row in ``heog_missing`` / ``vfp_missing``. :func:`ocular_terms` returns the covariate terms for a given set
of model rows: the two z columns, plus an indicator only if it is True on at least one model row (a constant column
would be inestimable). It refuses rows on which a covariate is still missing, so a complete-case fit can never
silently drop a child.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["PRIMARY_OCULAR", "S_OCU_EXTRA", "ocular_terms", "MissingCovariateError"]

#: Ocular covariates of the models -> their missingness indicator.
PRIMARY_OCULAR: dict[str, str] = {"heog_z": "heog_missing", "vfp_z": "vfp_missing"}
#: Added (descriptively) in S-OCU / S-OCU-H2 only.
S_OCU_EXTRA: tuple[str, ...] = ("closure_z", "gaze_dev_z", "gaze_dev_fallback_child")


class MissingCovariateError(ValueError):
    """A primary ocular covariate is missing on a model row (the missing-covariate rule was not applied)."""


def ocular_terms(rows: pd.DataFrame, *, window: str = "", mask: Sequence[bool] | np.ndarray | None = None
                 ) -> list[str]:
    """Ocular covariate columns of a model fitted on ``rows``.

    Parameters
    ----------
    rows
        The model rows (or a table plus ``mask``) with ``heog_z<window>``, ``vfp_z<window>`` and the indicators.
    window
        Window suffix (``""`` primary, ``"__s_w500"`` ...).
    mask
        Optional boolean selection of the model rows within ``rows``.

    Returns
    -------
    ``["heog_z", "vfp_z"]`` (with the suffix), followed by each indicator that is True on at least one model row.

    Raises
    ------
    MissingCovariateError
        If a covered covariate or its indicator column is absent, or the covariate is non-finite on a model row.
    """
    sel = np.ones(len(rows), bool) if mask is None else np.asarray(mask, dtype=bool)
    if len(sel) != len(rows):
        raise ValueError("mask must align with rows")
    terms: list[str] = []
    indicators: list[str] = []
    for col, ind in PRIMARY_OCULAR.items():
        c, i = f"{col}{window}", f"{ind}{window}"
        if c not in rows.columns or i not in rows.columns:
            raise MissingCovariateError(f"{c} / {i} not in the model table")
        x = pd.to_numeric(rows.loc[sel, c], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all():
            raise MissingCovariateError(f"{c} is missing on {int((~np.isfinite(x)).sum())} model row(s); the "
                                        "missing-covariate rule (0 + indicator) was not applied")
        terms.append(c)
        if pd.array(rows.loc[sel, i], dtype="boolean").fillna(False).to_numpy(bool).any():
            indicators.append(i)
    return terms + indicators
