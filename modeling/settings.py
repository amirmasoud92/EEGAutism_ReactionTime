"""Run settings of the model-based analyses: SESOI values, seeds, numbers of surrogates, bootstrap resamples and
sampler iterations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["AnalysisSettings", "A2_SEEDS", "SESOI_OF", "H2_SESOI", "PC3_EXPECTED", "TRUE_SCORE_RHO_BAR"]

TRUE_SCORE_RHO_BAR = 0.642                    # pooled AVSRT split-time reliability of alpha

#: SESOI per test (observed scale; S4 and the cross-task contrasts in r units).
SESOI_OF: dict[str, float] = {"P1": 0.0102, "H1A": 0.0102, "S1": 0.077, "S2": 0.0066, "S3": 0.0084, "S4": 0.03,
                              "S5": 0.0102, "S6": 0.0102}
#: SESOI of the group difference: the same value as the task's slope SESOI; M5-H2 in r units.
H2_SESOI: dict[str, float] = {"M1-H2": 0.0102, "M3-H2": 0.0066, "M4-H2": 0.0084, "M5-H2": 0.03}

#: Seeds of the A2 / A2-L circular-shift surrogates.
A2_SEEDS: dict[str, int] = {"P1": 6_100_001, "H1A": 6_100_002, "S5": 6_100_003, "S2": 6_100_011,
                            "S3": 6_100_021, "S4": 6_100_031, "S6": 6_100_041, "S1": 6_100_051}

#: Behavioural model check (PC3): contrasts and expected signs.
PC3_EXPECTED: dict[str, int] = {"PC3_modality_cAV": -1, "PC3_soa_q75_minus_q25": -1, "PC3_prev_miss": +1,
                                "PC3_soa_prev": -1}


@dataclass(frozen=True)
class AnalysisSettings:
    """Run settings (the defaults are the values used for the reported results)."""

    n_surrogates: int = 10_000            # A2 / A2-L circular-shift surrogates
    srel_B: int = 499                     # S-REL child bootstrap
    srel_seed: int = 20261020
    e3c_B: int = 1_000                    # E3c parametric bootstrap
    e3c_seed: int = 20261020
    smatch_reps: int = 20                 # S-MATCH subsamples
    smatch_seed: int = 20261020
    cens_iter: int = 4_000                # S-CENS (brms), warmup iter / 2
    cens_chains: int = 4
    m6_iter: int = 4_000                  # M6 ex-Gaussian model, warmup iter / 2, seed 20261020
    m6_refit_iter: int = 8_000            # one refit if not converged, warmup 4,000, seed 20261021
    m6_chains: int = 4
    m6_seed: int = 20261020
    m6_refit_seed: int = 20261021
    second_impl: bool = True              # statsmodels second implementation of the LMMs
    run_a1: bool = True                   # A1 (nlme AR(1))

    def to_dict(self) -> dict:
        return asdict(self)
