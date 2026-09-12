"""Physical and physiological constants for the bone-predictor package.

All values are aligned with the PBKM Manual (O'Flaherty & Reponen 1997),
Hawkins et al. (1992), and Rao et al. (1995).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

# ==============================================================================
# Hawkins [18F]Fluoride 2-Tissue PET Reference Constants (Adult Baseline)
# Hawkins et al. (1992) J. Nucl. Med. 33(4):633-642
# ==============================================================================
HAWKINS_K1: float = 0.106  # Plasma-to-ECF clearance [mL/min/mL bone]
HAWKINS_k2: float = 0.258  # ECF-to-plasma reverse rate [min^-1]
HAWKINS_k3_ADULT: float = 0.132  # Mineral incorporation rate in adults [min^-1]
HAWKINS_k4_ADULT: float = 0.002  # Mineral release rate in adults [min^-1]
HAWKINS_Ki_ADULT: float = 0.0359  # Net influx clearance [mL/min/mL bone]
HAWKINS_V_B: float = 0.050  # Bone vascular volume fraction [mL blood / mL bone]

# ==============================================================================
# Skeletal Architecture & Remodeling Baselines (PBKM Manual Section 3.1 & 3.3)
# ==============================================================================
ADULT_TURNOVER_BASELINE: float = 0.10  # Whole-skeleton turnover [yr^-1] (10% / year)
CORTICAL_VOLUME_FRACTION: float = 0.80  # Cortical bone fraction of total volume (80%)
TRABECULAR_VOLUME_FRACTION: float = 0.20  # Trabecular bone fraction of total volume (20%)
CORTICAL_TURNOVER_APPORTIONMENT: float = 0.35  # Cortical share of total turnover (35%)
TRABECULAR_TURNOVER_APPORTIONMENT: float = 0.65  # Trabecular share of total turnover (65%)

# Derived baseline fractional rates [yr^-1]
# TFBFR = (0.65 / 0.20) * 0.10 = 0.325 yr^-1 (32.5% / year)
TRABECULAR_TURNOVER_BASELINE: float = (
    TRABECULAR_TURNOVER_APPORTIONMENT / TRABECULAR_VOLUME_FRACTION
) * ADULT_TURNOVER_BASELINE

# CFBFR = (0.35 / 0.80) * 0.10 = 0.04375 yr^-1 (4.375% / year)
CORTICAL_TURNOVER_BASELINE: float = (
    CORTICAL_TURNOVER_APPORTIONMENT / CORTICAL_VOLUME_FRACTION
) * ADULT_TURNOVER_BASELINE

# ==============================================================================
# Bone Composition & Chemical Constants (Rao et al. 1995; Boivin et al. 1989)
# ==============================================================================
ASH_FRACTION: float = 0.56  # Ash weight fraction of fresh wet bone (56%)
C_CONTROL: float = 282.0  # Adult unexposed baseline bone fluoride concentration [mg/kg ash]
# Adult normalized calcium accretion flux constants [kg Ca / (kg body weight * yr)]
# Sex-specific baselines: Males = 0.00263, Females = 0.00304 (PBKM Manual Table 1)
C_ADULT_BY_SEX: dict[str, float] = {
    "F": 0.00304,
    "M": 0.00263,
}


def _normalize_sex_constant(sex: Any) -> str:
    """Normalize sex indicator specifically within constants module."""
    if isinstance(sex, bool):
        raise ValueError("Boolean values are not valid sex indicators")
    s_str = str(sex).strip().upper()
    if s_str in ("M", "MALE", "1"):
        return "M"
    if s_str in ("F", "FEMALE", "0"):
        return "F"
    raise ValueError(f"Unknown sex indicator: {sex!r}. Expected 'M' or 'F'.")


def c_adult(sex: Any) -> float | np.ndarray:
    """Return adult normalized calcium accretion flux [kg Ca / (kg body weight * yr)].

    Parameters
    ----------
    sex : str or array-like
        Biological sex indicator ('M' or 'F').

    Returns
    -------
    float or np.ndarray
        Adult calcium flux constant: 0.00263 for males, 0.00304 for females.
    """
    if isinstance(sex, (list, tuple, np.ndarray)):
        arr = np.asarray(sex)
        vfunc = np.vectorize(lambda s: C_ADULT_BY_SEX[_normalize_sex_constant(s)], otypes=[float])
        return vfunc(arr)
    return C_ADULT_BY_SEX[_normalize_sex_constant(sex)]


class CAdultConstant(float):
    """Adult normalized calcium accretion flux constant with sex-specific dictionary lookup."""

    def __getitem__(self, key: Any) -> float:
        norm_key = _normalize_sex_constant(key)
        return C_ADULT_BY_SEX[norm_key]

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except Exception:
            return default

    def __call__(self, sex: Any) -> float | np.ndarray:
        return c_adult(sex)


C_ADULT: CAdultConstant = CAdultConstant(0.0030)
CA_BIRTH: float = 0.028  # Neonatal skeletal calcium content at birth [kg Ca] (28 g)

MOLAR_MASS_FLUORINE: float = 18.9984  # Fluorine atomic weight [g/mol]
MOLAR_MASS_CALCIUM: float = 40.078  # Calcium atomic weight [g/mol]


# ==============================================================================
# O'Flaherty 6-Parameter Body Weight Growth Parameters (PBKM Manual Table 1)
# ==============================================================================
@dataclass(frozen=True)
class OFlahertyGrowthParams:
    """Parameters for the O'Flaherty 6-parameter human body weight growth model."""

    w_birth: float  # Birth body weight [kg]
    w_child: float  # Hyperbolic growth increment capacity [kg]
    half: float  # Hyperbolic half-capacity age [years]
    w_adult: float  # Adolescent logistic growth increment [kg]
    kappa: float  # Logistic timing multiplier [dimensionless]
    lambda_: float  # Logistic acceleration rate [kg^-1 yr^-1]

    @property
    def mature_weight(self) -> float:
        """Theoretical mature adult asymptotic weight (w_birth + w_child + w_adult) [kg]."""
        return self.w_birth + self.w_child + self.w_adult


OFLAHERTY_MALE = OFlahertyGrowthParams(
    w_birth=3.5,
    w_child=23.0,
    half=3.0,
    w_adult=50.0,
    kappa=600.0,
    lambda_=0.0095,
)

OFLAHERTY_FEMALE = OFlahertyGrowthParams(
    w_birth=3.5,
    w_child=22.0,
    half=3.0,
    w_adult=34.0,
    kappa=600.0,
    lambda_=0.0170,
)

GROWTH_PARAMS: dict[str, OFlahertyGrowthParams] = {
    "M": OFLAHERTY_MALE,
    "F": OFLAHERTY_FEMALE,
}

# ==============================================================================
# Allometric Bone Volume & Mass Scaling Coefficients (PBKM Manual Table 1)
# ==============================================================================
BONE_VOLUME_PREFACTOR: float = 0.0168  # VBONE = 0.0168 * WBODY^1.188 [L]
BONE_VOLUME_EXPONENT: float = 1.188
BONE_MASS_PREFACTOR: float = 0.0290  # WBONE = 0.0290 * WBODY^1.21 [kg]
BONE_MASS_EXPONENT: float = 1.21

# ==============================================================================
# Modeling & Bayesian Workflow Defaults
# ==============================================================================
RANDOM_SEED: int = 42
DEFAULT_HDI_PROB: float = 0.94
