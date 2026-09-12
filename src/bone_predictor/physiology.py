"""Physiological functions and allometric scaling for the human skeleton.

Implements the O'Flaherty 6-parameter human body weight growth equations,
bone volume and mass allometric power-law scaling, and dynamic skeletal
remodeling/resorption balance equations from the PBKM Manual
(O'Flaherty & Reponen 1997).
"""

from typing import Any

import numpy as np

from bone_predictor.constants import (
    ADULT_TURNOVER_BASELINE,
    BONE_MASS_EXPONENT,
    BONE_MASS_PREFACTOR,
    BONE_VOLUME_EXPONENT,
    BONE_VOLUME_PREFACTOR,
    CORTICAL_TURNOVER_APPORTIONMENT,
    CORTICAL_TURNOVER_BASELINE,
    CORTICAL_VOLUME_FRACTION,
    GROWTH_PARAMS,
    TRABECULAR_TURNOVER_APPORTIONMENT,
    TRABECULAR_TURNOVER_BASELINE,
    TRABECULAR_VOLUME_FRACTION,
    OFlahertyGrowthParams,
)


def normalize_sex(sex: Any) -> str:
    """Validate and normalize sex selector to 'M' or 'F'.

    Parameters
    ----------
    sex : str or int
        Sex indicator ('M', 'F', 'male', 'female', 'm', 'f', 1, 0).
        Boolean inputs are strictly rejected.

    Returns
    -------
    str
        Normalized sex string ('M' or 'F').

    Raises
    ------
    ValueError
        If sex selector is invalid or boolean.
    """
    if isinstance(sex, bool):
        raise ValueError("Boolean values are not valid sex indicators")

    s_str = str(sex).strip().upper()
    if s_str in ("M", "MALE", "1"):
        return "M"
    if s_str in ("F", "FEMALE", "0"):
        return "F"
    raise ValueError(f"Unknown sex indicator: {sex!r}. Expected 'M', 'F', 'Male', or 'Female'.")


def body_weight(age: float | np.ndarray, sex: Any) -> float | np.ndarray:
    """Calculate deterministic human body weight [kg] using the O'Flaherty 6-parameter model.

    PBKM Manual Table 1 & Section 3.1.1:
    BW(a) = WBIRTH + (WCHILD * a)/(HALF + a) + WADULT / (1 + KAPPA * exp(-LAMBDA * WADULT * a))

    Parameters
    ----------
    age : float or np.ndarray
        Chronological age in years (>= 0).
    sex : str or int
        Biological sex ('M' or 'F').

    Returns
    -------
    float or np.ndarray
        Body weight in kg.

    Raises
    ------
    ValueError
        If age is negative or non-finite, or if sex is unrecognized.
    """
    norm_sex = normalize_sex(sex)
    params: OFlahertyGrowthParams = GROWTH_PARAMS[norm_sex]

    arr_age = np.asarray(age, dtype=float)
    if not np.all(np.isfinite(arr_age)):
        raise ValueError("Age must contain only finite values")
    if np.any(arr_age < 0):
        raise ValueError("Age must be non-negative")

    hyperbolic = (params.w_child * arr_age) / (params.half + arr_age)
    eff_rate = params.lambda_ * params.w_adult
    logistic = params.w_adult / (1.0 + params.kappa * np.exp(-eff_rate * arr_age))

    bw = params.w_birth + hyperbolic + logistic
    return float(bw) if np.ndim(age) == 0 else bw


def body_weight_derivative(age: float | np.ndarray, sex: Any) -> float | np.ndarray:
    """Calculate analytical derivative of body weight with respect to age (d(BW)/dt) [kg/year].

    Parameters
    ----------
    age : float or np.ndarray
        Chronological age in years (>= 0).
    sex : str or int
        Biological sex ('M' or 'F').

    Returns
    -------
    float or np.ndarray
        Rate of body weight growth in kg/year.
    """
    norm_sex = normalize_sex(sex)
    params: OFlahertyGrowthParams = GROWTH_PARAMS[norm_sex]

    arr_age = np.asarray(age, dtype=float)
    if not np.all(np.isfinite(arr_age)):
        raise ValueError("Age must contain only finite values")
    if np.any(arr_age < 0):
        raise ValueError("Age must be non-negative")

    # d/da [ (WCHILD * a) / (HALF + a) ] = (WCHILD * HALF) / (HALF + a)^2
    d_hyperbolic = (params.w_child * params.half) / ((params.half + arr_age) ** 2)

    # d/da [ WADULT / (1 + KAPPA * exp(-k * a)) ] =
    #   (WADULT * KAPPA * k * exp(-k * a)) / (1 + KAPPA * exp(-k * a))^2
    eff_rate = params.lambda_ * params.w_adult
    exp_term = np.exp(-eff_rate * arr_age)
    denom = 1.0 + params.kappa * exp_term
    d_logistic = (params.w_adult * params.kappa * eff_rate * exp_term) / (denom**2)

    dbw_dt = d_hyperbolic + d_logistic
    return float(dbw_dt) if np.ndim(age) == 0 else dbw_dt


def bone_volume(body_weight_kg: float | np.ndarray) -> float | np.ndarray:
    """Calculate skeletal bone volume [liters] from body weight [kg].

    PBKM Manual Table 1 & Section 3.1.2:
    VBONE = 0.0168 * WBODY^1.188 [L]

    Parameters
    ----------
    body_weight_kg : float or np.ndarray
        Subject body weight in kg (must be > 0).

    Returns
    -------
    float or np.ndarray
        Skeletal volume in liters.
    """
    bw = np.asarray(body_weight_kg, dtype=float)
    if not np.all(np.isfinite(bw)):
        raise ValueError("Body weight must contain only finite values")
    if np.any(bw <= 0):
        raise ValueError("Body weight must be strictly positive")
    vol = BONE_VOLUME_PREFACTOR * (bw**BONE_VOLUME_EXPONENT)
    return float(vol) if np.ndim(body_weight_kg) == 0 else vol


def bone_mass(body_weight_kg: float | np.ndarray) -> float | np.ndarray:
    """Calculate skeletal bone mass [kg] from body weight [kg].

    PBKM Manual Table 1 & Section 3.1.2:
    WBONE = 0.0290 * WBODY^1.21 [kg]

    Parameters
    ----------
    body_weight_kg : float or np.ndarray
        Subject body weight in kg (must be > 0).

    Returns
    -------
    float or np.ndarray
        Skeletal mass in kg.
    """
    bw = np.asarray(body_weight_kg, dtype=float)
    if not np.all(np.isfinite(bw)):
        raise ValueError("Body weight must contain only finite values")
    if np.any(bw <= 0):
        raise ValueError("Body weight must be strictly positive")
    mass = BONE_MASS_PREFACTOR * (bw**BONE_MASS_EXPONENT)
    return float(mass) if np.ndim(body_weight_kg) == 0 else mass


def bone_density(body_weight_kg: float | np.ndarray) -> float | np.ndarray:
    """Calculate apparent skeletal bulk density [kg/L] (WBONE / VBONE).

    Parameters
    ----------
    body_weight_kg : float or np.ndarray
        Subject body weight in kg (> 0).

    Returns
    -------
    float or np.ndarray
        Bone density in kg/L (or g/cm^3).
    """
    return bone_mass(body_weight_kg) / bone_volume(body_weight_kg)


def compartmental_bone_volumes(
    v_bone_liters: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Partition total bone volume into cortical and trabecular compartments.

    PBKM Manual Section 3.1.2:
    CV_BONE = 0.80 * VBONE [L]
    TV_BONE = 0.20 * VBONE [L]

    Parameters
    ----------
    v_bone_liters : float or np.ndarray
        Total skeletal volume in liters.

    Returns
    -------
    tuple of (cv_bone, tv_bone)
        Cortical and trabecular bone volumes in liters.
    """
    vb = np.asarray(v_bone_liters, dtype=float)
    if not np.all(np.isfinite(vb)):
        raise ValueError("Bone volume must contain only finite values")
    if np.any(vb <= 0):
        raise ValueError("Bone volume must be strictly positive")
    cv = CORTICAL_VOLUME_FRACTION * vb
    tv = TRABECULAR_VOLUME_FRACTION * vb
    if np.ndim(v_bone_liters) == 0:
        return float(cv), float(tv)
    return cv, tv


def bone_volume_growth_rate(age: float | np.ndarray, sex: Any) -> float | np.ndarray:
    """Calculate instantaneous rate of bone volume expansion d(VBONE)/dt [liters/year].

    Uses the chain rule:
    d(VBONE)/dt = 0.0168 * 1.188 * WBODY^0.188 * d(WBODY)/dt [L/yr]

    Parameters
    ----------
    age : float or np.ndarray
        Chronological age in years (>= 0).
    sex : str or int
        Biological sex ('M' or 'F').

    Returns
    -------
    float or np.ndarray
        Rate of bone volume expansion in L/year.
    """
    bw = body_weight(age, sex)
    dbw_dt = body_weight_derivative(age, sex)

    # dV/dt = a * b * BW^(b - 1) * dBW/dt
    dv_dt = (
        BONE_VOLUME_PREFACTOR
        * BONE_VOLUME_EXPONENT
        * (np.asarray(bw) ** (BONE_VOLUME_EXPONENT - 1.0))
        * np.asarray(dbw_dt)
    )
    return float(dv_dt) if np.ndim(age) == 0 else dv_dt


def skeletal_remodeling_balance(
    age: float | np.ndarray,
    sex: Any,
    r_form: float | np.ndarray = 1.0,
) -> dict[str, float | np.ndarray]:
    """Calculate skeletal remodeling and resorption balance fluxes according to PBKM Manual.

    Equations:
    - Formation Rate FBFR = 0.10 * r_form [yr^-1]
    - Formation Flux BFR = FBFR * VBONE [L/yr]
    - Bone Volume Growth d(VBONE)/dt = BFR - BRR [L/yr]
    - Resorption Flux BRR = max(0, BFR - d(VBONE)/dt) [L/yr]
    - Resorption Rate FBRR = BRR / VBONE [yr^-1]
    - Compartmental Partitioning:
      - Trabecular: 65% of fluxes in 20% of volume
      - Cortical: 35% of fluxes in 80% of volume

    Parameters
    ----------
    age : float or np.ndarray
        Chronological age in years (>= 0).
    sex : str or int
        Biological sex ('M' or 'F').
    r_form : float or np.ndarray, default=1.0
        Relative bone formation activity index (RAI_ALP).

    Returns
    -------
    dict
        Dictionary containing volume, rates, and fluxes:
        - 'v_bone': Skeletal volume [L]
        - 'dv_dt': Skeletal volume expansion rate [L/yr]
        - 'fbfr': Whole-skeleton fractional formation rate [yr^-1]
        - 'bfr': Whole-skeleton formation volume flux [L/yr]
        - 'brr': Whole-skeleton resorption volume flux [L/yr]
        - 'fbrr': Whole-skeleton fractional resorption rate [yr^-1]
        - 'tfbfr': Trabecular fractional formation rate [yr^-1]
        - 'cfbfr': Cortical fractional formation rate [yr^-1]
        - 'tbfr': Trabecular formation volume flux [L/yr]
        - 'cbfr': Cortical formation volume flux [L/yr]
        - 'tbrr': Trabecular resorption volume flux [L/yr]
        - 'cbrr': Cortical resorption volume flux [L/yr]
    """
    arr_r = np.asarray(r_form, dtype=float)
    if not np.all(np.isfinite(arr_r)):
        raise ValueError("r_form must contain only finite values")
    if np.any(arr_r < 0):
        raise ValueError("r_form must be non-negative")

    arr_age, arr_r = np.broadcast_arrays(np.asarray(age, dtype=float), arr_r)
    bw = body_weight(arr_age, sex)
    vb = bone_volume(bw)
    dv_dt = bone_volume_growth_rate(arr_age, sex)

    # Whole-skeleton formation
    fbfr = ADULT_TURNOVER_BASELINE * arr_r
    bfr = fbfr * vb

    # Skeletal volume conservation: dV/dt = BFR - BRR => BRR = max(0, BFR - dV/dt)
    brr = np.maximum(0.0, bfr - dv_dt)
    fbrr = brr / vb

    # Compartmental rates and fluxes
    tbfr = TRABECULAR_TURNOVER_APPORTIONMENT * bfr
    cbfr = CORTICAL_TURNOVER_APPORTIONMENT * bfr

    tbrr = TRABECULAR_TURNOVER_APPORTIONMENT * brr
    cbrr = CORTICAL_TURNOVER_APPORTIONMENT * brr

    tfbfr = TRABECULAR_TURNOVER_BASELINE * arr_r
    cfbfr = CORTICAL_TURNOVER_BASELINE * arr_r

    res = {
        "v_bone": vb,
        "dv_dt": dv_dt,
        "fbfr": fbfr,
        "bfr": bfr,
        "brr": brr,
        "fbrr": fbrr,
        "tfbfr": tfbfr,
        "cfbfr": cfbfr,
        "tbfr": tbfr,
        "cbfr": cbfr,
        "tbrr": tbrr,
        "cbrr": cbrr,
    }
    if np.ndim(age) == 0 and np.ndim(r_form) == 0:
        return {k: float(v) for k, v in res.items()}
    return res
