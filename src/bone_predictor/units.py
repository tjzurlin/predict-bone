"""Unit conversion utilities for bone concentrations and mineral fractions.

Provides bidirectional conversions between:
- Bone ash concentration [mg/kg ash or ppm ash]
- Fresh wet bone concentration [mg/kg wet bone or ppm wet]
- Molar concentration [mmol solute / kg wet bone]

Reference: Rao et al. (1995) Toxicol. Appl. Pharmacol. 135:297-306
"""

from typing import TypeVar, cast

import numpy as np

from bone_predictor.constants import ASH_FRACTION, MOLAR_MASS_FLUORINE

T = TypeVar("T", float, np.ndarray)


def ash_to_wet(
    c_ash: T,
    ash_fraction: float = ASH_FRACTION,
) -> T:
    """Convert bone ash concentration to wet bone concentration (ppm wet bone).

    Parameters
    ----------
    c_ash : float or np.ndarray
        Concentration in mg/kg ash (or ppm ash). Must be non-negative.
    ash_fraction : float, default=0.56
        Mineral ash mass fraction of fresh wet bone.

    Returns
    -------
    float or np.ndarray
        Concentration in mg/kg fresh wet bone.
    """
    arr = np.asarray(c_ash)
    if np.any(arr < 0):
        raise ValueError("Bone concentration must be non-negative")
    if not (0.0 < ash_fraction <= 1.0):
        raise ValueError("Ash fraction must be in (0, 1]")
    res = arr * ash_fraction
    return cast(T, float(res) if np.ndim(c_ash) == 0 else res)


def wet_to_ash(
    c_wet: T,
    ash_fraction: float = ASH_FRACTION,
) -> T:
    """Convert wet bone concentration to bone ash concentration (mg/kg ash).

    Parameters
    ----------
    c_wet : float or np.ndarray
        Concentration in mg/kg fresh wet bone (or ppm wet). Must be non-negative.
    ash_fraction : float, default=0.56
        Mineral ash mass fraction of fresh wet bone.

    Returns
    -------
    float or np.ndarray
        Concentration in mg/kg bone ash.
    """
    arr = np.asarray(c_wet)
    if np.any(arr < 0):
        raise ValueError("Bone concentration must be non-negative")
    if not (0.0 < ash_fraction <= 1.0):
        raise ValueError("Ash fraction must be in (0, 1]")
    res = arr / ash_fraction
    return cast(T, float(res) if np.ndim(c_wet) == 0 else res)


def wet_to_molar(
    c_wet: T,
    molar_mass: float = MOLAR_MASS_FLUORINE,
) -> T:
    """Convert wet bone concentration (mg/kg) to molar concentration (mmol/kg wet bone).

    Parameters
    ----------
    c_wet : float or np.ndarray
        Concentration in mg/kg fresh wet bone. Must be non-negative.
    molar_mass : float, default=18.9984
        Molecular or atomic weight in g/mol (or mg/mmol).

    Returns
    -------
    float or np.ndarray
        Concentration in mmol solute / kg fresh wet bone.
    """
    arr = np.asarray(c_wet)
    if np.any(arr < 0):
        raise ValueError("Bone concentration must be non-negative")
    if molar_mass <= 0:
        raise ValueError("Molar mass must be positive")
    res = arr / molar_mass
    return cast(T, float(res) if np.ndim(c_wet) == 0 else res)


def molar_to_wet(
    c_molar: T,
    molar_mass: float = MOLAR_MASS_FLUORINE,
) -> T:
    """Convert molar concentration (mmol/kg wet bone) to wet bone concentration (mg/kg).

    Parameters
    ----------
    c_molar : float or np.ndarray
        Concentration in mmol solute / kg fresh wet bone. Must be non-negative.
    molar_mass : float, default=18.9984
        Molecular or atomic weight in g/mol (or mg/mmol).

    Returns
    -------
    float or np.ndarray
        Concentration in mg/kg fresh wet bone.
    """
    arr = np.asarray(c_molar)
    if np.any(arr < 0):
        raise ValueError("Bone concentration must be non-negative")
    if molar_mass <= 0:
        raise ValueError("Molar mass must be positive")
    res = arr * molar_mass
    return cast(T, float(res) if np.ndim(c_molar) == 0 else res)


def ash_to_molar(
    c_ash: T,
    ash_fraction: float = ASH_FRACTION,
    molar_mass: float = MOLAR_MASS_FLUORINE,
) -> T:
    """Convert bone ash concentration (mg/kg ash) to molar concentration (mmol/kg wet bone)."""
    return wet_to_molar(ash_to_wet(c_ash, ash_fraction=ash_fraction), molar_mass=molar_mass)


def molar_to_ash(
    c_molar: T,
    ash_fraction: float = ASH_FRACTION,
    molar_mass: float = MOLAR_MASS_FLUORINE,
) -> T:
    """Convert molar concentration (mmol/kg wet bone) to bone ash concentration (mg/kg ash)."""
    return wet_to_ash(molar_to_wet(c_molar, molar_mass=molar_mass), ash_fraction=ash_fraction)


#: Effective bone ash density relative to MCSim mineral volume compartment [kg ash / L mineral].
#: Derivation: (ASH_FRACTION * skeletal_mass_fraction) / (V_Boc * f_BoMi)
#:           = (0.56 * 0.085) / (0.0372 * 0.355) = 0.0476 / 0.01321 = 3.60 kg ash / L mineral
#: See: run-model/papers/reports/boivin_turnover_to_pbpk_mapping.md
RHO_ASH_MINERAL: float = 3.60


def ash_to_mineral_vol(
    c_ash: T,
    rho_mineral: float = RHO_ASH_MINERAL,
) -> T:
    """Convert bone ash concentration (mg/kg ash) to mineral volume concentration (mg/L mineral)."""
    arr = np.asarray(c_ash)
    if np.any(arr < 0):
        raise ValueError("Bone concentration must be non-negative")
    res = arr * rho_mineral
    return cast(T, float(res) if np.ndim(c_ash) == 0 else res)


def mineral_vol_to_ash(
    c_mineral: T,
    rho_mineral: float = RHO_ASH_MINERAL,
) -> T:
    """Convert mineral volume concentration (mg/L mineral) to bone ash concentration (mg/kg ash)."""
    arr = np.asarray(c_mineral)
    if np.any(arr < 0):
        raise ValueError("Bone concentration must be non-negative")
    res = arr / rho_mineral
    return cast(T, float(res) if np.ndim(c_mineral) == 0 else res)


_UNIT_ALIASES: dict[str, str] = {
    "mg/kg_ash": "ash",
    "ppm_ash": "ash",
    "ash": "ash",
    "mg/kg_wet": "wet",
    "ppm_wet": "wet",
    "ppm": "wet",
    "wet": "wet",
    "mmol/kg": "molar",
    "mmol/kg_wet": "molar",
    "molar": "molar",
    "mg/l_mineral": "mineral",
    "mg/l": "mineral",
    "mineral": "mineral",
}


def convert_concentration(
    value: T,
    from_unit: str,
    to_unit: str,
    ash_fraction: float = ASH_FRACTION,
    molar_mass: float = MOLAR_MASS_FLUORINE,
    rho_mineral: float = RHO_ASH_MINERAL,
) -> T:
    """Convert bone concentration between arbitrary supported units.

    Supported unit strings (case-insensitive):
    - Ash: 'mg/kg_ash', 'ppm_ash', 'ash'
    - Wet: 'mg/kg_wet', 'ppm_wet', 'ppm', 'wet'
    - Molar: 'mmol/kg', 'mmol/kg_wet', 'molar'
    - Mineral: 'mg/l_mineral', 'mg/l', 'mineral'

    Parameters
    ----------
    value : float or np.ndarray
        Input concentration value.
    from_unit : str
        Source unit.
    to_unit : str
        Destination unit.
    ash_fraction : float, default=0.56
        Ash mass fraction.
    molar_mass : float, default=18.9984
        Solute molar mass.

    Returns
    -------
    float or np.ndarray
        Converted concentration.
    """
    u_from = _UNIT_ALIASES.get(from_unit.lower().strip())
    u_to = _UNIT_ALIASES.get(to_unit.lower().strip())

    if u_from is None:
        raise ValueError(
            f"Unrecognized source unit '{from_unit}'. Supported: {sorted(set(_UNIT_ALIASES))}"
        )
    if u_to is None:
        raise ValueError(
            f"Unrecognized target unit '{to_unit}'. Supported: {sorted(set(_UNIT_ALIASES))}"
        )

    arr = np.asarray(value)
    if np.any(arr < 0):
        raise ValueError("Bone concentration must be non-negative")

    if u_from == u_to:
        return cast(T, float(arr) if np.ndim(value) == 0 else arr)

    # Convert to intermediate wet (mg/kg)
    if u_from == "ash":
        wet = ash_to_wet(value, ash_fraction=ash_fraction)
    elif u_from == "molar":
        wet = molar_to_wet(value, molar_mass=molar_mass)
    elif u_from == "mineral":
        c_ash = mineral_vol_to_ash(value, rho_mineral=rho_mineral)
        wet = ash_to_wet(c_ash, ash_fraction=ash_fraction)
    else:
        wet = float(arr) if np.ndim(value) == 0 else arr

    # Convert from wet to target
    if u_to == "ash":
        return wet_to_ash(wet, ash_fraction=ash_fraction)
    elif u_to == "molar":
        return wet_to_molar(wet, molar_mass=molar_mass)
    elif u_to == "mineral":
        c_ash = wet_to_ash(wet, ash_fraction=ash_fraction)
        return ash_to_mineral_vol(c_ash, rho_mineral=rho_mineral)
    else:
        return cast(T, wet)
