"""Unit tests for the physical and physiological constants module."""

import numpy as np
import pytest

from bone_predictor.constants import (
    ADULT_TURNOVER_BASELINE,
    ASH_FRACTION,
    BONE_MASS_EXPONENT,
    BONE_MASS_PREFACTOR,
    BONE_VOLUME_EXPONENT,
    BONE_VOLUME_PREFACTOR,
    C_ADULT,
    C_CONTROL,
    CA_BIRTH,
    CORTICAL_TURNOVER_APPORTIONMENT,
    CORTICAL_TURNOVER_BASELINE,
    CORTICAL_VOLUME_FRACTION,
    GROWTH_PARAMS,
    HAWKINS_K1,
    HAWKINS_V_B,
    MOLAR_MASS_CALCIUM,
    MOLAR_MASS_FLUORINE,
    OFLAHERTY_FEMALE,
    OFLAHERTY_MALE,
    TRABECULAR_TURNOVER_APPORTIONMENT,
    TRABECULAR_TURNOVER_BASELINE,
    TRABECULAR_VOLUME_FRACTION,
    HAWKINS_k2,
    HAWKINS_k3_ADULT,
    HAWKINS_k4_ADULT,
    HAWKINS_Ki_ADULT,
)


def test_hawkins_pet_constants() -> None:
    """Verify Hawkins [18F]Fluoride reference rate constants and extraction equation."""
    assert HAWKINS_K1 == pytest.approx(0.106)
    assert HAWKINS_k2 == pytest.approx(0.258)
    assert HAWKINS_k3_ADULT == pytest.approx(0.132)
    assert HAWKINS_k4_ADULT == pytest.approx(0.002)
    assert HAWKINS_V_B == pytest.approx(0.050)

    # Net clearance Ki formula check
    calculated_ki = (HAWKINS_K1 * HAWKINS_k3_ADULT) / (HAWKINS_k2 + HAWKINS_k3_ADULT)
    assert np.isclose(calculated_ki, HAWKINS_Ki_ADULT, atol=1e-3)
    assert np.isclose(HAWKINS_Ki_ADULT, 0.0359, atol=1e-4)


def test_skeletal_architecture_and_turnover_fractions() -> None:
    """Verify skeletal volume fractions and turnover apportionments."""
    # Volume fractions partition unity
    assert CORTICAL_VOLUME_FRACTION == pytest.approx(0.80)
    assert TRABECULAR_VOLUME_FRACTION == pytest.approx(0.20)
    assert CORTICAL_VOLUME_FRACTION + TRABECULAR_VOLUME_FRACTION == pytest.approx(1.0)

    # Turnover shares partition unity
    assert CORTICAL_TURNOVER_APPORTIONMENT == pytest.approx(0.35)
    assert TRABECULAR_TURNOVER_APPORTIONMENT == pytest.approx(0.65)
    assert CORTICAL_TURNOVER_APPORTIONMENT + TRABECULAR_TURNOVER_APPORTIONMENT == pytest.approx(1.0)

    # Whole-skeleton baseline is 10% per year
    assert ADULT_TURNOVER_BASELINE == pytest.approx(0.10)

    # Derived rates: Trabecular = (0.65 / 0.20) * 0.10 = 0.325 yr^-1
    assert TRABECULAR_TURNOVER_BASELINE == pytest.approx(0.325)

    # Cortical = (0.35 / 0.80) * 0.10 = 0.04375 yr^-1
    assert CORTICAL_TURNOVER_BASELINE == pytest.approx(0.04375)

    # Volume-weighted consistency verification: 0.20*TFBFR + 0.80*CFBFR == FBFR
    weighted_sum = (
        TRABECULAR_VOLUME_FRACTION * TRABECULAR_TURNOVER_BASELINE
        + CORTICAL_VOLUME_FRACTION * CORTICAL_TURNOVER_BASELINE
    )
    assert weighted_sum == pytest.approx(ADULT_TURNOVER_BASELINE)


def test_oflaherty_growth_constants() -> None:
    """Verify O'Flaherty 6-parameter human body weight model constants."""
    # Male parameters
    assert OFLAHERTY_MALE.w_birth == pytest.approx(3.5)
    assert OFLAHERTY_MALE.w_child == pytest.approx(23.0)
    assert OFLAHERTY_MALE.half == pytest.approx(3.0)
    assert OFLAHERTY_MALE.w_adult == pytest.approx(50.0)
    assert OFLAHERTY_MALE.kappa == pytest.approx(600.0)
    assert OFLAHERTY_MALE.lambda_ == pytest.approx(0.0095)
    assert OFLAHERTY_MALE.mature_weight == pytest.approx(76.5)

    # Female parameters
    assert OFLAHERTY_FEMALE.w_birth == pytest.approx(3.5)
    assert OFLAHERTY_FEMALE.w_child == pytest.approx(22.0)
    assert OFLAHERTY_FEMALE.half == pytest.approx(3.0)
    assert OFLAHERTY_FEMALE.w_adult == pytest.approx(34.0)
    assert OFLAHERTY_FEMALE.kappa == pytest.approx(600.0)
    assert OFLAHERTY_FEMALE.lambda_ == pytest.approx(0.0170)
    assert OFLAHERTY_FEMALE.mature_weight == pytest.approx(59.5)

    # Mapping
    assert GROWTH_PARAMS["M"] == OFLAHERTY_MALE
    assert GROWTH_PARAMS["F"] == OFLAHERTY_FEMALE


def test_chemical_and_physiological_constants() -> None:
    """Verify ash fraction, control concentrations, and atomic weights."""
    assert ASH_FRACTION == pytest.approx(0.56)
    assert C_CONTROL == pytest.approx(282.0)
    assert C_ADULT == pytest.approx(0.0030)
    assert CA_BIRTH == pytest.approx(0.028)
    assert MOLAR_MASS_FLUORINE == pytest.approx(18.9984)
    assert MOLAR_MASS_CALCIUM == pytest.approx(40.078)

    # Allometry coefficients
    assert BONE_VOLUME_PREFACTOR == pytest.approx(0.0168)
    assert BONE_VOLUME_EXPONENT == pytest.approx(1.188)
    assert BONE_MASS_PREFACTOR == pytest.approx(0.0290)
    assert BONE_MASS_EXPONENT == pytest.approx(1.21)
