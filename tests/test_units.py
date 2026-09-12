"""Unit tests for the bone concentration unit conversion module."""

from typing import Any

import numpy as np
import pytest

from bone_predictor.constants import ASH_FRACTION, MOLAR_MASS_FLUORINE
from bone_predictor.units import (
    ash_to_molar,
    ash_to_wet,
    convert_concentration,
    molar_to_ash,
    molar_to_wet,
    wet_to_ash,
    wet_to_molar,
)


def test_ash_wet_roundtrip_scalar() -> None:
    """Test scalar conversions between ash and wet bone concentrations."""
    c_ash = 1000.0
    c_wet = ash_to_wet(c_ash)
    assert c_wet == pytest.approx(560.0)
    recovered_ash = wet_to_ash(c_wet)
    assert recovered_ash == pytest.approx(c_ash)


def test_wet_molar_roundtrip_scalar() -> None:
    """Test scalar conversions between wet bone (mg/kg) and molar (mmol/kg)."""
    c_wet = 18.9984
    c_molar = wet_to_molar(c_wet)
    assert c_molar == pytest.approx(1.0)
    recovered_wet = molar_to_wet(c_molar)
    assert recovered_wet == pytest.approx(c_wet)


def test_ash_molar_roundtrip() -> None:
    """Test direct conversions between ash (mg/kg) and molar (mmol/kg)."""
    c_ash = 3500.0
    c_molar = ash_to_molar(c_ash)
    expected_molar = (c_ash * ASH_FRACTION) / MOLAR_MASS_FLUORINE
    assert c_molar == pytest.approx(expected_molar)
    recovered_ash = molar_to_ash(c_molar)
    assert recovered_ash == pytest.approx(c_ash)


def test_vectorized_conversions() -> None:
    """Test conversions on 1D numpy arrays."""
    c_ash_arr = np.array([0.0, 282.0, 1000.0, 5000.0, 12000.0])
    c_wet_arr = ash_to_wet(c_ash_arr)
    assert isinstance(c_wet_arr, np.ndarray)
    assert len(c_wet_arr) == len(c_ash_arr)
    assert np.allclose(c_wet_arr, c_ash_arr * 0.56)

    c_molar_arr = wet_to_molar(c_wet_arr)
    recovered_wet = molar_to_wet(c_molar_arr)
    assert np.allclose(recovered_wet, c_wet_arr)


@pytest.mark.parametrize(
    ("from_u", "to_u"),
    [
        ("mg/kg_ash", "ppm_wet"),
        ("ash", "wet"),
        ("wet", "molar"),
        ("molar", "ash"),
        ("ppm_ash", "mmol/kg"),
    ],
)
def test_convert_concentration_dispatch(from_u: str, to_u: str) -> None:
    """Test generic convert_concentration dispatcher with various aliases."""
    val = 500.0
    res = convert_concentration(val, from_u, to_u)
    assert res > 0
    # Identity conversion when units match
    assert convert_concentration(val, from_u, from_u) == pytest.approx(val)


def test_unit_conversion_errors() -> None:
    """Verify appropriate exceptions for invalid inputs and units."""
    # Negative concentration
    with pytest.raises(ValueError, match="non-negative"):
        ash_to_wet(-10.0)
    with pytest.raises(ValueError, match="non-negative"):
        wet_to_ash(-5.0)
    with pytest.raises(ValueError, match="non-negative"):
        wet_to_molar(np.array([10.0, -1.0]))

    # Invalid ash fraction
    with pytest.raises(ValueError, match="Ash fraction"):
        ash_to_wet(100.0, ash_fraction=0.0)
    with pytest.raises(ValueError, match="Ash fraction"):
        ash_to_wet(100.0, ash_fraction=1.5)

    # Invalid molar mass
    with pytest.raises(ValueError, match="Molar mass"):
        wet_to_molar(100.0, molar_mass=-2.0)

    # Unrecognized unit string
    with pytest.raises(ValueError, match="Unrecognized source unit"):
        convert_concentration(100.0, "unknown_unit", "wet")
    with pytest.raises(ValueError, match="Unrecognized target unit"):
        convert_concentration(100.0, "ash", "invalid_unit")


@pytest.mark.parametrize(
    ("func", "input_list", "expected_factor"),
    [
        (ash_to_wet, [0.0, 100.0, 500.0, 12000.0], ASH_FRACTION),
        (wet_to_ash, [0.0, 56.0, 280.0, 6720.0], 1.0 / ASH_FRACTION),
        (wet_to_molar, [0.0, 18.9984, 37.9968], 1.0 / MOLAR_MASS_FLUORINE),
        (molar_to_wet, [0.0, 1.0, 2.0], MOLAR_MASS_FLUORINE),
        (ash_to_molar, [0.0, 1000.0, 3500.0], ASH_FRACTION / MOLAR_MASS_FLUORINE),
        (molar_to_ash, [0.0, 1.0, 2.0], MOLAR_MASS_FLUORINE / ASH_FRACTION),
    ],
)
def test_units_conversion_list_inputs(
    func: Any, input_list: list[float], expected_factor: float
) -> None:
    """Verify all unit conversion functions accept Python lists and return ndarrays."""
    res = func(input_list)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(input_list)
    expected = np.array(input_list) * expected_factor
    assert np.allclose(res, expected)


@pytest.mark.parametrize(
    ("from_u", "to_u", "input_list"),
    [
        ("ash", "wet", [100.0, 200.0]),
        ("wet", "ash", [56.0, 112.0]),
        ("wet", "molar", [18.9984, 37.9968]),
        ("molar", "wet", [1.0, 2.0]),
        ("ash", "molar", [1000.0, 2000.0]),
        ("molar", "ash", [1.0, 2.0]),
        ("ash", "ash", [100.0, 200.0]),  # identity branch
        ("wet", "wet", [50.0, 100.0]),
        ("ppm_ash", "mmol/kg", [1000.0, 3500.0]),
    ],
)
def test_convert_concentration_list_inputs(from_u: str, to_u: str, input_list: list[float]) -> None:
    """Verify convert_concentration handles Python lists across aliases and identity."""
    res = convert_concentration(input_list, from_u, to_u)
    assert isinstance(res, np.ndarray)
    assert len(res) == len(input_list)
    assert np.all(res >= 0)


@pytest.mark.parametrize(
    "func",
    [ash_to_wet, wet_to_ash, wet_to_molar, molar_to_wet, ash_to_molar, molar_to_ash],
)
def test_units_conversion_list_negative_raises(func: Any) -> None:
    """Verify conversion functions raise ValueError on Python lists containing negative values."""
    with pytest.raises(ValueError, match="non-negative"):
        func([10.0, -5.0, 20.0])
