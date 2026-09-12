"""Unit tests for the skeletal physiology and allometry module."""

from typing import Any

import numpy as np
import pytest

from bone_predictor.physiology import (
    body_weight,
    body_weight_derivative,
    bone_density,
    bone_mass,
    bone_volume,
    bone_volume_growth_rate,
    compartmental_bone_volumes,
    normalize_sex,
    skeletal_remodeling_balance,
)


def test_normalize_sex() -> None:
    """Verify sex normalization and validation."""
    assert normalize_sex("M") == "M"
    assert normalize_sex("Male") == "M"
    assert normalize_sex("m") == "M"
    assert normalize_sex("1") == "M"
    assert normalize_sex(1) == "M"

    assert normalize_sex("F") == "F"
    assert normalize_sex("Female") == "F"
    assert normalize_sex("f") == "F"
    assert normalize_sex("0") == "F"
    assert normalize_sex(0) == "F"

    # Booleans and invalid strings rejected
    with pytest.raises(ValueError, match="Boolean"):
        normalize_sex(True)
    with pytest.raises(ValueError, match="Boolean"):
        normalize_sex(False)
    with pytest.raises(ValueError, match="Unknown sex"):
        normalize_sex("unknown")


def test_body_weight_milestones() -> None:
    """Verify deterministic body weights across key life milestones."""
    # 1. Birth (age = 0)
    # Male: 3.5 + 50 / 601 = 3.58319... kg
    bw_m_birth = body_weight(0.0, "M")
    assert bw_m_birth == pytest.approx(3.5 + 50.0 / 601.0, rel=1e-3)
    assert 3.55 < bw_m_birth < 3.65

    # Female: 3.5 + 34 / 601 = 3.55657... kg
    bw_f_birth = body_weight(0.0, "F")
    assert bw_f_birth == pytest.approx(3.5 + 34.0 / 601.0, rel=1e-3)
    assert 3.50 < bw_f_birth < 3.60

    # 2. Puberty crossover timing:
    # Female growth spurt starts earlier (~10-12 yr)
    bw_m_11 = body_weight(11.0, "M")
    bw_f_11 = body_weight(11.0, "F")
    assert bw_m_11 > 25.0
    assert bw_f_11 > 25.0

    # 3. Mature adult weights (age >= 30 yr)
    bw_m_adult = body_weight(30.0, "M")
    bw_f_adult = body_weight(30.0, "F")
    assert bw_m_adult == pytest.approx(74.4, abs=0.5)
    assert bw_f_adult == pytest.approx(57.5, abs=0.5)

    # 4. Asymptotic convergence (age -> large, e.g. 5000 yr)
    bw_m_asymptote = body_weight(5000.0, "M")
    bw_f_asymptote = body_weight(5000.0, "F")
    assert bw_m_asymptote == pytest.approx(76.5, abs=0.1)
    assert bw_f_asymptote == pytest.approx(59.5, abs=0.1)


def test_body_weight_array_vectorization() -> None:
    """Verify vectorized evaluation of body weight."""
    ages = np.array([0.0, 1.0, 5.0, 15.0, 25.0, 50.0])
    bw_arr = body_weight(ages, "M")
    assert isinstance(bw_arr, np.ndarray)
    assert len(bw_arr) == len(ages)
    # Monotonically increasing growth
    assert np.all(np.diff(bw_arr) > 0)


def test_body_weight_derivative() -> None:
    """Verify analytical growth rate derivative behavior."""
    # Derivative at birth is positive (infant growth rate)
    dbw_birth = body_weight_derivative(0.0, "M")
    assert dbw_birth > 1.0  # > 1 kg/yr in infancy

    # Derivative at mature adult (age 40) is very small (< 0.05 kg/yr, residual hyperbolic tail)
    dbw_adult = body_weight_derivative(40.0, "M")
    assert dbw_adult < 0.05
    assert dbw_adult > 0.0

    # Growth rate is strictly positive during growth
    growth_ages = np.linspace(0.1, 20.0, 50)
    dbw_growth = body_weight_derivative(growth_ages, "M")
    assert np.all(dbw_growth > 0)


def test_body_weight_errors() -> None:
    """Verify input validation for body weight functions."""
    with pytest.raises(ValueError, match="non-negative"):
        body_weight(-1.0, "M")
    with pytest.raises(ValueError, match="finite"):
        body_weight(np.nan, "M")
    with pytest.raises(ValueError, match="finite"):
        body_weight(np.inf, "M")


def test_bone_volume_and_mass_allometry() -> None:
    """Verify allometric power-law scaling of bone volume and mass."""
    # Newborn (~3.5 kg)
    v_birth = bone_volume(3.5)
    w_birth = bone_mass(3.5)
    assert v_birth == pytest.approx(0.0168 * (3.5**1.188), rel=1e-4)
    assert w_birth == pytest.approx(0.0290 * (3.5**1.21), rel=1e-4)

    # Adult male (76.5 kg)
    v_adult_m = bone_volume(76.5)
    w_adult_m = bone_mass(76.5)
    # PBKM manual expected ~2.89 L volume and ~5.48 kg mass
    assert 2.80 < v_adult_m < 3.00
    assert 5.30 < w_adult_m < 5.65

    # Density WBONE / VBONE in physiological range [1.7, 1.95] kg/L
    dens_birth = bone_density(3.5)
    dens_adult = bone_density(76.5)
    assert 1.70 < dens_birth < 1.85
    assert 1.80 < dens_adult < 1.95

    # Invalid non-positive body weights
    with pytest.raises(ValueError, match="strictly positive"):
        bone_volume(0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        bone_mass(-5.0)


def test_compartmental_bone_volumes() -> None:
    """Verify 80% cortical and 20% trabecular volume partition."""
    v_total = 2.5  # liters
    cv, tv = compartmental_bone_volumes(v_total)
    assert cv == pytest.approx(0.80 * 2.5)
    assert tv == pytest.approx(0.20 * 2.5)
    assert cv + tv == pytest.approx(v_total)

    with pytest.raises(ValueError, match="strictly positive"):
        compartmental_bone_volumes(0.0)


def test_bone_volume_growth_rate() -> None:
    """Verify dynamic bone volume expansion rate d(VBONE)/dt."""
    dv_birth = bone_volume_growth_rate(0.5, "M")
    assert dv_birth > 0  # Rapid skeletal expansion in infancy

    dv_adult = bone_volume_growth_rate(35.0, "M")
    assert dv_adult < 0.005  # Approaches zero after skeletal maturity


def test_skeletal_remodeling_balance() -> None:
    """Verify skeletal volume conservation and remodeling balance."""
    # Adult steady-state (age 30, r_form = 1.0)
    bal_adult = skeletal_remodeling_balance(30.0, "M", r_form=1.0)
    assert np.isclose(bal_adult["fbfr"], 0.10, atol=1e-3)
    assert np.isclose(bal_adult["tfbfr"], 0.325, atol=1e-3)
    assert np.isclose(bal_adult["cfbfr"], 0.04375, atol=1e-4)

    # In adult, volume expansion is very small (< 0.01 L/yr), so BFR approx BRR
    assert bal_adult["dv_dt"] < 0.01
    assert np.isclose(bal_adult["bfr"], bal_adult["brr"], atol=0.01)

    # Compartmental consistency
    assert bal_adult["tbfr"] + bal_adult["cbfr"] == pytest.approx(bal_adult["bfr"])
    assert bal_adult["tbrr"] + bal_adult["cbrr"] == pytest.approx(bal_adult["brr"])

    # Growing child (age 5, r_form = 2.0)
    bal_child = skeletal_remodeling_balance(5.0, "F", r_form=2.0)
    assert bal_child["dv_dt"] > 0
    # Net bone formation exceeds resorption during growth: BFR - BRR == dV/dt
    assert np.isclose(bal_child["bfr"] - bal_child["brr"], bal_child["dv_dt"], atol=1e-4)


@pytest.mark.parametrize(
    "v_list",
    [
        [2.5],
        [1.0, 2.0, 3.0],
        [0.0765, 0.5, 1.5, 2.89],
    ],
)
def test_compartmental_bone_volumes_list(v_list: list[float]) -> None:
    """Verify compartmental_bone_volumes accepts Python lists and returns ndarray tuples."""
    cv, tv = compartmental_bone_volumes(v_list)
    assert isinstance(cv, np.ndarray)
    assert isinstance(tv, np.ndarray)
    v_arr = np.array(v_list)
    assert np.allclose(cv, 0.80 * v_arr)
    assert np.allclose(tv, 0.20 * v_arr)
    assert np.allclose(cv + tv, v_arr)


@pytest.mark.parametrize(
    "func",
    [bone_volume, bone_mass, bone_density],
)
@pytest.mark.parametrize(
    "invalid_input",
    [
        np.nan,
        np.inf,
        -np.inf,
        [70.0, np.nan],
        [70.0, np.inf],
        np.array([70.0, np.nan]),
        np.array([70.0, np.inf]),
    ],
)
def test_allometric_functions_non_finite(func: Any, invalid_input: Any) -> None:
    """Verify bone_volume, bone_mass, and bone_density reject non-finite inputs."""
    with pytest.raises(ValueError, match="finite"):
        func(invalid_input)


@pytest.mark.parametrize(
    "invalid_input",
    [
        np.nan,
        np.inf,
        -np.inf,
        [2.5, np.nan],
        [2.5, np.inf],
        np.array([2.5, np.nan]),
        np.array([2.5, np.inf]),
    ],
)
def test_compartmental_bone_volumes_non_finite(invalid_input: Any) -> None:
    """Verify compartmental_bone_volumes rejects non-finite inputs."""
    with pytest.raises(ValueError, match="finite"):
        compartmental_bone_volumes(invalid_input)


@pytest.mark.parametrize(
    "r_form",
    [
        np.array([0.5, 1.0, 1.5, 2.0]),  # posterior MCMC array draws
        [0.5, 1.0, 1.5, 2.0],  # python list
        np.array([1.0]),  # 1D single-element array
        [1.0],  # 1-element list
    ],
)
def test_skeletal_remodeling_balance_scalar_age_array_or_list_r_form(r_form: Any) -> None:
    """Verify skeletal_remodeling_balance handles scalar age with array or list r_form."""
    bal = skeletal_remodeling_balance(30.0, "M", r_form=r_form)
    n = len(r_form)

    # formation rate scales with r_form
    expected_fbfr = 0.10 * np.array(r_form)
    assert isinstance(bal["fbfr"], np.ndarray)
    assert len(bal["fbfr"]) == n
    assert np.allclose(bal["fbfr"], expected_fbfr)

    # formation flux scales with vb and r_form
    assert isinstance(bal["bfr"], np.ndarray)
    assert len(bal["bfr"]) == n
    assert np.allclose(bal["bfr"], expected_fbfr * bal["v_bone"])

    # compartmental fluxes partition 65% trabecular / 35% cortical
    assert isinstance(bal["tbfr"], np.ndarray)
    assert isinstance(bal["cbfr"], np.ndarray)
    assert np.allclose(bal["tbfr"] + bal["cbfr"], bal["bfr"])
    assert np.allclose(bal["tbrr"] + bal["cbrr"], bal["brr"])


@pytest.mark.parametrize(
    "neg_r",
    [
        -1.0,
        -0.001,
        [-0.5, 1.0],
        np.array([1.0, -0.2]),
    ],
)
def test_skeletal_remodeling_balance_negative_r_form_raises(neg_r: Any) -> None:
    """Verify negative r_form values raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        skeletal_remodeling_balance(30.0, "M", r_form=neg_r)


@pytest.mark.parametrize(
    "non_finite_r",
    [
        np.nan,
        np.inf,
        -np.inf,
        [1.0, np.nan],
        np.array([1.0, np.inf]),
    ],
)
def test_skeletal_remodeling_balance_non_finite_r_form_raises(non_finite_r: Any) -> None:
    """Verify non-finite r_form values raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        skeletal_remodeling_balance(30.0, "M", r_form=non_finite_r)
