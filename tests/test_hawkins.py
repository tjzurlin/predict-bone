"""Unit tests for Hawkins PET kinetics and bone growth bridge (HawkinsCalibrator).

Verifies:
1. Adult baseline anchoring (age 30 => r_form ~ 1.0, r_resorp ~ 1.0).
2. Infant remodeling surge (age 0.5 => r_form > 5.0).
3. PBKM fractional rates (FBFR, TFBFR, CFBFR) matching exact manual proportions.
4. Calcium flux mass balance (net accretion > 0 in children, ~0 in adults).
5. PET kinetics (k3, k4, Ki) with and without C_bone turnover suppression.
6. Vectorized inputs across ages and array handling.
7. Integration with fitted BiomarkerModel and TurnoverModel fixtures.
8. Modifier grid generation across lifespan and schema validation.
9. Error handling and input bounds validation.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from bone_predictor.biomarkers import BiomarkerModel
from bone_predictor.constants import (
    ADULT_TURNOVER_BASELINE,
    CORTICAL_TURNOVER_BASELINE,
    HAWKINS_K1,
    TRABECULAR_TURNOVER_BASELINE,
    HAWKINS_k2,
    HAWKINS_k3_ADULT,
    HAWKINS_k4_ADULT,
)
from bone_predictor.hawkins import HawkinsCalibrator, _analytical_relative_activity
from bone_predictor.turnover import TurnoverModel


# ==============================================================================
# 1. Adult Baseline Anchoring & Infant Remodeling Surge
# ==============================================================================
def test_analytical_relative_activity_adult_baseline() -> None:
    """Verify that healthy 30-year-old adults anchor at relative activity ~1.00."""
    r_form_f = _analytical_relative_activity(30.0, sex="F", marker="alp")
    r_resorp_f = _analytical_relative_activity(30.0, sex="F", marker="hp")
    r_form_m = _analytical_relative_activity(30.0, sex="M", marker="alp")
    r_resorp_m = _analytical_relative_activity(30.0, sex="M", marker="hp")

    assert np.isclose(r_form_f, 1.00, atol=0.01)
    assert np.isclose(r_resorp_f, 1.00, atol=0.01)
    assert np.isclose(r_form_m, 1.00, atol=0.01)
    assert np.isclose(r_resorp_m, 1.00, atol=0.01)


def test_analytical_relative_activity_infant_surge() -> None:
    """Verify that 6-month infants experience an intense growth surge (> 5.0x)."""
    r_form_f = _analytical_relative_activity(0.5, sex="F", marker="alp")
    r_resorp_f = _analytical_relative_activity(0.5, sex="F", marker="hp")
    r_form_m = _analytical_relative_activity(0.5, sex="M", marker="alp")
    r_resorp_m = _analytical_relative_activity(0.5, sex="M", marker="hp")

    assert r_form_f > 5.0
    assert r_resorp_f > 5.0
    assert r_form_m > 5.0
    assert r_resorp_m > 5.0


def test_calibrator_predict_relative_activity_scalar() -> None:
    """Test predict_relative_activity returns scalar floats for scalar age input."""
    calibrator = HawkinsCalibrator()
    res = calibrator.predict_relative_activity(30.0, sex="M")

    assert isinstance(res, dict)
    assert "r_form" in res and "r_resorp" in res
    assert isinstance(res["r_form"], float)
    assert isinstance(res["r_resorp"], float)
    assert np.isclose(res["r_form"], 1.0, atol=0.01)
    assert np.isclose(res["r_resorp"], 1.0, atol=0.01)


# ==============================================================================
# 2. PBKM Fractional Rates
# ==============================================================================
def test_fractional_rates_pbkm_proportions() -> None:
    """Verify FBFR, TFBFR, and CFBFR match PBKM Manual baseline values & proportions.

    Whole-skeleton: 10%/yr (0.10 yr^-1)
    Trabecular: (0.65 / 0.20) * 0.10 = 32.5%/yr (0.325 yr^-1)
    Cortical: (0.35 / 0.80) * 0.10 = 4.375%/yr (0.04375 yr^-1)
    """
    calibrator = HawkinsCalibrator()
    rates = calibrator.fractional_rates(30.0, sex="M")

    assert np.isclose(rates["FBFR"], ADULT_TURNOVER_BASELINE, atol=0.001)
    assert np.isclose(rates["TFBFR"], TRABECULAR_TURNOVER_BASELINE, atol=0.001)
    assert np.isclose(rates["CFBFR"], CORTICAL_TURNOVER_BASELINE, atol=0.0001)

    # Check exact proportions
    assert np.isclose(rates["TFBFR"] / rates["FBFR"], 0.65 / 0.20)
    assert np.isclose(rates["CFBFR"] / rates["FBFR"], 0.35 / 0.80)

    # Lowercase alias checks
    assert rates["fbfr"] == rates["FBFR"]
    assert rates["tfbfr"] == rates["TFBFR"]
    assert rates["cfbfr"] == rates["CFBFR"]


def test_fractional_rates_infancy_scaling() -> None:
    """Verify that infant fractional formation rates scale with infant r_form."""
    calibrator = HawkinsCalibrator()
    act = calibrator.predict_relative_activity(0.5, sex="F")
    rates = calibrator.fractional_rates(0.5, sex="F")

    expected_fbfr = 0.10 * act["r_form"]
    expected_tfbfr = 0.325 * act["r_form"]
    expected_cfbfr = 0.04375 * act["r_form"]

    assert np.isclose(rates["FBFR"], expected_fbfr)
    assert np.isclose(rates["TFBFR"], expected_tfbfr)
    assert np.isclose(rates["CFBFR"], expected_cfbfr)
    assert rates["FBFR"] > 0.50  # > 50%/year whole skeleton turnover in infancy


# ==============================================================================
# 3. Calcium Flux & Mass Balance
# ==============================================================================
def test_calcium_flux_mass_balance_adult() -> None:
    """Verify that in healthy adults, net calcium accretion is approximately zero."""
    calibrator = HawkinsCalibrator()
    fluxes_m = calibrator.calcium_flux(30.0, sex="M")
    fluxes_f = calibrator.calcium_flux(30.0, sex="F")

    # In equilibrium, f_form ~ f_resorp => net accretion ~ 0
    assert np.isclose(fluxes_m["net_accretion"], 0.0, atol=0.001)
    assert np.isclose(fluxes_f["net_accretion"], 0.0, atol=0.001)
    assert np.isclose(fluxes_m["f_form"] - fluxes_m["f_resorp"], fluxes_m["net_accretion"])
    assert np.isclose(fluxes_f["f_form"] - fluxes_f["f_resorp"], fluxes_f["net_accretion"])

    assert fluxes_m["f_form"] > 0.1
    assert fluxes_f["f_form"] > 0.1


def test_calcium_flux_positive_in_children() -> None:
    """Verify that net calcium accretion is strictly positive across growing children."""
    calibrator = HawkinsCalibrator()
    for child_age in [0.5, 2.0, 7.0, 11.0, 13.0]:
        fluxes = calibrator.calcium_flux(child_age, sex="F")
        assert fluxes["net_accretion"] > 0.0, (
            f"Expected positive accretion at age {child_age}, got {fluxes['net_accretion']}"
        )
        assert fluxes["f_form"] > fluxes["f_resorp"]


# ==============================================================================
# 4. Hawkins Fluoride PET Microparameters
# ==============================================================================
def test_pet_kinetics_adult_baseline_unexposed() -> None:
    """Verify PET microparameters at adult baseline without fluoride suppression."""
    calibrator = HawkinsCalibrator()
    pet = calibrator.predict_pet_kinetics(30.0, sex="M", c_bone=None)

    assert np.isclose(pet["k3"], HAWKINS_k3_ADULT, atol=0.001)
    assert np.isclose(pet["k4"], HAWKINS_k4_ADULT, atol=0.0001)
    expected_ki = (HAWKINS_K1 * HAWKINS_k3_ADULT) / (HAWKINS_k2 + HAWKINS_k3_ADULT)
    assert np.isclose(pet["Ki"], expected_ki, atol=0.0005)
    assert np.isclose(pet["Ki"], 0.0359, atol=0.0005)
    assert np.isclose(pet["r_to"], 1.0)


def test_pet_kinetics_with_fluoride_suppression() -> None:
    """Verify that elevated bone fluoride suppresses k3 and Ki without altering k4."""
    calibrator = HawkinsCalibrator()
    # High bone fluoride: 10,000 mg/kg ash
    pet_suppressed = calibrator.predict_pet_kinetics(30.0, sex="M", c_bone=10000.0)
    pet_baseline = calibrator.predict_pet_kinetics(30.0, sex="M", c_bone=282.0)

    # Suppression factor r_to should be < 1.0 (typically ~0.33 to 0.40)
    assert pet_suppressed["r_to"] < 0.60
    assert pet_suppressed["k3"] < pet_baseline["k3"]
    assert pet_suppressed["Ki"] < pet_baseline["Ki"]

    # k4 is unsuppressed by fluoride
    assert np.isclose(pet_suppressed["k4"], pet_baseline["k4"])


# ==============================================================================
# 5. Unified predict_modifiers
# ==============================================================================
def test_predict_modifiers_scalar() -> None:
    """Verify predict_modifiers returns complete dictionary with scalar values."""
    calibrator = HawkinsCalibrator()
    mods = calibrator.predict_modifiers(30.0, sex="F", c_bone=282.0)

    expected_keys = {
        "age",
        "sex",
        "r_form",
        "r_resorp",
        "r_to",
        "FBFR",
        "TFBFR",
        "CFBFR",
        "f_form",
        "f_resorp",
        "net_accretion",
        "k3",
        "k4",
        "Ki",
    }
    assert expected_keys.issubset(mods.keys())
    assert isinstance(mods["age"], float)
    assert mods["sex"] == "F"
    assert isinstance(mods["r_form"], float)
    assert isinstance(mods["Ki"], float)


def test_predict_modifiers_vectorized() -> None:
    """Verify predict_modifiers handles 1D numpy array of ages seamlessly."""
    calibrator = HawkinsCalibrator()
    ages = np.array([0.5, 5.0, 15.0, 30.0, 60.0])
    mods = calibrator.predict_modifiers(ages, sex="M", c_bone=282.0)

    assert isinstance(mods["r_form"], np.ndarray)
    assert len(mods["r_form"]) == len(ages)
    assert len(mods["FBFR"]) == len(ages)
    assert len(mods["k3"]) == len(ages)
    assert len(mods["net_accretion"]) == len(ages)

    # Compare 4th point (age 30) to scalar evaluation
    scalar_mods = calibrator.predict_modifiers(30.0, sex="M", c_bone=282.0)
    assert np.isclose(mods["r_form"][3], scalar_mods["r_form"])
    assert np.isclose(mods["Ki"][3], scalar_mods["Ki"])


# ==============================================================================
# 6. Integration with Fitted Models
# ==============================================================================
def test_calibrator_with_mock_biomarker_and_turnover_models(
    mock_biomarker_idata: Any,
    mock_idata: Any,
    sample_stepan_df: pd.DataFrame,
    sample_boivin_df: pd.DataFrame,
) -> None:
    """Verify HawkinsCalibrator seamlessly uses fitted PyMC models when provided."""
    bio_model = BiomarkerModel(data=sample_stepan_df)
    bio_model.idata = mock_biomarker_idata

    turnover_model = TurnoverModel(data=sample_boivin_df)
    turnover_model.idata = mock_idata

    calibrator = HawkinsCalibrator(biomarker_model=bio_model, turnover_model=turnover_model)

    mods = calibrator.predict_modifiers(30.0, sex="F", c_bone=3500.0)
    assert mods["r_form"] > 0.0
    assert mods["r_resorp"] > 0.0
    assert 0.0 < mods["r_to"] < 1.0  # Suppression at 3500 mg/kg
    assert mods["Ki"] > 0.0


# ==============================================================================
# 7. Constants Overrides
# ==============================================================================
def test_constants_override() -> None:
    """Verify user constants_override alters PET constants."""
    overrides = {
        "HAWKINS_k3_ADULT": 0.200,
        "HAWKINS_K1": 0.150,
        "ADULT_TURNOVER_BASELINE": 0.12,
    }
    calibrator = HawkinsCalibrator(constants_override=overrides)

    assert calibrator.k3_adult == 0.200
    assert calibrator.k1 == 0.150
    assert calibrator.adult_turnover_baseline == 0.12

    mods = calibrator.predict_modifiers(30.0, sex="M")
    assert np.isclose(mods["k3"], 0.200, atol=0.005)
    assert np.isclose(mods["FBFR"], 0.12, atol=0.005)


# ==============================================================================
# 8. modifier_grid Generation & Schema Validation
# ==============================================================================
def test_modifier_grid_both_sexes() -> None:
    """Verify modifier_grid generates valid table for both sexes."""
    calibrator = HawkinsCalibrator()
    df = calibrator.modifier_grid(age_min=0.0, age_max=80.0, step=10.0, sex="both")

    expected_cols = [
        "age",
        "sex",
        "body_weight",
        "r_form",
        "r_resorp",
        "r_to",
        "FBFR",
        "TFBFR",
        "CFBFR",
        "f_form",
        "f_resorp",
        "net_accretion",
        "k3",
        "k4",
        "Ki",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in modifier_grid"

    # Number of age points: 0, 10, 20, 30, 40, 50, 60, 70, 80 => 9 points * 2 sexes = 18 rows
    assert len(df) == 18
    assert set(df["sex"].unique()) == {"F", "M"}
    assert df["age"].min() == 0.01  # Clamped minimum
    assert df["age"].max() == 80.0
    assert not df.isna().any().any()


def test_modifier_grid_single_sex() -> None:
    """Verify modifier_grid generates table for single sex."""
    calibrator = HawkinsCalibrator()
    df = calibrator.modifier_grid(age_min=1.0, age_max=50.0, step=5.0, sex="F")

    assert len(df) == 11
    assert (df["sex"] == "F").all()
    assert not df.isna().any().any()


# ==============================================================================
# 9. Error Handling & Bounds Checking
# ==============================================================================
def test_calibrator_invalid_age_raises() -> None:
    """Verify non-positive or non-finite ages raise ValueError."""
    calibrator = HawkinsCalibrator()
    with pytest.raises(ValueError, match="strictly positive"):
        calibrator.predict_modifiers(0.0, sex="M")
    with pytest.raises(ValueError, match="strictly positive"):
        calibrator.predict_modifiers(-5.0, sex="M")
    with pytest.raises(ValueError, match="finite"):
        calibrator.predict_modifiers(np.nan, sex="M")


def test_calibrator_invalid_sex_raises() -> None:
    """Verify invalid sex parameter raises ValueError."""
    calibrator = HawkinsCalibrator()
    with pytest.raises(ValueError, match="Unknown sex indicator"):
        calibrator.predict_modifiers(30.0, sex="InvalidSex")


def test_modifier_grid_invalid_bounds_raises() -> None:
    """Verify invalid step or age range in modifier_grid raises ValueError."""
    calibrator = HawkinsCalibrator()
    with pytest.raises(ValueError, match="step must be strictly positive"):
        calibrator.modifier_grid(age_min=0.0, age_max=80.0, step=0.0)
    with pytest.raises(ValueError, match="age_max must be strictly greater"):
        calibrator.modifier_grid(age_min=50.0, age_max=20.0, step=1.0)
