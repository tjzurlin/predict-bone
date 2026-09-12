"""Tests for the Bone Turnover Suppression Model (TurnoverModel / r_TO).

Includes fast sub-second unit tests using mock_idata and synthetic fixtures,
and an end-to-end MCMC convergence test marked @pytest.mark.slow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import arviz_base as azb
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from bone_predictor.constants import C_CONTROL
from bone_predictor.data import load_boivin_data
from bone_predictor.turnover import TurnoverModel


# ==============================================================================
# Fast Unit Tests: Initialization & Formulation Switching
# ==============================================================================
def test_turnover_init_default() -> None:
    """Test default initialization has 4p_hill formulation and standard C_CONTROL."""
    model = TurnoverModel()
    assert model.formulation == "4p_hill"
    assert np.isclose(model.c_control, C_CONTROL)
    assert model.model is None
    assert model.idata is None


@pytest.mark.parametrize("formulation", ["4p_hill", "hill_2p", "log_linear", "hormesis"])
def test_turnover_init_supported_formulations(formulation: str) -> None:
    """Test initialization across all supported built-in formulations."""
    model = TurnoverModel(formulation=formulation)  # type: ignore[arg-type]
    assert model.formulation == formulation


def test_turnover_init_invalid_formulation() -> None:
    """Test that an unknown formulation name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown formulation 'invalid_model'"):
        TurnoverModel(formulation="invalid_model")  # type: ignore[arg-type]


def test_turnover_custom_build_fn(sample_boivin_df: pd.DataFrame) -> None:
    """Test initializing with a custom build_fn switches formulation to 'custom'."""

    def my_builder(df: pd.DataFrame) -> pm.Model:
        with pm.Model() as m:
            pm.Data("c", df["c_bone"].to_numpy())
            a = pm.Normal("a", mu=0.08, sigma=0.01)
            pm.Normal("obs", mu=a, sigma=0.01, observed=df["cn_bfr"].to_numpy())
        return m

    model = TurnoverModel(build_fn=my_builder, data=sample_boivin_df)
    assert model.formulation == "custom"
    assert model.build_fn is my_builder

    pm_model = model.build()
    assert isinstance(pm_model, pm.Model)
    assert "a" in pm_model.named_vars


def test_turnover_custom_missing_fn_raises() -> None:
    """Test that formulation='custom' without a build_fn raises ValueError."""
    with pytest.raises(ValueError, match=r"build_fn.*must be provided"):
        TurnoverModel(formulation="custom", build_fn=None)


# ==============================================================================
# Fast Unit Tests: Data Validation
# ==============================================================================
def test_turnover_data_validation_valid(sample_boivin_df: pd.DataFrame) -> None:
    """Test valid DataFrame is accepted and sorted by concentration."""
    model = TurnoverModel(data=sample_boivin_df)
    assert model.data is not None
    assert len(model.data) == len(sample_boivin_df)
    assert list(model.data["c_bone"]) == sorted(model.data["c_bone"])


def test_turnover_data_validation_missing_columns() -> None:
    """Test missing required columns raises ValueError."""
    bad_df = pd.DataFrame({"some_col": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        TurnoverModel(data=bad_df)


def test_turnover_data_validation_empty_df() -> None:
    """Test empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame(columns=["c_bone", "cn_bfr"])
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        TurnoverModel(data=empty_df)


def test_turnover_data_validation_non_positive() -> None:
    """Test non-positive concentrations or rates raise ValueError."""
    bad_c = pd.DataFrame({"c_bone": [-10.0, 200.0], "cn_bfr": [0.05, 0.05]})
    with pytest.raises(ValueError, match="strictly positive"):
        TurnoverModel(data=bad_c)

    bad_bfr = pd.DataFrame({"c_bone": [100.0, 200.0], "cn_bfr": [0.0, 0.05]})
    with pytest.raises(ValueError, match="strictly positive"):
        TurnoverModel(data=bad_bfr)


def test_turnover_data_validation_invalid_type() -> None:
    """Test passing a non-DataFrame raises TypeError."""
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        TurnoverModel(data="invalid_path_string")  # type: ignore[arg-type]


# ==============================================================================
# Fast Unit Tests: Model Building & Graph Structure
# ==============================================================================
@pytest.mark.parametrize(
    ("formulation", "expected_vars"),
    [
        (
            "4p_hill",
            [
                "bfr0",
                "r_floor",
                "eta_km",
                "n_hill",
                "sigma_log",
                "km",
                "bfr_floor",
                "mu",
                "bfr_obs",
            ],
        ),
        (
            "hill_2p",
            ["bfr0", "eta_km", "n_hill", "sigma_log", "km", "bfr_floor", "mu", "bfr_obs"],
        ),
        ("log_linear", ["a", "b", "sigma_log", "mu", "bfr_obs"]),
        (
            "hormesis",
            [
                "bfr0",
                "r_floor",
                "eta_km",
                "n_hill",
                "f",
                "sigma_log",
                "km",
                "bfr_floor",
                "mu",
                "bfr_obs",
            ],
        ),
    ],
)
def test_turnover_build_variables(
    sample_boivin_df: pd.DataFrame, formulation: str, expected_vars: list[str]
) -> None:
    """Test each formulation graph registers expected deterministic and random variables."""
    model = TurnoverModel(formulation=formulation, data=sample_boivin_df)  # type: ignore[arg-type]
    pm_model = model.build()
    assert isinstance(pm_model, pm.Model)
    assert "obs_id" in pm_model.coords
    for var in expected_vars:
        assert var in pm_model.named_vars, f"Expected '{var}' in {formulation} model variables"


def test_turnover_sample_prior_predictive(sample_boivin_df: pd.DataFrame) -> None:
    """Test prior predictive sampling runs cleanly and returns InferenceData."""
    model = TurnoverModel(data=sample_boivin_df)
    prior = model.sample_prior_predictive(draws=10, random_seed=42)
    assert prior is not None
    assert hasattr(prior, "prior")
    assert "bfr0" in prior.prior


# ==============================================================================
# Fast Unit Tests: r_to Duck-Typing & Calculation (using mock_idata)
# ==============================================================================
def test_turnover_r_to_unfitted_raises() -> None:
    """Test calling r_to on an unfitted model raises RuntimeError."""
    model = TurnoverModel()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.r_to(282.0)


def test_turnover_r_to_scalar_float(mock_idata: Any) -> None:
    """Test scalar float input returns a float, and r_to(282.0) is identically 100.0%."""
    model = TurnoverModel()
    model.idata = mock_idata

    val = model.r_to(282.0)
    assert isinstance(val, float)
    assert np.isclose(val, 100.0, atol=1e-5)


def test_turnover_r_to_high_concentration_floor(mock_idata: Any) -> None:
    """Test that extreme concentration approaches the suppression floor (~25-35%)."""
    model = TurnoverModel()
    model.idata = mock_idata

    r_hi = model.r_to(12000.0)
    assert isinstance(r_hi, float)
    # With synthetic mock_idata priors, r_floor is beta(4,8) ~ 0.33
    assert 20.0 <= r_hi <= 45.0


def test_turnover_r_to_fractional_mode(mock_idata: Any) -> None:
    """Test as_percentage=False returns fractional modifier (1.0 at baseline)."""
    model = TurnoverModel()
    model.idata = mock_idata

    val = model.r_to(282.0, as_percentage=False)
    assert isinstance(val, float)
    assert np.isclose(val, 1.0, atol=1e-5)


def test_turnover_r_to_list_input(mock_idata: Any) -> None:
    """Test Python list input returns a matching 1D NumPy array."""
    model = TurnoverModel()
    model.idata = mock_idata

    inputs = [282.0, 3500.0, 12000.0]
    out = model.r_to(inputs)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.isclose(out[0], 100.0, atol=1e-5)
    # Monotonic suppression
    assert out[0] > out[1] > out[2]


def test_turnover_r_to_ndarray_input(mock_idata: Any) -> None:
    """Test NumPy ndarray input preserves array shape."""
    model = TurnoverModel()
    model.idata = mock_idata

    c_arr = np.array([[282.0, 3500.0], [5000.0, 12000.0]])
    out = model.r_to(c_arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert np.isclose(out[0, 0], 100.0, atol=1e-5)


def test_turnover_r_to_invalid_inputs(mock_idata: Any) -> None:
    """Test non-positive and non-finite inputs raise ValueError."""
    model = TurnoverModel()
    model.idata = mock_idata

    with pytest.raises(ValueError, match="positive and finite"):
        model.r_to(-50.0)

    with pytest.raises(ValueError, match="positive and finite"):
        model.r_to(0.0)

    with pytest.raises(ValueError, match="positive and finite"):
        model.r_to(float("nan"))

    with pytest.raises(ValueError, match="strictly positive"):
        model.r_to([282.0, -100.0])

    with pytest.raises(ValueError, match="finite values"):
        model.r_to([282.0, float("inf")])


# ==============================================================================
# Fast Unit Tests: Alternative Formulations r_to Evaluation
# ==============================================================================
def test_turnover_r_to_hill_2p() -> None:
    """Test r_to calculation for hill_2p formulation."""
    rng = np.random.default_rng(42)
    n_c, n_d = 2, 20
    data = {
        "posterior": {
            "bfr0": rng.normal(0.076, 0.005, size=(n_c, n_d)),
            "eta_km": rng.normal(np.log(3500.0), 0.1, size=(n_c, n_d)),
            "n_hill": rng.gamma(3.0, 1.5, size=(n_c, n_d)),
            "sigma_log": rng.exponential(0.3, size=(n_c, n_d)),
        }
    }
    idata = azb.from_dict(data)

    model = TurnoverModel(formulation="hill_2p")
    model.idata = idata

    r_ctrl = model.r_to(282.0)
    assert np.isclose(r_ctrl, 100.0, atol=1e-5)
    r_high = model.r_to(20000.0)
    assert r_high < 10.0  # Zero-floor suppresses toward 0


def test_turnover_r_to_log_linear() -> None:
    """Test r_to calculation for log_linear formulation."""
    rng = np.random.default_rng(42)
    n_c, n_d = 2, 20
    data = {
        "posterior": {
            "a": rng.normal(0.15, 0.01, size=(n_c, n_d)),
            "b": rng.normal(0.013, 0.001, size=(n_c, n_d)),
            "sigma_log": rng.exponential(0.3, size=(n_c, n_d)),
        }
    }
    idata = azb.from_dict(data)

    model = TurnoverModel(formulation="log_linear")
    model.idata = idata

    r_ctrl = model.r_to(282.0)
    assert np.isclose(r_ctrl, 100.0, atol=1e-5)


def test_turnover_r_to_hormesis() -> None:
    """Test r_to calculation for hormesis formulation."""
    rng = np.random.default_rng(42)
    n_c, n_d = 2, 20
    bfr0 = rng.normal(0.076, 0.005, size=(n_c, n_d))
    r_floor = rng.beta(4.0, 8.0, size=(n_c, n_d))
    data = {
        "posterior": {
            "bfr0": bfr0,
            "r_floor": r_floor,
            "bfr_floor": bfr0 * r_floor,
            "eta_km": rng.normal(np.log(3500.0), 0.1, size=(n_c, n_d)),
            "n_hill": rng.gamma(3.0, 1.5, size=(n_c, n_d)),
            "f": rng.normal(1e-5, 1e-6, size=(n_c, n_d)),
            "sigma_log": rng.exponential(0.3, size=(n_c, n_d)),
        }
    }
    idata = azb.from_dict(data)

    model = TurnoverModel(formulation="hormesis")
    model.idata = idata

    r_ctrl = model.r_to(282.0)
    assert np.isclose(r_ctrl, 100.0, atol=1e-5)


# ==============================================================================
# Fast Unit Tests: Clearance Grid
# ==============================================================================
def test_turnover_clearance_grid_structure(mock_idata: Any) -> None:
    """Test clearance_grid returns expected columns, shape, and 94% HDI bounds."""
    model = TurnoverModel()
    model.idata = mock_idata

    grid = model.clearance_grid(c_min=100.0, c_max=10000.0, n_points=25)
    assert isinstance(grid, pd.DataFrame)
    assert len(grid) == 25

    expected_cols = ["c_bone", "mean", "median", "std", "hdi_lower", "hdi_upper"]
    assert list(grid.columns) == expected_cols

    # Verify concentration bounds
    assert np.isclose(grid["c_bone"].iloc[0], 100.0)
    assert np.isclose(grid["c_bone"].iloc[-1], 10000.0)

    # Verify HDI bounds envelop median
    assert (grid["hdi_lower"] <= grid["median"] + 1e-6).all()
    assert (grid["median"] <= grid["hdi_upper"] + 1e-6).all()

    # Verify general downward trend of suppression
    assert grid["mean"].iloc[0] > grid["mean"].iloc[-1]


def test_turnover_clearance_grid_validation(mock_idata: Any) -> None:
    """Test clearance_grid input bounds validation."""
    model = TurnoverModel()
    model.idata = mock_idata

    with pytest.raises(ValueError, match="c_min must be strictly positive"):
        model.clearance_grid(c_min=0.0, c_max=1000.0)

    with pytest.raises(ValueError, match="c_max .* must be strictly greater than c_min"):
        model.clearance_grid(c_min=500.0, c_max=500.0)

    with pytest.raises(ValueError, match="n_points must be at least 2"):
        model.clearance_grid(c_min=100.0, c_max=1000.0, n_points=1)


# ==============================================================================
# Fast Unit Tests: Prediction, Summary & Diagnostics
# ==============================================================================
def test_turnover_predict(mock_idata: Any) -> None:
    """Test predict returns trajectory and observation draws."""
    model = TurnoverModel()
    model.idata = mock_idata

    preds = model.predict([282.0, 3500.0, 10000.0], random_seed=42)
    assert "c_bone" in preds
    assert "mu" in preds
    assert "y_obs" in preds
    assert preds["mu"].shape[1] == 3
    assert preds["y_obs"].shape[1] == 3


def test_turnover_summary_and_diagnostics(mock_idata: Any) -> None:
    """Test summary and diagnostics extraction using ArviZ."""
    model = TurnoverModel()
    model.idata = mock_idata

    sm = model.summary()
    assert isinstance(sm, pd.DataFrame)
    assert "bfr0" in sm.index

    diag = model.diagnostics()
    assert isinstance(diag, dict)
    assert "converged" in diag
    assert "max_rhat" in diag
    assert "divergences" in diag
    assert diag["divergences"] == 0


# ==============================================================================
# Fast Unit Tests: NetCDF Save & Load Roundtrip
# ==============================================================================
def test_turnover_netcdf_save_load_roundtrip(mock_idata: Any, tmp_path: Path) -> None:
    """Test saving and reloading fitted model via NetCDF h5netcdf engine."""
    model = TurnoverModel()
    model.idata = mock_idata

    save_file = tmp_path / "turnover_test.nc"
    model.save(save_file)
    assert save_file.is_file()

    loaded_model = TurnoverModel.load(save_file)
    assert loaded_model.idata is not None

    # Verify identical prediction after loading
    r_orig = model.r_to([282.0, 3500.0, 12000.0])
    r_load = loaded_model.r_to([282.0, 3500.0, 12000.0])
    np.testing.assert_allclose(r_orig, r_load, rtol=1e-5)


def test_turnover_save_unfitted_raises(tmp_path: Path) -> None:
    """Test saving an unfitted model raises RuntimeError."""
    model = TurnoverModel()
    with pytest.raises(RuntimeError, match="No fitted InferenceData to save"):
        model.save(tmp_path / "unfitted.nc")


def test_turnover_load_nonexistent_raises() -> None:
    """Test loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="NetCDF file not found"):
        TurnoverModel.load("non_existent_turnover_file_xyz.nc")


def test_turnover_compare_formulations_with_models_dict() -> None:
    """Test compare_formulations directly with a dict of synthetic InferenceData."""
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_obs = 2, 30, 15

    def make_mock(offset: float) -> azb.InferenceData:
        return azb.from_dict(
            {
                "posterior": {"bfr0": rng.normal(size=(n_chains, n_draws))},
                "log_likelihood": {
                    "bfr_obs": -rng.exponential(1.0, size=(n_chains, n_draws, n_obs)) - offset
                },
            }
        )

    mock_models = {
        "4p_hill": make_mock(0.1),
        "hill_2p": make_mock(0.4),
        "log_linear": make_mock(0.8),
    }

    model = TurnoverModel()
    comp = model.compare_formulations(models=mock_models)
    assert isinstance(comp, pd.DataFrame)
    assert len(comp) == 3
    assert "rank" in comp.columns
    assert "weight" in comp.columns
    assert "elpd_loo" in comp.columns
    assert "d_elpd" in comp.columns


# ==============================================================================
# Slow Integration Test: Full MCMC Sampling on Boivin Histomorphometry Data
# ==============================================================================
@pytest.mark.slow
def test_turnover_mcmc_boivin_convergence() -> None:
    """Full Bayesian MCMC sampling on Boivin histomorphometric dataset.

    Verifies:
    1. Sampling completes with 0 divergences.
    2. Max R-hat <= 1.01.
    3. Bulk and Tail ESS >= 400.
    4. Km posterior median/mean is within [1500, 7000] mg/kg.
    5. r_TO(282) is ~100% (within 0.1%).
    6. r_TO(12000) is within [25%, 35%].
    """
    df = load_boivin_data()
    model = TurnoverModel(data=df)

    idata = model.fit(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.95,
        random_seed=42,
    )
    assert idata is not None

    diag = model.diagnostics()
    assert diag["divergences"] == 0, f"Expected 0 divergences, got {diag['divergences']}"
    assert diag["max_rhat"] <= 1.01, f"Max R-hat exceeded 1.01: {diag['max_rhat']}"
    assert diag["min_ess_bulk"] >= 400, f"Min ESS bulk below 400: {diag['min_ess_bulk']}"
    assert diag["min_ess_tail"] >= 400, f"Min ESS tail below 400: {diag['min_ess_tail']}"
    assert diag["converged"] is True

    # Parameter estimates
    sm = model.summary()
    km_mean = sm.loc["km", "mean"]
    assert 1500.0 <= km_mean <= 7000.0, f"Km mean {km_mean} outside expected [1500, 7000]"

    # Modifier evaluations
    r_ctrl = model.r_to(282.0)
    assert np.isclose(r_ctrl, 100.0, atol=0.1), f"r_TO(282) = {r_ctrl}, expected ~100.0%"

    r_plateau = model.r_to(12000.0)
    assert 25.0 <= r_plateau <= 35.0, f"r_TO(12000) = {r_plateau}%, expected in [25, 35]%"
