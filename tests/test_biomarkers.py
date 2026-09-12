"""Tests for the Biomarker Spectral Deconvolution Model (BiomarkerModel).

Includes fast sub-second unit tests using mock_biomarker_idata and synthetic fixtures,
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

from bone_predictor.biomarkers import BiomarkerModel
from bone_predictor.data import load_stepan_markers


# ==============================================================================
# Fast Unit Tests: Initialization & Parameter Validation
# ==============================================================================
def test_biomarker_init_default() -> None:
    """Test default initialization has K=3, joint marker, and hierarchical sex model."""
    model = BiomarkerModel()
    assert model.n_components == 3
    assert model.marker == "joint"
    assert model.sex_model == "hierarchical"
    assert model.model is None
    assert model.idata is None


@pytest.mark.parametrize("k", [2, 3, 4])
def test_biomarker_init_valid_components(k: int) -> None:
    """Test initialization across supported component architectures K=2, 3, 4."""
    model = BiomarkerModel(n_components=k)
    assert model.n_components == k


def test_biomarker_init_invalid_components() -> None:
    """Test that invalid n_components raises ValueError."""
    with pytest.raises(ValueError, match="Invalid n_components"):
        BiomarkerModel(n_components=1)

    with pytest.raises(ValueError, match="Invalid n_components"):
        BiomarkerModel(n_components=5)


@pytest.mark.parametrize("marker", ["alp", "hp", "joint"])
def test_biomarker_init_valid_markers(marker: str) -> None:
    """Test initialization across supported marker options."""
    model = BiomarkerModel(marker=marker)  # type: ignore[arg-type]
    assert model.marker == marker


def test_biomarker_init_invalid_marker() -> None:
    """Test that invalid marker name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid marker"):
        BiomarkerModel(marker="invalid_marker")  # type: ignore[arg-type]


@pytest.mark.parametrize("sex_model", ["hierarchical", "independent"])
def test_biomarker_init_valid_sex_models(sex_model: str) -> None:
    """Test initialization across supported sex models."""
    model = BiomarkerModel(sex_model=sex_model)  # type: ignore[arg-type]
    assert model.sex_model == sex_model


def test_biomarker_init_invalid_sex_model() -> None:
    """Test that invalid sex model raises ValueError."""
    with pytest.raises(ValueError, match="Invalid sex_model"):
        BiomarkerModel(sex_model="invalid_sex_model")  # type: ignore[arg-type]


# ==============================================================================
# Fast Unit Tests: Data Validation
# ==============================================================================
def test_biomarker_data_validation_valid(sample_stepan_df: pd.DataFrame) -> None:
    """Test valid DataFrame is accepted and standardized."""
    model = BiomarkerModel(data=sample_stepan_df)
    assert model.data is not None
    assert len(model.data) == len(sample_stepan_df)
    assert set(model.data["sex"]).issubset({"F", "M"})
    assert set(model.data["marker"]).issubset({"alp", "hp"})


def test_biomarker_data_validation_missing_columns() -> None:
    """Test missing required columns raises ValueError."""
    bad_df = pd.DataFrame({"age": [1.0, 2.0], "value": [10.0, 20.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        BiomarkerModel(data=bad_df)


def test_biomarker_data_validation_empty_df() -> None:
    """Test empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame(columns=["marker", "sex", "age", "value"])
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        BiomarkerModel(data=empty_df)


def test_biomarker_data_validation_non_positive_values() -> None:
    """Test non-positive values raise ValueError."""
    bad_val = pd.DataFrame(
        {"marker": ["alp", "alp"], "sex": ["F", "M"], "age": [1.0, 2.0], "value": [10.0, -5.0]}
    )
    with pytest.raises(ValueError, match="strictly positive"):
        BiomarkerModel(data=bad_val)


def test_biomarker_data_validation_invalid_type() -> None:
    """Test passing non-DataFrame raises TypeError."""
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        BiomarkerModel(data="not_a_df")  # type: ignore[arg-type]


# ==============================================================================
# Fast Unit Tests: Model Building & Graph Structure
# ==============================================================================
@pytest.mark.parametrize(
    ("n_components", "expected_vars"),
    [
        (
            2,
            [
                "c_f_alp",
                "c_m_alp",
                "c_f_hp",
                "c_m_hp",
                "lambda_inf",
                "gamma_inf",
                "A_inf_alp",
                "A_inf_hp",
                "mu_pub_f",
                "mu_pub_m",
                "A_pub_f_alp",
                "A_pub_m_alp",
                "sigma_log_alp",
                "sigma_log_hp",
                "mu",
                "obs",
            ],
        ),
        (
            3,
            [
                "c_f_alp",
                "c_m_alp",
                "c_f_hp",
                "c_m_hp",
                "lambda_inf",
                "gamma_inf",
                "A_inf_alp",
                "A_inf_hp",
                "mu_pub_f",
                "mu_pub_m",
                "A_pub_f_alp",
                "A_pub_m_alp",
                "mu_meno",
                "sigma_meno",
                "A_meno_alp",
                "A_meno_hp",
                "sigma_log_alp",
                "sigma_log_hp",
                "mu",
                "obs",
            ],
        ),
        (
            4,
            [
                "c_f_alp",
                "c_m_alp",
                "c_f_hp",
                "c_m_hp",
                "lambda_inf",
                "gamma_inf",
                "A_inf_alp",
                "A_inf_hp",
                "mu_pub_f",
                "mu_pub_m",
                "A_pub_f_alp",
                "A_pub_m_alp",
                "mu_meno",
                "sigma_meno",
                "A_meno_alp",
                "A_meno_hp",
                "mu_mid",
                "sigma_mid",
                "A_mid_alp",
                "A_mid_hp",
                "sigma_log_alp",
                "sigma_log_hp",
                "mu",
                "obs",
            ],
        ),
    ],
)
def test_biomarker_build_components(
    sample_stepan_df: pd.DataFrame, n_components: int, expected_vars: list[str]
) -> None:
    """Test model graph registers appropriate variables for K=2, 3, 4."""
    model = BiomarkerModel(n_components=n_components, data=sample_stepan_df)
    pm_model = model.build()
    assert isinstance(pm_model, pm.Model)
    assert "obs_id" in pm_model.coords
    for var in expected_vars:
        assert var in pm_model.named_vars, f"Expected '{var}' in K={n_components} model"


def test_biomarker_build_single_marker_alp(sample_stepan_df: pd.DataFrame) -> None:
    """Test single marker 'alp' model builds with ALP parameters only."""
    model = BiomarkerModel(marker="alp", data=sample_stepan_df)
    pm_model = model.build()
    assert isinstance(pm_model, pm.Model)
    assert "c_f_alp" in pm_model.named_vars
    assert "c_female" in pm_model.named_vars
    assert "c_f_hp" not in pm_model.named_vars


def test_biomarker_build_independent_sex(sample_stepan_df: pd.DataFrame) -> None:
    """Test independent sex model builds without hierarchical deviation shifts."""
    model = BiomarkerModel(sex_model="independent", data=sample_stepan_df)
    pm_model = model.build()
    assert isinstance(pm_model, pm.Model)
    assert "c_f_alp" in pm_model.named_vars
    assert "mu_pub_female" in pm_model.named_vars
    assert "mu_pub_base" not in pm_model.named_vars


def test_biomarker_sample_prior_predictive(sample_stepan_df: pd.DataFrame) -> None:
    """Test prior predictive sampling runs cleanly and returns InferenceData."""
    model = BiomarkerModel(data=sample_stepan_df)
    prior = model.sample_prior_predictive(draws=10, random_seed=42)
    assert prior is not None
    assert hasattr(prior, "prior")
    assert "c_f_alp" in prior.prior


# ==============================================================================
# Fast Unit Tests: Relative Activity Index (using mock_biomarker_idata)
# ==============================================================================
def test_biomarker_rai_unfitted_raises() -> None:
    """Test calling relative_activity_index on unfitted model raises RuntimeError."""
    model = BiomarkerModel()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.relative_activity_index(30.0, sex="F", marker="alp")


def test_biomarker_rai_scalar_float(mock_biomarker_idata: Any) -> None:
    """Test scalar float input returns a float."""
    model = BiomarkerModel()
    model.idata = mock_biomarker_idata

    val = model.relative_activity_index(30.0, sex="F", marker="alp")
    assert isinstance(val, float)
    # Adult baseline at age 30 should be ~1.00
    assert np.isclose(val, 1.00, atol=0.05)


def test_biomarker_rai_array_broadcasting(mock_biomarker_idata: Any) -> None:
    """Test array and list inputs preserve shape and return np.ndarray."""
    model = BiomarkerModel()
    model.idata = mock_biomarker_idata

    ages = [0.5, 11.5, 30.0, 54.0]
    out = model.relative_activity_index(ages, sex="F", marker="alp")
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)

    arr_2d = np.array([[0.5, 11.5], [30.0, 54.0]])
    out_2d = model.relative_activity_index(arr_2d, sex="F", marker="alp")
    assert isinstance(out_2d, np.ndarray)
    assert out_2d.shape == (2, 2)


def test_biomarker_rai_physiological_milestone_bounds(mock_biomarker_idata: Any) -> None:
    """Verify RAI values satisfy biological acceptance criteria across milestones."""
    model = BiomarkerModel()
    model.idata = mock_biomarker_idata

    # 1. Adult baseline anchor at age 30: RAI ~ 1.00
    rai_adult_f = model.relative_activity_index(30.0, sex="F", marker="alp")
    rai_adult_m = model.relative_activity_index(30.0, sex="M", marker="alp")
    assert np.isclose(rai_adult_f, 1.00, atol=0.05)
    assert np.isclose(rai_adult_m, 1.00, atol=0.05)

    # 2. Infancy surge at age 0.5: RAI > 5.0x
    rai_inf_f = model.relative_activity_index(0.5, sex="F", marker="alp")
    rai_inf_m = model.relative_activity_index(0.5, sex="M", marker="alp")
    assert rai_inf_f > 5.0
    assert rai_inf_m > 5.0

    # 3. Pubertal peak timing: Female ~11.5 yr, Male ~13.8 yr
    rai_pub_f_peak = model.relative_activity_index(11.5, sex="F", marker="alp")
    rai_pub_f_early = model.relative_activity_index(8.0, sex="F", marker="alp")
    assert rai_pub_f_peak > rai_pub_f_early
    assert rai_pub_f_peak > 3.0

    rai_pub_m_peak = model.relative_activity_index(13.8, sex="M", marker="alp")
    rai_pub_m_early = model.relative_activity_index(8.0, sex="M", marker="alp")
    assert rai_pub_m_peak > rai_pub_m_early
    assert rai_pub_m_peak > 3.0

    # 4. Postmenopausal elevation in females: age 54 > baseline, male has no surge
    rai_meno_f = model.relative_activity_index(54.0, sex="F", marker="alp")
    rai_meno_m = model.relative_activity_index(54.0, sex="M", marker="alp")
    assert rai_meno_f > 1.05  # Elevation due to postmenopausal component
    assert np.isclose(rai_meno_m, 1.00, atol=0.05)  # Identically 0 menopause in males


def test_biomarker_rai_single_marker_mismatch_raises(mock_biomarker_idata: Any) -> None:
    """Test querying wrong marker on single-marker fitted model raises ValueError."""
    model = BiomarkerModel(marker="alp")
    model.idata = mock_biomarker_idata

    with pytest.raises(ValueError, match="Model was fitted for marker='alp'"):
        model.relative_activity_index(30.0, sex="F", marker="hp")


def test_biomarker_rai_invalid_inputs(mock_biomarker_idata: Any) -> None:
    """Test non-positive or non-finite inputs raise ValueError."""
    model = BiomarkerModel()
    model.idata = mock_biomarker_idata

    with pytest.raises(ValueError, match="strictly positive"):
        model.relative_activity_index(-5.0, sex="F", marker="alp")

    with pytest.raises(ValueError, match="strictly positive"):
        model.relative_activity_index(0.0, sex="F", marker="alp")

    with pytest.raises(ValueError, match="finite"):
        model.relative_activity_index(float("nan"), sex="F", marker="alp")

    with pytest.raises(ValueError, match="Invalid marker"):
        model.relative_activity_index(30.0, sex="F", marker="unknown")


# ==============================================================================
# Fast Unit Tests: Grid, Prediction, Summary & Diagnostics
# ==============================================================================
def test_biomarker_relative_activity_grid(mock_biomarker_idata: Any) -> None:
    """Test relative_activity_grid returns expected columns and valid HDI bounds."""
    model = BiomarkerModel()
    model.idata = mock_biomarker_idata

    grid = model.relative_activity_grid(age_min=0.0, age_max=80.0, step=1.0, sex="F", marker="alp")
    assert isinstance(grid, pd.DataFrame)
    assert len(grid) == 81

    expected_cols = ["age", "sex", "marker", "mean", "median", "std", "hdi_lower", "hdi_upper"]
    assert list(grid.columns) == expected_cols
    assert (grid["hdi_lower"] <= grid["median"] + 1e-6).all()
    assert (grid["median"] <= grid["hdi_upper"] + 1e-6).all()


def test_biomarker_predict(mock_biomarker_idata: Any) -> None:
    """Test predict returns trajectory, simulated observations, and RAI draws."""
    model = BiomarkerModel()
    model.idata = mock_biomarker_idata

    preds = model.predict([0.5, 11.5, 30.0], sex="F", marker="alp", random_seed=42)
    assert "age" in preds
    assert "mu" in preds
    assert "y_obs" in preds
    assert "rai" in preds
    assert preds["mu"].shape[1] == 3
    assert preds["y_obs"].shape[1] == 3
    assert preds["rai"].shape[1] == 3


def test_biomarker_summary_and_diagnostics(mock_biomarker_idata: Any) -> None:
    """Test summary and diagnostics extraction using ArviZ."""
    model = BiomarkerModel()
    model.idata = mock_biomarker_idata

    sm = model.summary()
    assert isinstance(sm, pd.DataFrame)
    assert "c_f_alp" in sm.index

    diag = model.diagnostics()
    assert isinstance(diag, dict)
    assert "converged" in diag
    assert "max_rhat" in diag
    assert "divergences" in diag
    assert diag["divergences"] == 0


# ==============================================================================
# Fast Unit Tests: NetCDF Save & Load Roundtrip
# ==============================================================================
def test_biomarker_netcdf_save_load_roundtrip(mock_biomarker_idata: Any, tmp_path: Path) -> None:
    """Test saving and reloading fitted model via NetCDF h5netcdf engine."""
    model = BiomarkerModel(n_components=3, marker="joint", sex_model="hierarchical")
    model.idata = mock_biomarker_idata

    save_file = tmp_path / "biomarker_test.nc"
    model.save(save_file)
    assert save_file.is_file()

    loaded_model = BiomarkerModel.load(save_file)
    assert loaded_model.idata is not None
    assert loaded_model.n_components == 3
    assert loaded_model.marker == "joint"

    # Verify identical prediction before and after load
    test_ages = [0.5, 11.5, 30.0, 54.0]
    rai_orig = model.relative_activity_index(test_ages, sex="F", marker="alp")
    rai_load = loaded_model.relative_activity_index(test_ages, sex="F", marker="alp")
    np.testing.assert_allclose(rai_orig, rai_load, rtol=1e-5)


def test_biomarker_save_unfitted_raises(tmp_path: Path) -> None:
    """Test saving unfitted model raises RuntimeError."""
    model = BiomarkerModel()
    with pytest.raises(RuntimeError, match="No fitted InferenceData to save"):
        model.save(tmp_path / "unfitted.nc")


def test_biomarker_load_nonexistent_raises() -> None:
    """Test loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="NetCDF file not found"):
        BiomarkerModel.load("non_existent_biomarker_file_xyz.nc")


# ==============================================================================
# Fast Unit Tests: Component Comparison (using mock models dict)
# ==============================================================================
def test_biomarker_compare_components_with_models_dict() -> None:
    """Test compare_components directly with a dict of synthetic InferenceData."""
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_obs = 2, 30, 15

    def make_mock(offset: float) -> azb.InferenceData:
        return azb.from_dict(
            {
                "posterior": {"a": rng.normal(size=(n_chains, n_draws))},
                "log_likelihood": {
                    "obs": -rng.exponential(1.0, size=(n_chains, n_draws, n_obs)) - offset
                },
            }
        )

    mock_models = {
        "K=2": make_mock(0.5),
        "K=3": make_mock(0.1),
        "K=4": make_mock(0.2),
    }

    model = BiomarkerModel()
    comp = model.compare_components(models=mock_models)
    assert isinstance(comp, pd.DataFrame)
    assert len(comp) == 3
    assert "rank" in comp.columns
    assert "weight" in comp.columns
    assert "elpd_loo" in comp.columns
    assert "d_elpd" in comp.columns


# ==============================================================================
# Fast Unit Tests: Plotting Helpers
# ==============================================================================
def test_biomarker_plots(mock_biomarker_idata: Any, sample_stepan_df: pd.DataFrame) -> None:
    """Test plot_components and plot_relative_activity execute cleanly."""
    model = BiomarkerModel(data=sample_stepan_df)
    model.idata = mock_biomarker_idata

    fig_comp = model.plot_components(marker="alp", sex="F")
    assert fig_comp is not None

    fig_rai = model.plot_relative_activity(marker="alp")
    assert fig_rai is not None


# ==============================================================================
# Slow Integration Test: Full MCMC Sampling on Actual Stepan Biomarker Data
# ==============================================================================
@pytest.mark.slow
def test_biomarker_mcmc_stepan_convergence() -> None:
    """Full Bayesian MCMC sampling on Stepan et al. biomarker dataset.

    Verifies:
    1. Sampling completes with 0 divergences.
    2. Max R-hat <= 1.01.
    3. Bulk and Tail ESS >= 400.
    4. Adult baselines: mean c_ALP in [8.0, 14.0] U/L, mean c_HP in [12.0, 20.0] mmol/mol.
    5. Pubertal peak timing dimorphism: Female in [10.0, 13.0] yr, Male in [12.0, 15.5] yr.
    6. Infancy surge: RAI_ALP(0.5) > 5.0x baseline.
    7. Adult baseline anchor: RAI_ALP(30) approx 1.00 +- 0.08.
    """
    df = load_stepan_markers()
    model = BiomarkerModel(n_components=3, marker="joint", sex_model="hierarchical", data=df)

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

    sm = model.summary()

    # Baseline acceptance
    c_f_alp_mean = sm.loc["c_f_alp", "mean"]
    c_m_alp_mean = sm.loc["c_m_alp", "mean"]
    assert 8.0 <= c_f_alp_mean <= 14.0, f"Female ALP baseline {c_f_alp_mean} outside [8, 14]"
    assert 8.0 <= c_m_alp_mean <= 14.0, f"Male ALP baseline {c_m_alp_mean} outside [8, 14]"

    c_f_hp_mean = sm.loc["c_f_hp", "mean"]
    c_m_hp_mean = sm.loc["c_m_hp", "mean"]
    assert 12.0 <= c_f_hp_mean <= 20.0, f"Female HP baseline {c_f_hp_mean} outside [12, 20]"
    assert 12.0 <= c_m_hp_mean <= 20.0, f"Male HP baseline {c_m_hp_mean} outside [12, 20]"

    # Pubertal timing acceptance
    mu_pub_f_mean = sm.loc["mu_pub_f", "mean"]
    mu_pub_m_mean = sm.loc["mu_pub_m", "mean"]
    assert 10.0 <= mu_pub_f_mean <= 13.0, f"Female pubertal timing {mu_pub_f_mean} outside [10, 13]"
    assert 12.0 <= mu_pub_m_mean <= 15.5, f"Male pubertal timing {mu_pub_m_mean} outside [12, 15.5]"

    # Relative Activity Index checks
    rai_inf = model.relative_activity_index(0.5, sex="F", marker="alp")
    assert rai_inf > 5.0, f"Infancy RAI {rai_inf} expected > 5.0x"

    rai_adult = model.relative_activity_index(30.0, sex="F", marker="alp")
    assert np.isclose(rai_adult, 1.00, atol=0.08), f"Adult baseline RAI {rai_adult} expected ~1.00"
