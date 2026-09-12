"""Shared test fixtures for the bone-predictor test suite.

Configures headless Matplotlib backend, autouse figure cleanup, and provides
fast in-memory synthetic datasets and mock InferenceData fixtures.
"""

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

from bone_predictor.data import PreparedData

# Ensure headless matplotlib backend for non-interactive test runs
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures() -> Any:
    """Close all open matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def sample_boivin_df() -> pd.DataFrame:
    """Fast in-memory synthetic Boivin histomorphometry DataFrame."""
    return pd.DataFrame(
        {
            "c_bone": [282.0, 2400.0, 3500.0, 5000.0, 12000.0],
            "cn_bfr": [0.076, 0.065, 0.045, 0.025, 0.020],
            "cn_bfr_control": [100.0, 85.5, 59.2, 32.9, 26.3],
        }
    )


@pytest.fixture
def sample_accretion_df() -> pd.DataFrame:
    """Fast in-memory synthetic calcium accretion DataFrame."""
    return pd.DataFrame(
        {
            "study": ["Test Study"] * 4,
            "age": [0.5, 12.0, 25.0, 60.0],
            "sex": ["F", "F", "M", "M"],
            "f": [0.12, 0.45, 0.23, 0.22],
            "bw": [6.5, 38.0, 75.0, 76.0],
            "nbfr": [0.12 / 6.5, 0.45 / 38.0, 0.23 / 75.0, 0.22 / 76.0],
        }
    )


@pytest.fixture
def sample_stepan_df() -> pd.DataFrame:
    """Fast in-memory synthetic Stepan bone biomarker DataFrame."""
    return pd.DataFrame(
        {
            "marker": ["alp", "alp", "hp", "hp"],
            "sex": ["F", "M", "F", "M"],
            "age": [0.5, 14.0, 55.0, 30.0],
            "value": [120.0, 140.0, 35.0, 18.0],
            "units": ["U/L", "U/L", "mmol/mol", "mmol/mol"],
        }
    )


@pytest.fixture
def sample_prepared_data(
    sample_boivin_df: pd.DataFrame,
    sample_accretion_df: pd.DataFrame,
    sample_stepan_df: pd.DataFrame,
) -> PreparedData:
    """Fast in-memory PreparedData instance for sub-second test execution."""
    return PreparedData(
        boivin_turnover=sample_boivin_df,
        calcium_accretion=sample_accretion_df,
        stepan_markers=sample_stepan_df,
        data_hash="synthetic_test_hash_1234567890abcdef",
    )


@pytest.fixture
def mock_idata() -> Any:
    """Synthetic InferenceData object for fast downstream testing without MCMC."""
    from arviz_base import from_dict

    rng = np.random.default_rng(1985)
    n_chains, n_draws, n_obs = 2, 50, 15

    bfr0 = rng.normal(0.076, 0.005, size=(n_chains, n_draws))
    r_floor = rng.beta(4.0, 8.0, size=(n_chains, n_draws))
    eta_km = rng.normal(np.log(3500.0), 0.2, size=(n_chains, n_draws))
    km = np.exp(eta_km)
    bfr_floor = bfr0 * r_floor
    n_hill = rng.gamma(3.0, 1.5, size=(n_chains, n_draws))
    sigma_log = rng.exponential(0.3, size=(n_chains, n_draws))

    data = {
        "posterior": {
            "bfr0": bfr0,
            "r_floor": r_floor,
            "eta_km": eta_km,
            "km": km,
            "bfr_floor": bfr_floor,
            "n_hill": n_hill,
            "sigma_log": sigma_log,
        },
        "sample_stats": {
            "diverging": np.zeros((n_chains, n_draws), dtype=bool),
            "energy": rng.normal(10.0, 1.0, size=(n_chains, n_draws)),
        },
        "observed_data": {"bfr": rng.uniform(0.01, 0.08, size=n_obs)},
        "log_likelihood": {"bfr": -rng.exponential(1.0, size=(n_chains, n_draws, n_obs))},
    }
    return from_dict(data)


@pytest.fixture
def mock_biomarker_idata() -> Any:
    """Synthetic InferenceData for fast BiomarkerModel unit testing without MCMC."""
    from arviz_base import from_dict

    rng = np.random.default_rng(1985)
    n_chains, n_draws, n_obs = 2, 50, 20

    c_f_alp = rng.normal(9.94, 0.15, size=(n_chains, n_draws))
    c_m_alp = rng.normal(10.52, 0.15, size=(n_chains, n_draws))
    c_f_hp = rng.normal(16.81, 0.20, size=(n_chains, n_draws))
    c_m_hp = rng.normal(15.92, 0.20, size=(n_chains, n_draws))

    lambda_inf = rng.normal(1.33, 0.03, size=(n_chains, n_draws))
    gamma_inf = rng.normal(1.15, 0.03, size=(n_chains, n_draws))
    A_inf_alp = rng.normal(110.0, 4.0, size=(n_chains, n_draws))
    A_inf_hp = rng.normal(165.0, 5.0, size=(n_chains, n_draws))

    mu_pub_f = rng.normal(11.5, 0.1, size=(n_chains, n_draws))
    mu_pub_m = rng.normal(13.8, 0.1, size=(n_chains, n_draws))
    sigma_pub_f = rng.normal(1.5, 0.05, size=(n_chains, n_draws))
    sigma_pub_m = rng.normal(1.6, 0.05, size=(n_chains, n_draws))
    A_pub_f_alp = rng.normal(75.0, 3.0, size=(n_chains, n_draws))
    A_pub_m_alp = rng.normal(95.0, 3.0, size=(n_chains, n_draws))
    A_pub_f_hp = rng.normal(80.0, 3.0, size=(n_chains, n_draws))
    A_pub_m_hp = rng.normal(90.0, 3.0, size=(n_chains, n_draws))

    mu_meno = rng.normal(54.0, 0.3, size=(n_chains, n_draws))
    sigma_meno = rng.normal(4.5, 0.1, size=(n_chains, n_draws))
    A_meno_alp = rng.normal(2.0, 0.2, size=(n_chains, n_draws))
    A_meno_hp = rng.normal(2.5, 0.2, size=(n_chains, n_draws))

    mu_mid = rng.normal(7.0, 0.1, size=(n_chains, n_draws))
    sigma_mid = rng.normal(0.8, 0.05, size=(n_chains, n_draws))
    A_mid_alp = rng.normal(5.0, 0.5, size=(n_chains, n_draws))
    A_mid_hp = rng.normal(6.0, 0.5, size=(n_chains, n_draws))

    sigma_log_alp = rng.exponential(0.25, size=(n_chains, n_draws))
    sigma_log_hp = rng.exponential(0.25, size=(n_chains, n_draws))

    data = {
        "posterior": {
            "c_f_alp": c_f_alp,
            "c_m_alp": c_m_alp,
            "c_female_alp": c_f_alp,
            "c_male_alp": c_m_alp,
            "c_f_hp": c_f_hp,
            "c_m_hp": c_m_hp,
            "c_female_hp": c_f_hp,
            "c_male_hp": c_m_hp,
            "c_female": c_f_alp,
            "c_male": c_m_alp,
            "lambda_inf": lambda_inf,
            "gamma_inf": gamma_inf,
            "A_inf_alp": A_inf_alp,
            "A_inf_hp": A_inf_hp,
            "A_inf": A_inf_alp,
            "mu_pub_f": mu_pub_f,
            "mu_pub_m": mu_pub_m,
            "mu_pub_female": mu_pub_f,
            "mu_pub_male": mu_pub_m,
            "sigma_pub_f": sigma_pub_f,
            "sigma_pub_m": sigma_pub_m,
            "sigma_pub_female": sigma_pub_f,
            "sigma_pub_male": sigma_pub_m,
            "A_pub_f_alp": A_pub_f_alp,
            "A_pub_m_alp": A_pub_m_alp,
            "A_pub_female_alp": A_pub_f_alp,
            "A_pub_male_alp": A_pub_m_alp,
            "A_pub_f_hp": A_pub_f_hp,
            "A_pub_m_hp": A_pub_m_hp,
            "A_pub_female_hp": A_pub_f_hp,
            "A_pub_male_hp": A_pub_m_hp,
            "A_pub_female": A_pub_f_alp,
            "A_pub_male": A_pub_m_alp,
            "mu_meno": mu_meno,
            "sigma_meno": sigma_meno,
            "A_meno_alp": A_meno_alp,
            "A_meno_hp": A_meno_hp,
            "A_meno": A_meno_alp,
            "mu_mid": mu_mid,
            "sigma_mid": sigma_mid,
            "A_mid_alp": A_mid_alp,
            "A_mid_hp": A_mid_hp,
            "A_mid": A_mid_alp,
            "sigma_log_alp": sigma_log_alp,
            "sigma_log_hp": sigma_log_hp,
            "sigma_log": sigma_log_alp,
        },
        "sample_stats": {
            "diverging": np.zeros((n_chains, n_draws), dtype=bool),
            "energy": rng.normal(10.0, 1.0, size=(n_chains, n_draws)),
        },
        "observed_data": {"obs": rng.uniform(10.0, 150.0, size=n_obs)},
        "log_likelihood": {"obs": -rng.exponential(1.0, size=(n_chains, n_draws, n_obs))},
    }
    return from_dict(data)
