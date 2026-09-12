"""Biomarker Spectral Deconvolution Model (BiomarkerModel) for bone-predictor.

Implements multi-component Bayesian spectral deconvolution for serum bone-specific
alkaline phosphatase (b-ALP) and urinary hydroxyproline-to-creatinine ratio (u-HP)
from Stepan et al. (1985) cross-sectional data (N=2,944).

Supported Component Architectures:
1. K=2 (Pediatric):
   mu(t, s) = c(s) + K_inf(t) + K_pub(t, s)
2. K=3 (Default Primary):
   mu(t, s) = c(s) + K_inf(t) + K_pub(t, s) + [I_(s=Female) * K_meno(t)]
3. K=4 (Lifespan Auxological):
   mu(t, s) = c(s) + K_inf(t) + K_mid(t) + K_pub(t, s) + [I_(s=Female) * K_meno(t)]

Mathematical Kernels:
- Adult Homeostatic Baseline c(s): Anchored to ages 29-45 yr.
- Infancy Surge Kernel K_inf(t): Weibull decay kernel.
    K_inf(t) = A_inf * exp(-(t / lambda_inf)^gamma_inf)
- Mid-Childhood Adrenarche Kernel K_mid(t) (K=4 only): Gaussian adrenarche pulse.
    K_mid(t) = A_mid * exp(-(t - mu_mid)^2 / (2 * sigma_mid^2))
- Pubertal Growth Spurt Kernel K_pub(t, s): Sex-dimorphic Gaussian pulse.
    K_pub(t, s) = A_pub(s) * exp(-(t - mu_pub(s))^2 / (2 * sigma_pub(s)^2))
- Postmenopausal Surge Kernel K_meno(t, s): Female-specific Gaussian pulse.
    K_meno(t, s) = I_(s=Female) * A_meno * exp(-(t - mu_meno)^2 / (2 * sigma_meno^2))

Relative Activity Index (RAI):
- RAI_ALP(t, s) = mu_ALP(t, s) / c_ALP(s)  [Osteoblastic formation rate proxy r_form]
- RAI_HP(t, s)  = mu_HP(t, s)  / c_HP(s)   [Osteoclastic resorption rate proxy r_resorp]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import arviz_base as azb
import arviz_stats as azs
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from bone_predictor.constants import DEFAULT_HDI_PROB, RANDOM_SEED
from bone_predictor.data import load_stepan_markers, normalize_sex

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

MarkerType = Literal["alp", "hp", "joint"]
SexModelType = Literal["hierarchical", "independent"]
VALID_COMPONENTS: set[int] = {2, 3, 4}
VALID_MARKERS: set[str] = {"alp", "hp", "joint"}
VALID_SEX_MODELS: set[str] = {"hierarchical", "independent"}


def _compute_hdi_bounds(
    samples_2d: np.ndarray, prob: float = DEFAULT_HDI_PROB
) -> tuple[np.ndarray, np.ndarray]:
    """Compute highest density interval bounds along sample axis (axis 0)."""
    try:
        da = xr.DataArray(samples_2d, dims=["draw", "point"])
        hdi_da = azs.hdi(da, prob=prob, dim="draw")
        hdi_lower = hdi_da.sel(ci_bound="lower").values
        hdi_upper = hdi_da.sel(ci_bound="upper").values
        return np.asarray(hdi_lower, dtype=float), np.asarray(hdi_upper, dtype=float)
    except Exception as exc:
        logger.debug("azs.hdi computation failed (%s); using percentile fallback", exc)
        alpha = (1.0 - prob) / 2.0
        lower = np.percentile(samples_2d, 100.0 * alpha, axis=0)
        upper = np.percentile(samples_2d, 100.0 * (1.0 - alpha), axis=0)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


class BiomarkerModel:
    """Multi-component spectral deconvolution model for bone biomarkers.

    Fits bone turnover biomarkers (b-ALP and u-HP) across the human lifespan
    and derives dimensionless Relative Activity Indices (RAI_ALP and RAI_HP).

    Attributes
    ----------
    n_components : int
        Number of biological deconvolution components (2, 3, or 4). Default: 3.
    marker : str
        Marker selection ('alp', 'hp', or 'joint'). Default: 'joint'.
    sex_model : str
        Sex parameter sharing mode ('hierarchical' or 'independent').
    data : pd.DataFrame | None
        Underlying cross-sectional biomarker observations.
    model : pm.Model | None
        Active PyMC model instance.
    idata : Any | None
        Fitted InferenceData / DataTree containing posterior draws.
    sampler_used : str | None
        Sampling backend used during fit ('nutpie' or 'pymc').
    """

    def __init__(
        self,
        n_components: int = 3,
        marker: MarkerType = "joint",
        sex_model: SexModelType = "hierarchical",
        data: pd.DataFrame | None = None,
    ) -> None:
        """Initialize BiomarkerModel.

        Parameters
        ----------
        n_components : int, default=3
            Component architecture (2: infancy + puberty; 3: + menopause; 4: + adrenarche).
        marker : Literal['alp', 'hp', 'joint'], default='joint'
            Biomarker fitting mode.
        sex_model : Literal['hierarchical', 'independent'], default='hierarchical'
            Sex parameterization strategy.
        data : pd.DataFrame | None, optional
            Pre-loaded or custom dataset. If None, loaded lazily from bundled Stepan data.
        """
        if n_components not in VALID_COMPONENTS:
            raise ValueError(
                f"Invalid n_components={n_components}. Must be one of {sorted(VALID_COMPONENTS)}."
            )

        marker_norm = marker.strip().lower()
        if marker_norm not in VALID_MARKERS:
            raise ValueError(f"Invalid marker='{marker}'. Must be one of {sorted(VALID_MARKERS)}.")

        sex_model_norm = sex_model.strip().lower()
        if sex_model_norm not in VALID_SEX_MODELS:
            raise ValueError(
                f"Invalid sex_model='{sex_model}'. Must be one of {sorted(VALID_SEX_MODELS)}."
            )

        self.n_components: int = n_components
        self.marker: str = marker_norm
        self.sex_model: str = sex_model_norm
        self.data: pd.DataFrame | None = self._validate_data(data) if data is not None else None
        self.model: pm.Model | None = None
        self.idata: Any | None = None
        self.sampler_used: str | None = None

    @staticmethod
    def _validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and validate biomarker DataFrame schema."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("Input data DataFrame cannot be empty")

        col_map: dict[str, str] = {}
        for col in df.columns:
            c_lower = str(col).strip().lower()
            if c_lower in ("marker", "biomarker", "marker_type"):
                col_map[col] = "marker"
            elif c_lower in ("sex", "gender"):
                col_map[col] = "sex"
            elif c_lower in ("age", "age_yr", "years"):
                col_map[col] = "age"
            elif c_lower in ("value", "val", "concentration", "level"):
                col_map[col] = "value"
            elif c_lower in ("units", "unit"):
                col_map[col] = "units"

        renamed = df.rename(columns=col_map)
        required = {"marker", "sex", "age", "value"}
        if not required.issubset(renamed.columns):
            missing = required - set(renamed.columns)
            raise ValueError(f"Missing required columns in dataset: {missing}")

        clean_df = renamed.copy()
        clean_df["marker"] = clean_df["marker"].astype(str).str.strip().str.lower()
        invalid_markers = set(clean_df["marker"]) - {"alp", "hp"}
        if invalid_markers:
            raise ValueError(f"Dataset contains invalid marker values: {invalid_markers}")

        clean_df["sex"] = clean_df["sex"].apply(normalize_sex)
        clean_df["age"] = pd.to_numeric(clean_df["age"], errors="coerce")
        clean_df["value"] = pd.to_numeric(clean_df["value"], errors="coerce")

        if clean_df[["age", "value"]].isna().any().any():
            raise ValueError("Dataset contains non-numeric or NaN values in age or value")
        if (clean_df["value"] <= 0).any():
            raise ValueError("Biomarker concentration 'value' must be strictly positive")
        if (clean_df["age"] < 0.01).any():
            raise ValueError("Ages must be clamped to >= 0.01 yr")

        return clean_df.reset_index(drop=True)

    def _get_data(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Resolve dataset, prioritizing passed data, cached data, or bundled Stepan."""
        if data is not None:
            df = self._validate_data(data)
        elif self.data is not None:
            df = self.data
        else:
            df = load_stepan_markers()
            self.data = df

        if self.marker in ("alp", "hp"):
            df = df[df["marker"] == self.marker].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"No observations remain after filtering for marker='{self.marker}'"
                )

        return df

    def build(self, data: pd.DataFrame | None = None) -> pm.Model:
        """Build the PyMC probabilistic model for spectral deconvolution.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Data to build model on. Defaults to cached or bundled Stepan data.

        Returns
        -------
        pm.Model
            Configured PyMC model instance.
        """
        df = self._get_data(data)
        self.data = df

        n_obs = len(df)
        age_obs = df["age"].to_numpy(dtype=float)
        y_obs = df["value"].to_numpy(dtype=float)
        is_male_obs = (df["sex"] == "M").to_numpy(dtype=float)
        is_female_obs = 1.0 - is_male_obs

        is_joint = self.marker == "joint"
        if is_joint:
            is_hp_obs = (df["marker"] == "hp").to_numpy(dtype=float)
            is_alp_obs = 1.0 - is_hp_obs
        else:
            is_alp_obs = (
                np.ones(n_obs, dtype=float)
                if self.marker == "alp"
                else np.zeros(n_obs, dtype=float)
            )
            is_hp_obs = 1.0 - is_alp_obs

        coords = {"obs_id": np.arange(n_obs)}

        with pm.Model(coords=coords) as model:
            age_data = pm.Data("age_data", age_obs, dims="obs_id")
            is_f_data = pm.Data("is_f_data", is_female_obs, dims="obs_id")
            is_m_data = pm.Data("is_m_data", is_male_obs, dims="obs_id")

            # ------------------------------------------------------------------
            # 1. Adult Baseline Plateau c(s)
            # ------------------------------------------------------------------
            if self.marker in ("alp", "joint"):
                if self.sex_model == "hierarchical":
                    c_pop_alp = pm.TruncatedNormal(
                        "c_pop_alp", mu=10.25, sigma=1.5, lower=4.0, upper=20.0
                    )
                    delta_c_alp = pm.Normal("delta_c_alp", mu=0.5, sigma=0.5)
                    c_f_alp = pm.Deterministic("c_f_alp", c_pop_alp - 0.5 * delta_c_alp)
                    c_m_alp = pm.Deterministic("c_m_alp", c_pop_alp + 0.5 * delta_c_alp)
                else:
                    c_f_alp = pm.TruncatedNormal(
                        "c_f_alp", mu=10.0, sigma=1.5, lower=4.0, upper=20.0
                    )
                    c_m_alp = pm.TruncatedNormal(
                        "c_m_alp", mu=10.5, sigma=1.5, lower=4.0, upper=20.0
                    )
                pm.Deterministic("c_female_alp", c_f_alp)
                pm.Deterministic("c_male_alp", c_m_alp)

            if self.marker in ("hp", "joint"):
                if self.sex_model == "hierarchical":
                    c_pop_hp = pm.TruncatedNormal(
                        "c_pop_hp", mu=16.35, sigma=2.0, lower=6.0, upper=30.0
                    )
                    delta_c_hp = pm.Normal("delta_c_hp", mu=-0.8, sigma=0.5)
                    c_f_hp = pm.Deterministic("c_f_hp", c_pop_hp - 0.5 * delta_c_hp)
                    c_m_hp = pm.Deterministic("c_m_hp", c_pop_hp + 0.5 * delta_c_hp)
                else:
                    c_f_hp = pm.TruncatedNormal("c_f_hp", mu=16.5, sigma=2.0, lower=6.0, upper=30.0)
                    c_m_hp = pm.TruncatedNormal("c_m_hp", mu=16.0, sigma=2.0, lower=6.0, upper=30.0)
                pm.Deterministic("c_female_hp", c_f_hp)
                pm.Deterministic("c_male_hp", c_m_hp)

            # Convenience alias for single-marker models
            if self.marker == "alp":
                pm.Deterministic("c_female", c_f_alp)
                pm.Deterministic("c_male", c_m_alp)
            elif self.marker == "hp":
                pm.Deterministic("c_female", c_f_hp)
                pm.Deterministic("c_male", c_m_hp)

            # ------------------------------------------------------------------
            # 2. Infancy Surge Kernel K_inf(t) - Weibull Decay
            # ------------------------------------------------------------------
            lambda_inf = pm.Gamma("lambda_inf", alpha=4.0, beta=3.0)
            gamma_inf = pm.Gamma("gamma_inf", alpha=4.0, beta=4.0)

            inf_shape = pm.math.exp(-((age_data / lambda_inf) ** gamma_inf))

            if self.marker in ("alp", "joint"):
                A_inf_alp = pm.HalfNormal("A_inf_alp", sigma=120.0)
                K_inf_alp = A_inf_alp * inf_shape

            if self.marker in ("hp", "joint"):
                A_inf_hp = pm.HalfNormal("A_inf_hp", sigma=160.0)
                K_inf_hp = A_inf_hp * inf_shape

            # ------------------------------------------------------------------
            # 3. Pubertal Growth Spurt Kernel K_pub(t, s) - Sex-Dimorphic Gaussian
            # ------------------------------------------------------------------
            if self.sex_model == "hierarchical":
                mu_pub_base = pm.TruncatedNormal(
                    "mu_pub_base", mu=11.5, sigma=0.5, lower=10.0, upper=13.0
                )
                mu_pub_shift = pm.TruncatedNormal(
                    "mu_pub_shift", mu=2.3, sigma=0.4, lower=0.5, upper=4.0
                )
                mu_pub_f = pm.Deterministic("mu_pub_female", mu_pub_base)
                mu_pub_m = pm.Deterministic("mu_pub_male", mu_pub_base + mu_pub_shift)
            else:
                mu_pub_f = pm.TruncatedNormal(
                    "mu_pub_female", mu=11.5, sigma=0.5, lower=10.0, upper=13.0
                )
                mu_pub_m = pm.TruncatedNormal(
                    "mu_pub_male", mu=13.8, sigma=0.5, lower=12.0, upper=15.5
                )

            pm.Deterministic("mu_pub_f", mu_pub_f)
            pm.Deterministic("mu_pub_m", mu_pub_m)

            sigma_pub_f = pm.TruncatedNormal(
                "sigma_pub_female", mu=1.5, sigma=0.3, lower=0.8, upper=3.0
            )
            sigma_pub_m = pm.TruncatedNormal(
                "sigma_pub_male", mu=1.6, sigma=0.3, lower=0.8, upper=3.0
            )
            pm.Deterministic("sigma_pub_f", sigma_pub_f)
            pm.Deterministic("sigma_pub_m", sigma_pub_m)

            pub_f_shape = pm.math.exp(-((age_data - mu_pub_f) ** 2) / (2.0 * sigma_pub_f**2))
            pub_m_shape = pm.math.exp(-((age_data - mu_pub_m) ** 2) / (2.0 * sigma_pub_m**2))

            if self.marker in ("alp", "joint"):
                A_pub_f_alp = pm.HalfNormal("A_pub_f_alp", sigma=90.0)
                A_pub_m_alp = pm.HalfNormal("A_pub_m_alp", sigma=110.0)
                pm.Deterministic("A_pub_female_alp", A_pub_f_alp)
                pm.Deterministic("A_pub_male_alp", A_pub_m_alp)
                K_pub_alp = (
                    is_f_data * A_pub_f_alp * pub_f_shape + is_m_data * A_pub_m_alp * pub_m_shape
                )

            if self.marker in ("hp", "joint"):
                A_pub_f_hp = pm.HalfNormal("A_pub_f_hp", sigma=80.0)
                A_pub_m_hp = pm.HalfNormal("A_pub_m_hp", sigma=90.0)
                pm.Deterministic("A_pub_female_hp", A_pub_f_hp)
                pm.Deterministic("A_pub_male_hp", A_pub_m_hp)
                K_pub_hp = (
                    is_f_data * A_pub_f_hp * pub_f_shape + is_m_data * A_pub_m_hp * pub_m_shape
                )

            # ------------------------------------------------------------------
            # 4. Female Postmenopausal Kernel K_meno(t, s) (K=3, 4)
            # ------------------------------------------------------------------
            if self.n_components >= 3:
                mu_meno = pm.TruncatedNormal("mu_meno", mu=54.0, sigma=2.0, lower=48.0, upper=62.0)
                sigma_meno = pm.TruncatedNormal(
                    "sigma_meno", mu=4.5, sigma=1.0, lower=2.0, upper=8.0
                )
                meno_shape = pm.math.exp(-((age_data - mu_meno) ** 2) / (2.0 * sigma_meno**2))

                if self.marker in ("alp", "joint"):
                    A_meno_alp = pm.HalfNormal("A_meno_alp", sigma=5.0)
                    K_meno_alp = is_f_data * A_meno_alp * meno_shape

                if self.marker in ("hp", "joint"):
                    A_meno_hp = pm.HalfNormal("A_meno_hp", sigma=6.0)
                    K_meno_hp = is_f_data * A_meno_hp * meno_shape

            # ------------------------------------------------------------------
            # 5. Mid-Childhood Adrenarche Kernel K_mid(t) (K=4 only)
            # ------------------------------------------------------------------
            if self.n_components == 4:
                mu_mid = pm.TruncatedNormal("mu_mid", mu=7.0, sigma=0.4, lower=5.5, upper=8.5)
                sigma_mid = pm.TruncatedNormal("sigma_mid", mu=0.8, sigma=0.2, lower=0.3, upper=2.0)
                mid_shape = pm.math.exp(-((age_data - mu_mid) ** 2) / (2.0 * sigma_mid**2))

                if self.marker in ("alp", "joint"):
                    A_mid_alp = pm.HalfNormal("A_mid_alp", sigma=15.0)
                    K_mid_alp = A_mid_alp * mid_shape

                if self.marker in ("hp", "joint"):
                    A_mid_hp = pm.HalfNormal("A_mid_hp", sigma=20.0)
                    K_mid_hp = A_mid_hp * mid_shape

            # ------------------------------------------------------------------
            # Observation Error Scales
            # ------------------------------------------------------------------
            if self.marker in ("alp", "joint"):
                sigma_log_alp = pm.Exponential("sigma_log_alp", lam=3.0)
            if self.marker in ("hp", "joint"):
                sigma_log_hp = pm.Exponential("sigma_log_hp", lam=3.0)

            # ------------------------------------------------------------------
            # Total Trajectory Composition
            # ------------------------------------------------------------------
            if self.marker in ("alp", "joint"):
                c_alp_i = is_f_data * c_f_alp + is_m_data * c_m_alp
                mu_alp = c_alp_i + K_inf_alp + K_pub_alp
                if self.n_components >= 3:
                    mu_alp = mu_alp + K_meno_alp
                if self.n_components == 4:
                    mu_alp = mu_alp + K_mid_alp
                pm.Deterministic("mu_alp", mu_alp, dims="obs_id")

            if self.marker in ("hp", "joint"):
                c_hp_i = is_f_data * c_f_hp + is_m_data * c_m_hp
                mu_hp = c_hp_i + K_inf_hp + K_pub_hp
                if self.n_components >= 3:
                    mu_hp = mu_hp + K_meno_hp
                if self.n_components == 4:
                    mu_hp = mu_hp + K_mid_hp
                pm.Deterministic("mu_hp", mu_hp, dims="obs_id")

            # Selection per observation
            if is_joint:
                is_alp_data = pm.Data("is_alp_data", is_alp_obs, dims="obs_id")
                is_hp_data = pm.Data("is_hp_data", is_hp_obs, dims="obs_id")
                mu = is_alp_data * mu_alp + is_hp_data * mu_hp
                sigma_log = is_alp_data * sigma_log_alp + is_hp_data * sigma_log_hp
            elif self.marker == "alp":
                mu = mu_alp
                sigma_log = sigma_log_alp
            else:
                mu = mu_hp
                sigma_log = sigma_log_hp

            pm.Deterministic("mu", mu, dims="obs_id")

            # ------------------------------------------------------------------
            # Robust Student-t Likelihood on Natural Log Scale (nu=4.0)
            # ------------------------------------------------------------------
            pm.StudentT(
                "obs",
                nu=4.0,
                mu=pm.math.log(mu),
                sigma=sigma_log,
                observed=np.log(y_obs),
                dims="obs_id",
            )

        self.model = model
        return model

    def sample_prior_predictive(
        self,
        draws: int = 500,
        random_seed: int = RANDOM_SEED,
        data: pd.DataFrame | None = None,
    ) -> azb.InferenceData:
        """Sample from the prior predictive distribution."""
        if self.model is None or data is not None:
            self.build(data=data)

        assert self.model is not None
        with self.model:
            prior = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)
        return prior

    def fit(
        self,
        data: pd.DataFrame | None = None,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        random_seed: int = RANDOM_SEED,
        sampler: Literal["auto", "nutpie", "pymc"] = "auto",
    ) -> Any:
        """Fit the biomarker model using MCMC sampling.

        Prefers the compiled Rust `nutpie` sampler for speed and stability,
        falling back to `pm.sample` if nutpie is unavailable.
        """
        self.build(data=data)
        assert self.model is not None

        idata: Any | None = None

        if sampler in ("auto", "nutpie"):
            try:
                import nutpie

                compiled = nutpie.compile_pymc_model(self.model)
                idata = nutpie.sample(
                    compiled,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    seed=random_seed,
                )
                self.sampler_used = "nutpie"
                logger.info("Sampled BiomarkerModel using compiled nutpie Rust NUTS engine.")
            except Exception as exc:
                if sampler == "nutpie":
                    raise
                logger.warning(
                    "nutpie sampling failed (%s); falling back to pm.sample",
                    exc,
                )

        if idata is None:
            with self.model:
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    progressbar=False,
                )
            self.sampler_used = "pymc"
            logger.info("Sampled BiomarkerModel using PyMC default NUTS engine.")

        # Attach log-likelihood for LOO/WAIC model comparison
        with self.model:
            try:
                pm.compute_log_likelihood(idata)
            except Exception as exc:
                logger.debug("Could not compute log_likelihood: %s", exc)

        # Attach posterior predictive draws
        with self.model:
            try:
                post_pred = pm.sample_posterior_predictive(
                    idata,
                    random_seed=random_seed,
                    progressbar=False,
                )
                if hasattr(idata, "extend"):
                    idata.extend(post_pred)
            except Exception as exc:
                logger.debug("Could not sample posterior predictive: %s", exc)

        self.idata = idata
        return idata

    def _extract_posterior_draws(self) -> dict[str, np.ndarray]:
        """Extract and flatten posterior draws across chains and iterations."""
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        post = getattr(self.idata, "posterior", None)
        if post is None:
            raise RuntimeError("InferenceData does not contain a 'posterior' group")

        draws: dict[str, np.ndarray] = {}
        for var_name in post.data_vars:
            arr = np.asarray(post[var_name].values)
            draws[var_name] = arr.reshape(-1)
        return draws

    def _predict_mu_samples(
        self,
        age_arr: np.ndarray,
        sex: str,
        marker: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate trajectory mu(t, s) and adult baseline c(s) across posterior draws.

        Parameters
        ----------
        age_arr : np.ndarray
            Array of shape (1, K) of evaluation ages.
        sex : str
            Normalized biological sex ('F' or 'M').
        marker : str
            Normalized biomarker name ('alp' or 'hp').

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mu_samples, c_samples) arrays of shape (n_samples, K) and (n_samples, 1).
        """
        draws = self._extract_posterior_draws()
        n_samples = next(iter(draws.values())).shape[0]

        is_female = sex == "F"

        # Baseline resolution
        c_key = f"c_{'f' if is_female else 'm'}_{marker}"
        if c_key in draws:
            c_samples = draws[c_key].reshape(-1, 1)
        elif f"c_{'female' if is_female else 'male'}_{marker}" in draws:
            c_samples = draws[f"c_{'female' if is_female else 'male'}_{marker}"].reshape(-1, 1)
        elif f"c_{'female' if is_female else 'male'}" in draws:
            c_samples = draws[f"c_{'female' if is_female else 'male'}"].reshape(-1, 1)
        elif f"c_{'f' if is_female else 'm'}" in draws:
            c_samples = draws[f"c_{'f' if is_female else 'm'}"].reshape(-1, 1)
        else:
            default_c = (
                (10.0 if is_female else 10.5) if marker == "alp" else (16.5 if is_female else 16.0)
            )
            c_samples = np.full((n_samples, 1), default_c)

        # Infancy kernel
        lambda_inf = draws.get("lambda_inf", np.full(n_samples, 1.33)).reshape(-1, 1)
        gamma_inf = draws.get("gamma_inf", np.full(n_samples, 1.0)).reshape(-1, 1)
        A_inf_key = f"A_inf_{marker}"
        if A_inf_key in draws:
            A_inf = draws[A_inf_key].reshape(-1, 1)
        elif "A_inf" in draws:
            A_inf = draws["A_inf"].reshape(-1, 1)
        else:
            A_inf = np.full((n_samples, 1), 110.0 if marker == "alp" else 160.0)

        inf_exponent = (age_arr / lambda_inf) ** gamma_inf
        inf_exponent = np.clip(inf_exponent, 0.0, 50.0)
        K_inf = A_inf * np.exp(-inf_exponent)

        # Puberty kernel
        mu_pub_key = f"mu_pub_{'female' if is_female else 'male'}"
        if mu_pub_key not in draws:
            mu_pub_key = f"mu_pub_{'f' if is_female else 'm'}"
        mu_pub = draws.get(mu_pub_key, np.full(n_samples, 11.5 if is_female else 13.8)).reshape(
            -1, 1
        )

        sigma_pub_key = f"sigma_pub_{'female' if is_female else 'male'}"
        if sigma_pub_key not in draws:
            sigma_pub_key = f"sigma_pub_{'f' if is_female else 'm'}"
        sigma_pub = draws.get(sigma_pub_key, np.full(n_samples, 1.5 if is_female else 1.6)).reshape(
            -1, 1
        )

        A_pub_key = f"A_pub_{'f' if is_female else 'm'}_{marker}"
        if A_pub_key in draws:
            A_pub = draws[A_pub_key].reshape(-1, 1)
        elif f"A_pub_{'female' if is_female else 'male'}_{marker}" in draws:
            A_pub = draws[f"A_pub_{'female' if is_female else 'male'}_{marker}"].reshape(-1, 1)
        elif f"A_pub_{'female' if is_female else 'male'}" in draws:
            A_pub = draws[f"A_pub_{'female' if is_female else 'male'}"].reshape(-1, 1)
        elif "A_pub" in draws:
            A_pub = draws["A_pub"].reshape(-1, 1)
        else:
            A_pub = np.full((n_samples, 1), 85.0 if marker == "alp" else 85.0)

        pub_exponent = ((age_arr - mu_pub) ** 2) / (2.0 * sigma_pub**2)
        pub_exponent = np.clip(pub_exponent, 0.0, 50.0)
        K_pub = A_pub * np.exp(-pub_exponent)

        # Menopause kernel (K>=3, female only)
        if self.n_components >= 3 and is_female:
            mu_meno = draws.get("mu_meno", np.full(n_samples, 54.0)).reshape(-1, 1)
            sigma_meno = draws.get("sigma_meno", np.full(n_samples, 4.5)).reshape(-1, 1)
            A_meno_key = f"A_meno_{marker}"
            if A_meno_key in draws:
                A_meno = draws[A_meno_key].reshape(-1, 1)
            elif "A_meno" in draws:
                A_meno = draws["A_meno"].reshape(-1, 1)
            else:
                A_meno = np.full((n_samples, 1), 2.0 if marker == "alp" else 2.5)

            meno_exponent = ((age_arr - mu_meno) ** 2) / (2.0 * sigma_meno**2)
            meno_exponent = np.clip(meno_exponent, 0.0, 50.0)
            K_meno = A_meno * np.exp(-meno_exponent)
        else:
            K_meno = np.zeros((n_samples, age_arr.shape[1]))

        # Mid-childhood adrenarche kernel (K=4 only)
        if self.n_components == 4:
            mu_mid = draws.get("mu_mid", np.full(n_samples, 7.0)).reshape(-1, 1)
            sigma_mid = draws.get("sigma_mid", np.full(n_samples, 0.8)).reshape(-1, 1)
            A_mid_key = f"A_mid_{marker}"
            if A_mid_key in draws:
                A_mid = draws[A_mid_key].reshape(-1, 1)
            elif "A_mid" in draws:
                A_mid = draws["A_mid"].reshape(-1, 1)
            else:
                A_mid = np.full((n_samples, 1), 5.0)

            mid_exponent = ((age_arr - mu_mid) ** 2) / (2.0 * sigma_mid**2)
            mid_exponent = np.clip(mid_exponent, 0.0, 50.0)
            K_mid = A_mid * np.exp(-mid_exponent)
        else:
            K_mid = np.zeros((n_samples, age_arr.shape[1]))

        mu_samples = c_samples + K_inf + K_pub + K_meno + K_mid
        return mu_samples, c_samples

    def relative_activity_samples(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: str,
        marker: str,
    ) -> np.ndarray:
        """Compute posterior trajectory draws of Relative Activity Index (RAI).

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Chronological age [years]. Must be strictly positive and finite.
        sex : str
            Biological sex ('F' or 'M').
        marker : str
            Biomarker ('alp' or 'hp').

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_points) of RAI values.
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        s_norm = normalize_sex(sex)
        m_norm = marker.strip().lower()
        if m_norm not in ("alp", "hp"):
            raise ValueError(f"Invalid marker='{marker}'. Expected 'alp' or 'hp'.")
        if self.marker != "joint" and self.marker != m_norm:
            raise ValueError(
                f"Model was fitted for marker='{self.marker}', cannot evaluate RAI for '{m_norm}'."
            )

        arr = np.atleast_1d(np.asarray(age, dtype=float)).ravel()
        if not np.all(np.isfinite(arr)):
            raise ValueError("All age values must be finite")
        if np.any(arr <= 0):
            raise ValueError("All age values must be strictly positive")

        age_eval = arr.reshape(1, -1)
        mu_samples, c_samples = self._predict_mu_samples(age_eval, sex=s_norm, marker=m_norm)
        return mu_samples / c_samples

    def relative_activity_index(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: str,
        marker: str,
    ) -> float | np.ndarray:
        """Calculate the adult-baseline normalized Relative Activity Index (RAI).

        Returns RAI_ALP (formation rate proxy r_form) or RAI_HP (resorption
        rate proxy r_resorp), normalized to sex-specific adult baseline c(s).

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Age in years (scalar or array).
        sex : str
            Biological sex ('F' or 'M', case-insensitive).
        marker : str
            Marker name ('alp' or 'hp', case-insensitive).

        Returns
        -------
        float or np.ndarray
            Relative Activity Index. Returns float for scalar input,
            or np.ndarray matching input array shape.
        """
        is_scalar = isinstance(age, (int, float, np.floating, np.integer))
        rai_samples = self.relative_activity_samples(age, sex=sex, marker=marker)
        rai_mean = np.mean(rai_samples, axis=0)

        if is_scalar:
            return float(rai_mean[0])

        arr_input = np.asarray(age)
        return rai_mean.reshape(arr_input.shape)

    def relative_activity_grid(
        self,
        age_min: float = 0.0,
        age_max: float = 85.0,
        step: float = 0.5,
        sex: str = "F",
        marker: str = "alp",
        prob: float = DEFAULT_HDI_PROB,
    ) -> pd.DataFrame:
        """Generate a dense lookup grid of RAI across age with 94% HDI.

        Parameters
        ----------
        age_min : float, default=0.0
            Minimum age [years]. Clamped to >= 0.01 yr.
        age_max : float, default=85.0
            Maximum age [years].
        step : float, default=0.5
            Age grid step size [years].
        sex : str, default='F'
            Biological sex ('F' or 'M').
        marker : str, default='alp'
            Biomarker ('alp' or 'hp').
        prob : float, default=0.94
            Credible interval probability mass.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: age, sex, marker, mean, median, std, hdi_lower, hdi_upper.
        """
        if step <= 0:
            raise ValueError("step must be strictly positive")
        if age_max <= age_min:
            raise ValueError("age_max must be strictly greater than age_min")

        ages = np.arange(age_min, age_max + 0.5 * step, step)
        ages[0] = max(ages[0], 0.01)

        s_norm = normalize_sex(sex)
        m_norm = marker.strip().lower()

        samples = self.relative_activity_samples(ages, sex=s_norm, marker=m_norm)
        mean_vals = np.mean(samples, axis=0)
        median_vals = np.median(samples, axis=0)
        std_vals = np.std(samples, axis=0)
        hdi_lower, hdi_upper = _compute_hdi_bounds(samples, prob=prob)

        return pd.DataFrame(
            {
                "age": ages,
                "sex": s_norm,
                "marker": m_norm,
                "mean": mean_vals,
                "median": median_vals,
                "std": std_vals,
                "hdi_lower": hdi_lower,
                "hdi_upper": hdi_upper,
            }
        )

    def predict(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: str,
        marker: str,
        random_seed: int = RANDOM_SEED,
    ) -> dict[str, np.ndarray]:
        """Generate posterior trajectory draws and simulated observation draws.

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Ages to evaluate.
        sex : str
            Biological sex ('F' or 'M').
        marker : str
            Biomarker ('alp' or 'hp').
        random_seed : int, default=42
            Seed for residual draws.

        Returns
        -------
        dict[str, np.ndarray]
            - 'age': 1D array of ages.
            - 'mu': Trajectory draws (n_samples, n_points).
            - 'y_obs': Simulated noisy observations (n_samples, n_points).
            - 'rai': Relative Activity Index draws (n_samples, n_points).
        """
        s_norm = normalize_sex(sex)
        m_norm = marker.strip().lower()

        arr = np.atleast_1d(np.asarray(age, dtype=float)).ravel()
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError("All age values must be strictly positive and finite")

        age_eval = arr.reshape(1, -1)
        mu_samples, c_samples = self._predict_mu_samples(age_eval, sex=s_norm, marker=m_norm)
        rai_samples = mu_samples / c_samples

        draws = self._extract_posterior_draws()
        sigma_key = f"sigma_log_{m_norm}" if f"sigma_log_{m_norm}" in draws else "sigma_log"
        sigma_log = draws.get(sigma_key, np.full((mu_samples.shape[0],), 0.25)).reshape(-1, 1)

        rng = np.random.default_rng(random_seed)
        t_residuals = rng.standard_t(df=4.0, size=mu_samples.shape)
        y_obs_samples = np.exp(np.log(mu_samples) + sigma_log * t_residuals)

        return {
            "age": arr,
            "mu": mu_samples,
            "y_obs": y_obs_samples,
            "rai": rai_samples,
        }

    def summary(self, ci_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
        """Return posterior parameter summary table using arviz_stats."""
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        return azs.summary(self.idata, ci_prob=ci_prob, ci_kind="hdi")

    def diagnostics(self) -> dict[str, Any]:
        """Run MCMC convergence diagnostics (R-hat, ESS, divergences)."""
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        sm = self.summary()
        rhat_col = "r_hat" if "r_hat" in sm.columns else "rhat"
        max_rhat = float(sm[rhat_col].max()) if rhat_col in sm.columns else 1.0

        ess_bulk_col = "ess_bulk" if "ess_bulk" in sm.columns else "ess"
        min_ess_bulk = float(sm[ess_bulk_col].min()) if ess_bulk_col in sm.columns else 1000.0

        ess_tail_col = "ess_tail" if "ess_tail" in sm.columns else ess_bulk_col
        min_ess_tail = float(sm[ess_tail_col].min()) if ess_tail_col in sm.columns else 1000.0

        divergences = 0
        sample_stats = getattr(self.idata, "sample_stats", None)
        if sample_stats is not None and "diverging" in sample_stats:
            divergences = int(np.asarray(sample_stats["diverging"].values).sum())

        rhat_pass = bool(max_rhat <= 1.01)
        ess_pass = bool(min_ess_bulk >= 400 and min_ess_tail >= 400)
        divergences_pass = bool(divergences == 0)

        return {
            "converged": bool(rhat_pass and ess_pass and divergences_pass),
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "divergences": divergences,
            "rhat_pass": rhat_pass,
            "ess_pass": ess_pass,
            "divergences_pass": divergences_pass,
            "sampler": self.sampler_used,
        }

    def compare_components(
        self,
        data: pd.DataFrame | None = None,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        target_accept: float = 0.95,
        random_seed: int = RANDOM_SEED,
        models: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compare K=2, K=3, and K=4 models using PSIS-LOO cross-validation.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Dataset to fit models on. Defaults to cached or bundled Stepan data.
        draws : int, default=500
            Draws per chain when fitting missing models.
        tune : int, default=500
            Warmup tuning steps when fitting missing models.
        chains : int, default=2
            Chains when fitting missing models.
        target_accept : float, default=0.95
            Target acceptance rate.
        random_seed : int, default=42
            Random seed for reproducibility.
        models : dict[str, Any] | None, optional
            Dictionary of pre-fitted BiomarkerModel instances or InferenceData objects.
            If provided, skips MCMC sampling and directly runs azs.compare.

        Returns
        -------
        pd.DataFrame
            Model comparison table with elpd_diff (d_elpd), elpd (elpd_loo), p (p_loo), weight.
        """
        if models is not None:
            comp_dict = {
                name: (m.idata if hasattr(m, "idata") and m.idata is not None else m)
                for name, m in models.items()
            }
        else:
            comp_dict = {}
            for k in (2, 3, 4):
                logger.info("Fitting K=%d model for component comparison...", k)
                m = BiomarkerModel(
                    n_components=k,
                    marker=self.marker,  # type: ignore[arg-type]
                    sex_model=self.sex_model,  # type: ignore[arg-type]
                    data=data,
                )
                m.fit(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=random_seed,
                )
                comp_dict[f"K={k}"] = m.idata

        res = azs.compare(comp_dict)
        # Ensure friendly aliases for standard column names
        if "elpd" in res.columns and "elpd_loo" not in res.columns:
            res["elpd_loo"] = res["elpd"]
        if "p" in res.columns and "p_loo" not in res.columns:
            res["p_loo"] = res["p"]
        if "elpd_diff" in res.columns and "d_elpd" not in res.columns:
            res["d_elpd"] = res["elpd_diff"]

        return res

    def plot_components(
        self,
        marker: str = "alp",
        sex: str = "F",
        age_grid: np.ndarray | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot deconvolution components and total trajectory."""
        import matplotlib.pyplot as plt

        s_norm = normalize_sex(sex)
        m_norm = marker.strip().lower()

        if age_grid is None:
            age_grid = np.linspace(0.1, 80.0, 300)

        mu_samples, c_samples = self._predict_mu_samples(
            age_grid.reshape(1, -1), sex=s_norm, marker=m_norm
        )
        mean_mu = np.mean(mu_samples, axis=0)
        hdi_lo, hdi_hi = _compute_hdi_bounds(mu_samples)
        mean_c = float(np.mean(c_samples))

        if ax is None:
            fig, ax_plot = plt.subplots(figsize=(8, 5))
        else:
            ax_plot = ax
            fig = ax_plot.get_figure()

        # Observations
        if self.data is not None:
            sub = self.data[(self.data["marker"] == m_norm) & (self.data["sex"] == s_norm)]
            ax_plot.scatter(
                sub["age"],
                sub["value"],
                alpha=0.25,
                color="gray",
                s=12,
                label=f"Stepan Data (N={len(sub)})",
            )

        # Total trajectory and adult baseline
        ax_plot.plot(
            age_grid, mean_mu, color="black", lw=2, label=f"Total Trajectory mu(t, {s_norm})"
        )
        ax_plot.fill_between(age_grid, hdi_lo, hdi_hi, color="black", alpha=0.15, label="94% HDI")
        ax_plot.axhline(
            mean_c, color="crimson", ls="--", label=f"Adult Baseline c({s_norm}) = {mean_c:.1f}"
        )

        unit_str = "U/L" if m_norm == "alp" else "mmol/mol"
        ax_plot.set_xlabel("Age [years]")
        ax_plot.set_ylabel(f"{m_norm.upper()} Concentration [{unit_str}]")
        ax_plot.set_title(f"Biomarker Spectral Deconvolution: {m_norm.upper()} ({s_norm})")
        ax_plot.legend(loc="best", fontsize=8)
        ax_plot.grid(True, alpha=0.3)

        return fig

    def plot_relative_activity(
        self,
        marker: str = "alp",
        age_grid: np.ndarray | None = None,
    ) -> plt.Figure:
        """Plot Relative Activity Index (RAI) trajectories across sexes."""
        import matplotlib.pyplot as plt

        m_norm = marker.strip().lower()
        if age_grid is None:
            age_grid = np.linspace(0.1, 80.0, 300)

        grid_f = self.relative_activity_grid(
            age_min=float(age_grid[0]),
            age_max=float(age_grid[-1]),
            step=0.25,
            sex="F",
            marker=m_norm,
        )
        grid_m = self.relative_activity_grid(
            age_min=float(age_grid[0]),
            age_max=float(age_grid[-1]),
            step=0.25,
            sex="M",
            marker=m_norm,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grid_f["age"], grid_f["mean"], color="magenta", lw=2, label="Female RAI")
        ax.fill_between(
            grid_f["age"], grid_f["hdi_lower"], grid_f["hdi_upper"], color="magenta", alpha=0.15
        )

        ax.plot(grid_m["age"], grid_m["mean"], color="blue", lw=2, label="Male RAI")
        ax.fill_between(
            grid_m["age"], grid_m["hdi_lower"], grid_m["hdi_upper"], color="blue", alpha=0.15
        )

        ax.axhline(1.0, color="gray", ls="--", label="Adult Baseline (1.00)")
        ax.set_xlabel("Age [years]")
        ax.set_ylabel(f"Relative Activity Index RAI_{m_norm.upper()}")
        ax.set_title(f"Lifespan Bone Turnover Modifier Trajectory (RAI_{m_norm.upper()})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return fig

    def save(self, filepath: str | Path) -> None:
        """Save fitted InferenceData to a NetCDF file via h5netcdf engine.

        Parameters
        ----------
        filepath : str or Path
            Destination .nc file path.
        """
        if self.idata is None:
            raise RuntimeError("No fitted InferenceData to save. Call fit() first.")

        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.idata, "attrs"):
            self.idata.attrs["n_components"] = self.n_components
            self.idata.attrs["marker"] = self.marker
            self.idata.attrs["sex_model"] = self.sex_model

        if hasattr(self.idata, "to_netcdf"):
            self.idata.to_netcdf(str(target), engine="h5netcdf")
        else:
            raise RuntimeError("InferenceData object does not have to_netcdf method.")

        logger.info("Saved BiomarkerModel InferenceData to %s", target)

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        n_components: int | None = None,
        marker: MarkerType | None = None,
        sex_model: SexModelType | None = None,
        data: pd.DataFrame | None = None,
    ) -> BiomarkerModel:
        """Load InferenceData from a NetCDF file via h5netcdf and return a BiomarkerModel.

        Parameters
        ----------
        filepath : str or Path
            Path to existing NetCDF file.
        n_components : int, optional
            Number of components. If None, restored from file attributes.
        marker : str, optional
            Marker name. If None, restored from file attributes.
        sex_model : str, optional
            Sex model type. If None, restored from file attributes.
        data : pd.DataFrame | None, optional
            Optional dataset for model context.

        Returns
        -------
        BiomarkerModel
            Configured BiomarkerModel instance with loaded posterior.
        """
        target = Path(filepath)
        if not target.is_file():
            raise FileNotFoundError(f"NetCDF file not found: {target}")

        loaded = xr.load_datatree(str(target), engine="h5netcdf")

        attrs = getattr(loaded, "attrs", {})
        res_n_comp = n_components if n_components is not None else int(attrs.get("n_components", 3))
        res_marker = marker if marker is not None else str(attrs.get("marker", "joint"))
        res_sex_model = (
            sex_model if sex_model is not None else str(attrs.get("sex_model", "hierarchical"))
        )

        instance = cls(
            n_components=res_n_comp,
            marker=res_marker,  # type: ignore[arg-type]
            sex_model=res_sex_model,  # type: ignore[arg-type]
            data=data,
        )
        instance.idata = loaded
        logger.info("Loaded BiomarkerModel InferenceData from %s", target)
        return instance
