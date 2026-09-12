"""Bone Turnover Suppression Model (r_TO) for bone-predictor.

Implements the concentration-dependent bone turnover suppression model fitted
to Boivin et al. histomorphometric cancellous bone formation rate data.

Mathematical Formulations Supported:
1. 4-Parameter Hill Model (default, '4p_hill'):
   mu(C) = BFR_floor + (BFR_0 - BFR_floor) / (1 + exp[n_hill * (ln(C) - eta_Km)])
   where eta_Km = ln(K_m), BFR_floor = BFR_0 * R_floor
   Priors:
     BFR_0 ~ Normal(0.076, 0.008)
     R_floor ~ Beta(4.0, 8.0)
     eta_Km ~ Normal(ln(3500.0), 0.40)
     n_hill ~ Gamma(3.0, 1.5)
     sigma_log ~ Exponential(3.0)
   Likelihood:
     ln(y) ~ StudentT(nu=4.0, mu=ln(mu), sigma=sigma_log)

2. 2-Parameter Hill Model ('hill_2p', zero floor):
   mu(C) = BFR_0 / (1 + exp[n_hill * (ln(C) - eta_Km)])

3. Log-Linear Model ('log_linear'):
   mu(C) = a - b * ln(C)

4. Brain-Cousens Hormesis Model ('hormesis'):
   mu(C) = BFR_floor + (BFR_0 - BFR_floor + f * C) / (1 + exp[n_hill * (ln(C) - eta_Km)])

5. Custom callable build_fn(data: pd.DataFrame) -> pm.Model.

Control-Normalized Turnover Modifier:
   r_TO(C) = [mu(C) / mu(C_control)] * 100% where C_control = 282.0 mg/kg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import arviz_base as azb
import arviz_stats as azs
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from bone_predictor.constants import C_CONTROL, DEFAULT_HDI_PROB, RANDOM_SEED
from bone_predictor.data import load_boivin_data

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

TurnoverFormulation = Literal["4p_hill", "log_linear", "hill_2p", "2p_hill", "hormesis", "custom"]
VALID_FORMULATIONS: set[str] = {"4p_hill", "log_linear", "hill_2p", "2p_hill", "hormesis", "custom"}


def _compute_hdi_bounds(
    samples_2d: np.ndarray, prob: float = DEFAULT_HDI_PROB
) -> tuple[np.ndarray, np.ndarray]:
    """Compute highest density interval bounds along sample axis (axis 0).

    Parameters
    ----------
    samples_2d : np.ndarray
        Array of shape (n_samples, n_points).
    prob : float, default=0.94
        Probability mass of the HDI.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (hdi_lower, hdi_upper) arrays of shape (n_points,).
    """
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


class TurnoverModel:
    """Concentration-dependent bone turnover suppression model (r_TO).

    Fits and evaluates the bone formation rate modifier as a function of
    trabecular bone fluoride concentration C_bone (mg/kg ash).

    Attributes
    ----------
    formulation : str
        Active mathematical formulation ('4p_hill', 'log_linear', 'hill_2p',
        'hormesis', or 'custom').
    build_fn : Callable[[pd.DataFrame], pm.Model] | None
        Optional custom model builder callable.
    c_control : float
        Baseline control concentration [mg/kg ash] (default: 282.0).
    data : pd.DataFrame | None
        Underlying histomorphometry dataset.
    model : pm.Model | None
        Compiled or active PyMC model.
    idata : Any | None
        Fitted InferenceData / DataTree containing posterior draws.
    sampler_used : str | None
        Sampling backend used during fit ('nutpie' or 'pymc').
    """

    def __init__(
        self,
        formulation: TurnoverFormulation = "4p_hill",
        build_fn: Callable[[pd.DataFrame], pm.Model] | None = None,
        data: pd.DataFrame | None = None,
        c_control: float = C_CONTROL,
    ) -> None:
        """Initialize TurnoverModel.

        Parameters
        ----------
        formulation : str, default='4p_hill'
            Formulation name ('4p_hill', 'log_linear', 'hill_2p', 'hormesis', 'custom').
        build_fn : Callable[[pd.DataFrame], pm.Model] | None, optional
            Custom function returning a pm.Model given a DataFrame.
        data : pd.DataFrame | None, optional
            Dataset with 'c_bone' and 'cn_bfr' columns. If None, loaded lazily.
        c_control : float, default=282.0
            Control concentration for r_TO normalization (mg/kg ash).
        """
        if build_fn is not None and formulation == "4p_hill":
            formulation = "custom"

        if formulation not in VALID_FORMULATIONS:
            raise ValueError(
                f"Unknown formulation '{formulation}'. "
                f"Supported formulations are: {sorted(VALID_FORMULATIONS)}"
            )

        if formulation == "custom" and build_fn is None:
            raise ValueError(
                "A callable `build_fn(data)` must be provided when formulation='custom'"
            )

        if formulation == "2p_hill":
            formulation = "hill_2p"

        self.formulation: str = formulation
        self.build_fn: Callable[[pd.DataFrame], pm.Model] | None = build_fn
        self.c_control: float = float(c_control)
        self.data: pd.DataFrame | None = self._validate_data(data) if data is not None else None
        self.model: pm.Model | None = None
        self.idata: Any | None = None
        self.sampler_used: str | None = None

    @staticmethod
    def _validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and validate histomorphometric DataFrame schema."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("Input data DataFrame cannot be empty")

        # Map possible column variations to canonical names
        col_map: dict[str, str] = {}
        for col in df.columns:
            c_str = str(col).strip()
            c_lower = c_str.lower()
            if "c_bone" in c_lower or c_lower == "c":
                col_map[col] = "c_bone"
            elif "cn-bfr/control" in c_lower or "cn_bfr_control" in c_lower:
                col_map[col] = "cn_bfr_control"
            elif "cn-bfr" in c_lower or "cn_bfr" in c_lower or "bfr" in c_lower:
                col_map[col] = "cn_bfr"

        renamed = df.rename(columns=col_map)
        required = {"c_bone", "cn_bfr"}
        if not required.issubset(renamed.columns):
            missing = required - set(renamed.columns)
            raise ValueError(f"Missing required columns in dataset: {missing}")

        clean_df = renamed.copy()
        clean_df["c_bone"] = pd.to_numeric(clean_df["c_bone"], errors="coerce")
        clean_df["cn_bfr"] = pd.to_numeric(clean_df["cn_bfr"], errors="coerce")

        if clean_df[["c_bone", "cn_bfr"]].isna().any().any():
            raise ValueError("Dataset contains non-numeric or NaN values in required columns")
        if (clean_df["c_bone"] <= 0).any():
            raise ValueError("Bone concentration 'c_bone' must be strictly positive")
        if (clean_df["cn_bfr"] <= 0).any():
            raise ValueError("Bone formation rate 'cn_bfr' must be strictly positive")

        return clean_df.sort_values(by="c_bone").reset_index(drop=True)

    def _get_data(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Resolve dataset, prioritizing passed data, cached data, or bundled Boivin."""
        if data is not None:
            return self._validate_data(data)
        if self.data is not None:
            return self.data
        loaded = load_boivin_data()
        self.data = loaded
        return loaded

    def build(self, data: pd.DataFrame | None = None) -> pm.Model:
        """Build the PyMC probabilistic model for the active formulation.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Data to build model on. Defaults to cached or bundled Boivin data.

        Returns
        -------
        pm.Model
            Configured PyMC model instance.
        """
        df = self._get_data(data)
        self.data = df

        if self.build_fn is not None or self.formulation == "custom":
            if self.build_fn is None:
                raise ValueError("build_fn must be provided for custom formulation")
            model = self.build_fn(df)
            self.model = model
            return model

        c_obs = df["c_bone"].to_numpy(dtype=float)
        y_obs = df["cn_bfr"].to_numpy(dtype=float)
        coords = {"obs_id": np.arange(len(df))}

        with pm.Model(coords=coords) as model:
            c_data = pm.Data("c_data", c_obs, dims="obs_id")
            log_c = pm.math.log(c_data)

            if self.formulation == "4p_hill":
                # Priors
                bfr0 = pm.Normal("bfr0", mu=0.076, sigma=0.008)
                r_floor = pm.Beta("r_floor", alpha=4.0, beta=8.0)
                eta_km = pm.Normal("eta_km", mu=np.log(3500.0), sigma=0.40)
                n_hill = pm.Gamma("n_hill", alpha=3.0, beta=1.5)
                sigma_log = pm.Exponential("sigma_log", lam=3.0)

                # Deterministics
                pm.Deterministic("km", pm.math.exp(eta_km))
                bfr_floor = pm.Deterministic("bfr_floor", bfr0 * r_floor)
                mu = pm.Deterministic(
                    "mu",
                    bfr_floor + (bfr0 - bfr_floor) / (1.0 + pm.math.exp(n_hill * (log_c - eta_km))),
                    dims="obs_id",
                )

            elif self.formulation == "hill_2p":
                # 2-Parameter Hill (zero floor)
                bfr0 = pm.Normal("bfr0", mu=0.076, sigma=0.008)
                eta_km = pm.Normal("eta_km", mu=np.log(3500.0), sigma=0.40)
                n_hill = pm.Gamma("n_hill", alpha=3.0, beta=1.5)
                sigma_log = pm.Exponential("sigma_log", lam=3.0)

                pm.Deterministic("km", pm.math.exp(eta_km))
                pm.Deterministic("bfr_floor", pt.as_tensor_variable(0.0))
                mu = pm.Deterministic(
                    "mu",
                    bfr0 / (1.0 + pm.math.exp(n_hill * (log_c - eta_km))),
                    dims="obs_id",
                )

            elif self.formulation == "log_linear":
                # 2-Parameter Log-Linear: mu(C) = a - b * ln(C)
                a = pm.Normal("a", mu=0.15, sigma=0.05)
                b = pm.HalfNormal("b", sigma=0.02)
                sigma_log = pm.Exponential("sigma_log", lam=3.0)

                mu = pm.Deterministic(
                    "mu",
                    pm.math.maximum(1e-5, a - b * log_c),
                    dims="obs_id",
                )

            elif self.formulation == "hormesis":
                # 5-Parameter Brain-Cousens Hormesis
                bfr0 = pm.Normal("bfr0", mu=0.076, sigma=0.008)
                r_floor = pm.Beta("r_floor", alpha=4.0, beta=8.0)
                eta_km = pm.Normal("eta_km", mu=np.log(3500.0), sigma=0.40)
                n_hill = pm.Gamma("n_hill", alpha=3.0, beta=1.5)
                f = pm.HalfNormal("f", sigma=1e-5)
                sigma_log = pm.Exponential("sigma_log", lam=3.0)

                pm.Deterministic("km", pm.math.exp(eta_km))
                bfr_floor = pm.Deterministic("bfr_floor", bfr0 * r_floor)
                mu = pm.Deterministic(
                    "mu",
                    bfr_floor
                    + (bfr0 - bfr_floor + f * c_data)
                    / (1.0 + pm.math.exp(n_hill * (log_c - eta_km))),
                    dims="obs_id",
                )

            # Robust Student-t Likelihood on natural log scale (nu=4.0)
            pm.StudentT(
                "bfr_obs",
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
        """Sample from the prior predictive distribution.

        Parameters
        ----------
        draws : int, default=500
            Number of prior draws.
        random_seed : int, default=42
            Seed for reproducible pseudo-random sampling.
        data : pd.DataFrame | None, optional
            Data to build model on if not already built.

        Returns
        -------
        azb.InferenceData
            InferenceData containing 'prior' and 'prior_predictive' groups.
        """
        if self.model is None or data is not None:
            self.build(data=data)

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
        """Fit the turnover model using NUTS MCMC sampling.

        Prefers the compiled Rust `nutpie` sampler for speed and stability,
        gracefully falling back to `pm.sample` if nutpie is unavailable.
        Attaches log-likelihood and posterior predictive samples.

        Parameters
        ----------
        data : pd.DataFrame | None, optional
            Dataset to fit. Defaults to bundled Boivin histomorphometry.
        draws : int, default=1000
            Posterior sample draws per chain.
        tune : int, default=1000
            Warmup / tuning iterations per chain.
        chains : int, default=4
            Number of Markov chains.
        target_accept : float, default=0.95
            Target acceptance probability.
        random_seed : int, default=42
            Random seed for reproducibility.
        sampler : Literal['auto', 'nutpie', 'pymc'], default='auto'
            Sampling engine selector.

        Returns
        -------
        Any
            Fitted InferenceData / DataTree with posterior, log_likelihood,
            and posterior_predictive groups.
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
                logger.info("Successfully sampled using compiled nutpie Rust NUTS engine.")
            except Exception as exc:
                if sampler == "nutpie":
                    raise
                logger.warning(
                    "nutpie sampling failed with error (%s); falling back to pm.sample",
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
            logger.info("Sampled using PyMC default NUTS engine.")

        # Compute log-likelihood for LOO/WAIC model comparison
        with self.model:
            try:
                pm.compute_log_likelihood(idata)
            except Exception as exc:
                logger.debug("Could not compute log_likelihood: %s", exc)

        # Sample posterior predictive for model criticism
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

    def sample_posterior_predictive(
        self,
        draws: int | None = None,
        random_seed: int = RANDOM_SEED,
        extend_inferencedata: bool = True,
    ) -> azb.InferenceData:
        """Sample posterior predictive observations from existing posterior draws.

        Parameters
        ----------
        draws : int | None, optional
            Number of predictive draws to generate.
        random_seed : int, default=42
            Random seed for reproducibility.
        extend_inferencedata : bool, default=True
            Whether to attach posterior_predictive group into self.idata.

        Returns
        -------
        azb.InferenceData
            InferenceData containing 'posterior_predictive'.
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")
        if self.model is None:
            self.build()

        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata,
                random_seed=random_seed,
                progressbar=False,
            )
            if extend_inferencedata and hasattr(self.idata, "extend"):
                self.idata.extend(post_pred)
        return post_pred

    def _extract_posterior_draws(self) -> dict[str, np.ndarray]:
        """Extract and flatten parameter arrays across chains and draws."""
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

    def _predict_mu_samples(self, c_arr: np.ndarray) -> np.ndarray:
        """Evaluate formation rate mu(C) across all posterior draws.

        Parameters
        ----------
        c_arr : np.ndarray
            Array of shape (1, K) of concentration values.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, K) of formation rates.
        """
        draws = self._extract_posterior_draws()
        n_samples = next(iter(draws.values())).shape[0]

        if self.formulation in ("4p_hill", "custom"):
            if "bfr0" in draws and ("eta_km" in draws or "km" in draws) and "n_hill" in draws:
                bfr0 = draws["bfr0"].reshape(-1, 1)
                if "bfr_floor" in draws:
                    bfr_floor = draws["bfr_floor"].reshape(-1, 1)
                elif "r_floor" in draws:
                    bfr_floor = bfr0 * draws["r_floor"].reshape(-1, 1)
                else:
                    bfr_floor = np.zeros((n_samples, 1))

                if "eta_km" in draws:
                    eta_km = draws["eta_km"].reshape(-1, 1)
                else:
                    eta_km = np.log(draws["km"].reshape(-1, 1))

                n_hill = draws["n_hill"].reshape(-1, 1)
                log_c = np.log(c_arr)  # shape (1, K)
                exponent = n_hill * (log_c - eta_km)
                # Numerical clipping for exp overflow
                exponent = np.clip(exponent, -50.0, 50.0)
                return bfr_floor + (bfr0 - bfr_floor) / (1.0 + np.exp(exponent))

        if self.formulation == "hill_2p":
            bfr0 = draws["bfr0"].reshape(-1, 1)
            eta_km = (
                draws["eta_km"].reshape(-1, 1)
                if "eta_km" in draws
                else np.log(draws["km"].reshape(-1, 1))
            )
            n_hill = draws["n_hill"].reshape(-1, 1)
            log_c = np.log(c_arr)
            exponent = np.clip(n_hill * (log_c - eta_km), -50.0, 50.0)
            return bfr0 / (1.0 + np.exp(exponent))

        if self.formulation == "log_linear":
            a = draws["a"].reshape(-1, 1)
            b = draws["b"].reshape(-1, 1)
            log_c = np.log(c_arr)
            return np.maximum(1e-5, a - b * log_c)

        if self.formulation == "hormesis":
            bfr0 = draws["bfr0"].reshape(-1, 1)
            bfr_floor = (
                draws["bfr_floor"].reshape(-1, 1)
                if "bfr_floor" in draws
                else bfr0 * draws["r_floor"].reshape(-1, 1)
            )
            eta_km = (
                draws["eta_km"].reshape(-1, 1)
                if "eta_km" in draws
                else np.log(draws["km"].reshape(-1, 1))
            )
            n_hill = draws["n_hill"].reshape(-1, 1)
            f = draws["f"].reshape(-1, 1)
            log_c = np.log(c_arr)
            exponent = np.clip(n_hill * (log_c - eta_km), -50.0, 50.0)
            return bfr_floor + (bfr0 - bfr_floor + f * c_arr) / (1.0 + np.exp(exponent))

        raise RuntimeError(
            f"Posterior does not contain expected parameters for formulation '{self.formulation}'. "
            f"Available posterior variables: {list(draws.keys())}"
        )

    def r_to(
        self,
        c_bone: float | Sequence[float] | np.ndarray,
        as_percentage: bool = True,
    ) -> float | np.ndarray:
        """Calculate the control-normalized bone turnover modifier r_TO(C).

        Evaluates r_TO(C) = [mu(C) / mu(C_control)] * 100% across posterior draws.
        Supports duck-typing across scalar floats, lists, and NumPy arrays.
        At C = C_control (282.0 mg/kg), r_TO is identically 100.0% (or 1.0).

        Parameters
        ----------
        c_bone : float, Sequence[float], or np.ndarray
            Trabecular bone fluoride concentration [mg/kg ash].
        as_percentage : bool, default=True
            If True, returns value as percentage (e.g. 100.0 at baseline).
            If False, returns fractional value (e.g. 1.0 at baseline).

        Returns
        -------
        float or np.ndarray
            Turnover modifier. Returns a scalar float for scalar input,
            or an np.ndarray with matching shape for array/list inputs.

        Raises
        ------
        RuntimeError
            If model is unfitted.
        ValueError
            If any concentration is non-positive or non-finite.
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        is_scalar = isinstance(c_bone, (int, float, np.floating, np.integer))

        if is_scalar:
            val = float(c_bone)
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"Bone concentration must be positive and finite, got {val}")
            c_eval = np.array([[val]])
        else:
            arr = np.asarray(c_bone, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ValueError("Concentration inputs must contain only finite values")
            if np.any(arr <= 0):
                raise ValueError("All concentration values must be strictly positive")
            c_eval = arr.reshape(1, -1)

        # Evaluate mu at input concentrations and at baseline control
        mu_eval = self._predict_mu_samples(c_eval)  # shape (n_samples, K)
        c_ctrl_arr = np.array([[self.c_control]])
        mu_ctrl = self._predict_mu_samples(c_ctrl_arr)  # shape (n_samples, 1)

        # Draw-by-draw normalized ratio
        ratio_samples = mu_eval / mu_ctrl
        if as_percentage:
            ratio_samples = ratio_samples * 100.0

        # Mean across posterior draws
        r_to_mean = np.mean(ratio_samples, axis=0)

        if is_scalar:
            return float(r_to_mean[0])

        arr_input = np.asarray(c_bone)
        return r_to_mean.reshape(arr_input.shape)

    def clearance_grid(
        self,
        c_min: float = 100.0,
        c_max: float = 15000.0,
        n_points: int = 100,
        hdi_prob: float = DEFAULT_HDI_PROB,
    ) -> pd.DataFrame:
        """Generate a dense lookup grid of r_TO with posterior uncertainty statistics.

        Parameters
        ----------
        c_min : float, default=100.0
            Minimum concentration bound [mg/kg ash].
        c_max : float, default=15000.0
            Maximum concentration bound [mg/kg ash].
        n_points : int, default=100
            Number of grid points.
        hdi_prob : float, default=0.94
            Highest density interval probability mass (94% HDI).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ['c_bone', 'mean', 'median', 'std', 'hdi_lower', 'hdi_upper'].
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")
        if c_min <= 0:
            raise ValueError(f"c_min must be strictly positive, got {c_min}")
        if c_max <= c_min:
            raise ValueError(f"c_max ({c_max}) must be strictly greater than c_min ({c_min})")
        if n_points < 2:
            raise ValueError(f"n_points must be at least 2, got {n_points}")

        c_grid = np.linspace(c_min, c_max, n_points)
        c_eval = c_grid.reshape(1, -1)

        mu_grid = self._predict_mu_samples(c_eval)  # shape (n_samples, n_points)
        c_ctrl_arr = np.array([[self.c_control]])
        mu_ctrl = self._predict_mu_samples(c_ctrl_arr)  # shape (n_samples, 1)

        pct_samples = (mu_grid / mu_ctrl) * 100.0  # shape (n_samples, n_points)

        mean_vals = np.mean(pct_samples, axis=0)
        median_vals = np.median(pct_samples, axis=0)
        std_vals = np.std(pct_samples, axis=0)
        hdi_lower, hdi_upper = _compute_hdi_bounds(pct_samples, prob=hdi_prob)

        return pd.DataFrame(
            {
                "c_bone": c_grid,
                "mean": mean_vals,
                "median": median_vals,
                "std": std_vals,
                "hdi_lower": hdi_lower,
                "hdi_upper": hdi_upper,
            }
        )

    def predict(
        self,
        c_bone: float | Sequence[float] | np.ndarray,
        random_seed: int = RANDOM_SEED,
    ) -> dict[str, np.ndarray]:
        """Generate posterior trajectory draws and simulated observations.

        Parameters
        ----------
        c_bone : float, Sequence[float], or np.ndarray
            Concentration grid points (mg/kg ash).
        random_seed : int, default=42
            Seed for residual sampling.

        Returns
        -------
        dict[str, np.ndarray]
            - 'c_bone': 1D array of evaluation concentrations.
            - 'mu': Trajectory draws of shape (n_samples, n_points).
            - 'y_obs': Simulated observations of shape (n_samples, n_points).
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        c_arr = np.atleast_1d(np.asarray(c_bone, dtype=float)).ravel()
        if not np.all(np.isfinite(c_arr)) or np.any(c_arr <= 0):
            raise ValueError("All concentration values must be strictly positive and finite")

        mu_samples = self._predict_mu_samples(c_arr.reshape(1, -1))
        draws = self._extract_posterior_draws()
        sigma_log = (
            draws["sigma_log"].reshape(-1, 1)
            if "sigma_log" in draws
            else np.full((mu_samples.shape[0], 1), 0.3)
        )

        rng = np.random.default_rng(random_seed)
        t_residuals = rng.standard_t(df=4.0, size=mu_samples.shape)
        y_obs_samples = np.exp(np.log(mu_samples) + sigma_log * t_residuals)

        return {
            "c_bone": c_arr,
            "mu": mu_samples,
            "y_obs": y_obs_samples,
        }

    def summary(self, ci_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
        """Return posterior parameter summary table using arviz_stats.

        Parameters
        ----------
        ci_prob : float, default=0.94
            Credible interval mass (94% HDI).

        Returns
        -------
        pd.DataFrame
            Summary dataframe with mean, sd, HDI bounds, r_hat, and ESS.
        """
        if self.idata is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        var_candidates = [
            "bfr0",
            "r_floor",
            "km",
            "bfr_floor",
            "n_hill",
            "sigma_log",
            "a",
            "b",
            "f",
        ]
        post = getattr(self.idata, "posterior", None)
        if post is not None:
            var_names = [v for v in var_candidates if v in post.data_vars]
        else:
            var_names = None

        return azs.summary(self.idata, var_names=var_names, ci_prob=ci_prob, ci_kind="hdi")

    def diagnostics(self) -> dict[str, Any]:
        """Run MCMC convergence diagnostics (R-hat, ESS, divergences).

        Returns
        -------
        dict[str, Any]
            Dictionary of convergence metrics and pass/fail booleans.
        """
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

    def compare_formulations(
        self,
        formulations: Sequence[str] = ("4p_hill", "hill_2p", "log_linear", "hormesis"),
        data: pd.DataFrame | None = None,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        target_accept: float = 0.95,
        random_seed: int = RANDOM_SEED,
        models: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compare turnover model formulations using PSIS-LOO cross-validation.

        Parameters
        ----------
        formulations : Sequence[str], default=('4p_hill', 'hill_2p', 'log_linear', 'hormesis')
            Formulation names to compare if models dict is not provided.
        data : pd.DataFrame | None, optional
            Dataset to fit models on. Defaults to cached or bundled Boivin data.
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
            Dictionary of pre-fitted TurnoverModel instances or InferenceData objects.
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
            for form in formulations:
                clean_form = "hill_2p" if form == "2p_hill" else form
                if clean_form == self.formulation and self.idata is not None:
                    comp_dict[form] = self.idata
                else:
                    logger.info("Fitting formulation '%s' for model comparison...", form)
                    m = TurnoverModel(
                        formulation=clean_form,  # type: ignore[arg-type]
                        data=data,
                        c_control=self.c_control,
                    )
                    m.fit(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        target_accept=target_accept,
                        random_seed=random_seed,
                    )
                    comp_dict[form] = m.idata

        res = azs.compare(comp_dict)
        if "elpd" in res.columns and "elpd_loo" not in res.columns:
            res["elpd_loo"] = res["elpd"]
        if "p" in res.columns and "p_loo" not in res.columns:
            res["p_loo"] = res["p"]
        if "elpd_diff" in res.columns and "d_elpd" not in res.columns:
            res["d_elpd"] = res["elpd_diff"]

        return res

    def save(self, filepath: str | Path) -> None:
        """Save fitted InferenceData to a NetCDF file via h5netcdf.

        Parameters
        ----------
        filepath : str or Path
            Destination .nc file path.
        """
        if self.idata is None:
            raise RuntimeError("No fitted InferenceData to save. Call fit() first.")
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.idata, "to_netcdf"):
            self.idata.to_netcdf(str(target), engine="h5netcdf")
        else:
            import arviz as az

            az.to_netcdf(self.idata, str(target), engine="h5netcdf")
        logger.info("Saved InferenceData to %s", target)

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        formulation: TurnoverFormulation = "4p_hill",
        data: pd.DataFrame | None = None,
    ) -> TurnoverModel:
        """Load InferenceData from a NetCDF file via h5netcdf and return a TurnoverModel.

        Parameters
        ----------
        filepath : str or Path
            Path to existing NetCDF file.
        formulation : str, default='4p_hill'
            Model formulation corresponding to the saved samples.
        data : pd.DataFrame | None, optional
            Optional dataset for model context.

        Returns
        -------
        TurnoverModel
            TurnoverModel instance with loaded posterior.
        """
        target = Path(filepath)
        if not target.is_file():
            raise FileNotFoundError(f"NetCDF file not found: {target}")

        try:
            loaded = xr.load_datatree(str(target), engine="h5netcdf")
        except Exception as exc:
            logger.debug("xr.load_datatree failed (%s); using arviz.from_netcdf", exc)
            import arviz as az

            loaded = az.from_netcdf(str(target), engine="h5netcdf")

        instance = cls(formulation=formulation, data=data)
        instance.idata = loaded
        logger.info("Loaded InferenceData from %s", target)
        return instance
