"""Hawkins [18F]Fluoride PET kinetics and physiological bone growth bridge.

Integrates biomarker relative activity indices (RAI_ALP -> r_form, RAI_HP -> r_resorp)
and bone turnover suppression modifiers (r_TO) with Hawkins et al. (1992) 2-tissue
compartmental PET kinetics and PBKM Manual (O'Flaherty & Reponen 1997) skeletal
growth and calcium accretion balance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from bone_predictor.constants import (
    ADULT_TURNOVER_BASELINE,
    C_ADULT_BY_SEX,
    C_CONTROL,
    CORTICAL_TURNOVER_BASELINE,
    HAWKINS_K1,
    TRABECULAR_TURNOVER_BASELINE,
    HAWKINS_k2,
    HAWKINS_k3_ADULT,
    HAWKINS_k4_ADULT,
)
from bone_predictor.physiology import body_weight, normalize_sex

if TYPE_CHECKING:
    from bone_predictor.biomarkers import BiomarkerModel
    from bone_predictor.turnover import TurnoverModel

logger = logging.getLogger(__name__)


def _analytical_relative_activity(
    age: float | np.ndarray,
    sex: str,
    marker: str,
) -> float | np.ndarray:
    """Analytical baseline relative activity index (RAI) without MCMC sampling.

    Reconstructs the K=3 component deconvolution model at default parameters:
    mu(t, s) = c(s) + K_inf(t) + K_pub(t, s) + [I_(s=Female) * K_meno(t)]
    RAI(t, s) = mu(t, s) / c(s)

    Parameters
    ----------
    age : float or np.ndarray
        Chronological age in years (> 0).
    sex : str
        Biological sex indicator ('M' or 'F').
    marker : str
        Biomarker indicator ('alp' for formation, 'hp' for resorption).

    Returns
    -------
    float or np.ndarray
        Analytical Relative Activity Index (r_form or r_resorp).
    """
    s_norm = normalize_sex(sex)
    m_norm = marker.strip().lower()
    if m_norm not in ("alp", "hp"):
        raise ValueError(f"Invalid marker='{marker}'. Expected 'alp' or 'hp'.")

    arr = np.asarray(age, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Age values must be finite")
    if np.any(arr <= 0):
        raise ValueError("Age values must be strictly positive (> 0 yr)")

    is_female = s_norm == "F"

    # Baseline adult plateau c(s)
    if m_norm == "alp":
        c_val = 10.0 if is_female else 10.5
        a_inf = 110.0
        a_pub = 85.0
        a_meno = 2.0
    else:
        c_val = 16.5 if is_female else 16.0
        a_inf = 160.0
        a_pub = 85.0
        a_meno = 2.5

    # Infancy Weibull decay kernel
    lambda_inf = 1.33
    gamma_inf = 1.0
    inf_exponent = np.clip((arr / lambda_inf) ** gamma_inf, 0.0, 50.0)
    k_inf = a_inf * np.exp(-inf_exponent)

    # Puberty Gaussian pulse kernel
    mu_pub = 11.5 if is_female else 13.8
    sigma_pub = 1.5 if is_female else 1.6
    pub_exponent = np.clip(((arr - mu_pub) ** 2) / (2.0 * sigma_pub**2), 0.0, 50.0)
    k_pub = a_pub * np.exp(-pub_exponent)

    # Postmenopause Gaussian pulse kernel (female only)
    if is_female:
        mu_meno = 54.0
        sigma_meno = 4.5
        meno_exponent = np.clip(((arr - mu_meno) ** 2) / (2.0 * sigma_meno**2), 0.0, 50.0)
        k_meno = a_meno * np.exp(-meno_exponent)
    else:
        k_meno = np.zeros_like(arr)

    mu_total = c_val + k_inf + k_pub + k_meno
    rai = mu_total / c_val
    return float(rai) if np.ndim(age) == 0 else rai


class HawkinsCalibrator:
    """Hawkins PET kinetics and physiological bone growth bridge calibrator.

    Bridges empirical biomarker deconvolution (b-ALP and u-HP) to PBKM human skeletal
    growth and turnover rates, independent of fluoride exposure, and incorporates
    concentration-dependent fluoride suppression modifiers onto Hawkins PET microparameters.

    Parameters
    ----------
    biomarker_model : BiomarkerModel | None, optional
        Fitted BiomarkerModel instance. If None or unfitted, uses analytical
        analytical baseline functions.
    turnover_model : TurnoverModel | None, optional
        Fitted TurnoverModel instance. If None, defaults to r_to = 1.0 (no suppression).
    constants_override : dict[str, float] | None, optional
        Optional dictionary of constant overrides for kinetics or physiology.
    """

    def __init__(
        self,
        biomarker_model: BiomarkerModel | None = None,
        turnover_model: TurnoverModel | None = None,
        constants_override: dict[str, float] | None = None,
    ) -> None:
        self.biomarker_model: BiomarkerModel | None = biomarker_model
        self.turnover_model: TurnoverModel | None = turnover_model

        # Default physical & physiological constants
        self.k1: float = HAWKINS_K1
        self.k2: float = HAWKINS_k2
        self.k3_adult: float = HAWKINS_k3_ADULT
        self.k4_adult: float = HAWKINS_k4_ADULT
        self.adult_turnover_baseline: float = ADULT_TURNOVER_BASELINE
        self.trabecular_turnover_baseline: float = TRABECULAR_TURNOVER_BASELINE
        self.cortical_turnover_baseline: float = CORTICAL_TURNOVER_BASELINE
        self.c_adult_f: float = C_ADULT_BY_SEX["F"]
        self.c_adult_m: float = C_ADULT_BY_SEX["M"]

        if constants_override:
            for k, v in constants_override.items():
                k_clean = k.strip().upper()
                if k_clean in ("K1", "HAWKINS_K1"):
                    self.k1 = float(v)
                elif k_clean in ("K2", "HAWKINS_K2"):
                    self.k2 = float(v)
                elif k_clean in ("K3_ADULT", "HAWKINS_K3_ADULT"):
                    self.k3_adult = float(v)
                elif k_clean in ("K4_ADULT", "HAWKINS_K4_ADULT"):
                    self.k4_adult = float(v)
                elif k_clean in ("ADULT_TURNOVER_BASELINE", "FBFR_BASELINE"):
                    self.adult_turnover_baseline = float(v)
                elif k_clean in ("TRABECULAR_TURNOVER_BASELINE", "TFBFR_BASELINE"):
                    self.trabecular_turnover_baseline = float(v)
                elif k_clean in ("CORTICAL_TURNOVER_BASELINE", "CFBFR_BASELINE"):
                    self.cortical_turnover_baseline = float(v)
                elif k_clean in ("C_ADULT_F", "C_ADULT_FEMALE"):
                    self.c_adult_f = float(v)
                elif k_clean in ("C_ADULT_M", "C_ADULT_MALE"):
                    self.c_adult_m = float(v)

    def predict_relative_activity(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: Any,
    ) -> dict[str, float | np.ndarray]:
        """Predict formation (r_form) and resorption (r_resorp) relative activities.

        Returns RAI_ALP (formation rate proxy) and RAI_HP (resorption rate proxy),
        strictly independent of fluoride concentration.

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Chronological age in years (> 0).
        sex : str or int
            Biological sex indicator ('M' or 'F').

        Returns
        -------
        dict[str, float | np.ndarray]
            Dictionary containing:
            - 'r_form': Relative formation activity index (RAI_ALP).
            - 'r_resorp': Relative resorption activity index (RAI_HP).
        """
        s_norm = normalize_sex(sex)
        is_scalar = isinstance(age, (int, float, np.floating, np.integer))

        if (
            self.biomarker_model is not None
            and getattr(self.biomarker_model, "idata", None) is not None
        ):
            try:
                r_form = self.biomarker_model.relative_activity_index(age, sex=s_norm, marker="alp")
                r_resorp = self.biomarker_model.relative_activity_index(
                    age, sex=s_norm, marker="hp"
                )
                if is_scalar:
                    return {"r_form": float(r_form), "r_resorp": float(r_resorp)}
                return {
                    "r_form": np.asarray(r_form, dtype=float),
                    "r_resorp": np.asarray(r_resorp, dtype=float),
                }
            except Exception as exc:
                logger.debug(
                    "BiomarkerModel evaluation failed (%s); using analytical fallback", exc
                )

        r_form_calc = _analytical_relative_activity(age, sex=s_norm, marker="alp")
        r_resorp_calc = _analytical_relative_activity(age, sex=s_norm, marker="hp")

        if is_scalar:
            return {"r_form": float(r_form_calc), "r_resorp": float(r_resorp_calc)}
        return {
            "r_form": np.asarray(r_form_calc, dtype=float),
            "r_resorp": np.asarray(r_resorp_calc, dtype=float),
        }

    def fractional_rates(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: Any,
    ) -> dict[str, float | np.ndarray]:
        """Calculate fractional bone formation rates matching PBKM Manual proportions.

        Equations:
        - Whole-skeleton FBFR(age, sex) = 0.10 * r_form [yr^-1] (10%/yr adult baseline)
        - Trabecular TFBFR(age, sex) = (0.65 / 0.20) * FBFR = 0.325 * r_form [yr^-1]
        - Cortical CFBFR(age, sex) = (0.35 / 0.80) * FBFR = 0.04375 * r_form [yr^-1]

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Chronological age in years (> 0).
        sex : str or int
            Biological sex ('M' or 'F').

        Returns
        -------
        dict[str, float | np.ndarray]
            Dictionary containing FBFR, TFBFR, and CFBFR fractional rates.
        """
        act = self.predict_relative_activity(age, sex=sex)
        r_form = act["r_form"]

        fbfr = self.adult_turnover_baseline * r_form
        tfbfr = self.trabecular_turnover_baseline * r_form
        cfbfr = self.cortical_turnover_baseline * r_form

        is_scalar = isinstance(age, (int, float, np.floating, np.integer))
        if is_scalar:
            return {
                "FBFR": float(fbfr),
                "TFBFR": float(tfbfr),
                "CFBFR": float(cfbfr),
                "fbfr": float(fbfr),
                "tfbfr": float(tfbfr),
                "cfbfr": float(cfbfr),
            }

        return {
            "FBFR": np.asarray(fbfr, dtype=float),
            "TFBFR": np.asarray(tfbfr, dtype=float),
            "CFBFR": np.asarray(cfbfr, dtype=float),
            "fbfr": np.asarray(fbfr, dtype=float),
            "tfbfr": np.asarray(tfbfr, dtype=float),
            "cfbfr": np.asarray(cfbfr, dtype=float),
        }

    def calcium_flux(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: Any,
    ) -> dict[str, float | np.ndarray]:
        """Calculate whole-body calcium accretion flux calibrated against tracer data.

        Equations:
        - f_form = c_adult(sex) * r_form * body_weight(age, sex) [kg Ca/year]
        - f_resorp = c_adult(sex) * r_resorp * body_weight(age, sex) [kg Ca/year]
        - net_accretion = f_form - f_resorp [kg Ca/year]

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Chronological age in years (> 0).
        sex : str or int
            Biological sex ('M' or 'F').

        Returns
        -------
        dict[str, float | np.ndarray]
            Dictionary containing f_form, f_resorp, net_accretion, and body_weight.
        """
        s_norm = normalize_sex(sex)
        act = self.predict_relative_activity(age, sex=s_norm)
        r_form = act["r_form"]
        r_resorp = act["r_resorp"]

        c_ad = self.c_adult_f if s_norm == "F" else self.c_adult_m
        bw = body_weight(age, s_norm)

        f_form = c_ad * r_form * bw
        f_resorp = c_ad * r_resorp * bw
        net_accretion = f_form - f_resorp

        is_scalar = isinstance(age, (int, float, np.floating, np.integer))
        if is_scalar:
            return {
                "f_form": float(f_form),
                "f_resorp": float(f_resorp),
                "net_accretion": float(net_accretion),
                "body_weight": float(bw),
            }

        return {
            "f_form": np.asarray(f_form, dtype=float),
            "f_resorp": np.asarray(f_resorp, dtype=float),
            "net_accretion": np.asarray(net_accretion, dtype=float),
            "body_weight": np.asarray(bw, dtype=float),
        }

    def _resolve_r_to(
        self,
        c_bone: float | Sequence[float] | np.ndarray | None,
        shape_reference: np.ndarray | None = None,
        is_scalar_age: bool = True,
    ) -> float | np.ndarray:
        """Resolve turnover suppression modifier r_TO(C_bone)."""
        if c_bone is None:
            if is_scalar_age or shape_reference is None:
                return 1.0
            return np.ones_like(shape_reference, dtype=float)

        if (
            self.turnover_model is not None
            and getattr(self.turnover_model, "idata", None) is not None
        ):
            try:
                r_to_res = self.turnover_model.r_to(c_bone, as_percentage=False)
                if isinstance(c_bone, (int, float, np.floating, np.integer)):
                    val = float(r_to_res)
                    if not is_scalar_age and shape_reference is not None:
                        return np.full_like(shape_reference, val, dtype=float)
                    return val
                return np.asarray(r_to_res, dtype=float)
            except Exception as exc:
                logger.debug("TurnoverModel evaluation failed (%s); using analytical 4P Hill", exc)

        # Analytical 4-parameter Hill fallback for r_TO(C)
        c_arr = np.asarray(c_bone, dtype=float)
        if not np.all(np.isfinite(c_arr)):
            raise ValueError("Concentration inputs must contain only finite values")
        if np.any(c_arr <= 0):
            raise ValueError("Concentration inputs must be strictly positive")

        bfr0 = 0.076
        r_floor = 0.333333
        bfr_floor = bfr0 * r_floor
        eta_km = np.log(3500.0)
        n_hill = 2.0

        def _hill_mu(conc: np.ndarray) -> np.ndarray:
            exp_term = np.clip(n_hill * (np.log(conc) - eta_km), -50.0, 50.0)
            return bfr_floor + (bfr0 - bfr_floor) / (1.0 + np.exp(exp_term))

        mu_c = _hill_mu(c_arr)
        mu_ctrl = _hill_mu(np.array([C_CONTROL]))
        r_to_val = mu_c / mu_ctrl

        if isinstance(c_bone, (int, float, np.floating, np.integer)):
            val = float(r_to_val[0]) if np.ndim(r_to_val) > 0 else float(r_to_val)
            if not is_scalar_age and shape_reference is not None:
                return np.full_like(shape_reference, val, dtype=float)
            return val
        return np.asarray(r_to_val, dtype=float)

    def predict_pet_kinetics(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: Any,
        c_bone: float | Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, float | np.ndarray]:
        """Predict Hawkins [18F]Fluoride PET microparameters k3, k4, and Ki.

        Equations:
        - k3 = r_form * r_to * HAWKINS_k3_ADULT (default: 0.132 min^-1)
        - k4 = r_resorp * HAWKINS_k4_ADULT (default: 0.002 min^-1)
        - Ki = (HAWKINS_K1 * k3) / (HAWKINS_k2 + k3)

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Chronological age in years (> 0).
        sex : str or int
            Biological sex ('M' or 'F').
        c_bone : float, Sequence[float], or np.ndarray | None, optional
            Trabecular bone fluoride concentration [mg/kg ash]. If None, r_to = 1.0.

        Returns
        -------
        dict[str, float | np.ndarray]
            Dictionary containing k3, k4, Ki, r_to, r_form, and r_resorp.
        """
        s_norm = normalize_sex(sex)
        act = self.predict_relative_activity(age, sex=s_norm)
        r_form = act["r_form"]
        r_resorp = act["r_resorp"]

        is_scalar = isinstance(age, (int, float, np.floating, np.integer))
        shape_ref = None if is_scalar else np.asarray(r_form)

        r_to = self._resolve_r_to(c_bone, shape_reference=shape_ref, is_scalar_age=is_scalar)

        k3 = r_form * r_to * self.k3_adult
        k4 = r_resorp * self.k4_adult
        ki = (self.k1 * k3) / (self.k2 + k3)

        if is_scalar:
            return {
                "k3": float(k3),
                "k4": float(k4),
                "Ki": float(ki),
                "ki": float(ki),
                "r_to": float(r_to),
                "r_form": float(r_form),
                "r_resorp": float(r_resorp),
            }

        return {
            "k3": np.asarray(k3, dtype=float),
            "k4": np.asarray(k4, dtype=float),
            "Ki": np.asarray(ki, dtype=float),
            "ki": np.asarray(ki, dtype=float),
            "r_to": np.asarray(r_to, dtype=float),
            "r_form": np.asarray(r_form, dtype=float),
            "r_resorp": np.asarray(r_resorp, dtype=float),
        }

    def predict_modifiers(
        self,
        age: float | Sequence[float] | np.ndarray,
        sex: Any,
        c_bone: float | Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Predict unified skeletal modifiers across remodeling, accretion, and PET kinetics.

        Parameters
        ----------
        age : float, Sequence[float], or np.ndarray
            Chronological age in years (> 0).
        sex : str or int
            Biological sex ('M' or 'F').
        c_bone : float, Sequence[float], or np.ndarray | None, optional
            Trabecular bone fluoride concentration [mg/kg ash].

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - age, sex
            - r_form, r_resorp, r_to
            - FBFR, TFBFR, CFBFR (and lowercase aliases)
            - f_form, f_resorp, net_accretion
            - k3, k4, Ki (and ki)
        """
        s_norm = normalize_sex(sex)
        is_scalar = isinstance(age, (int, float, np.floating, np.integer))

        rates = self.fractional_rates(age, sex=s_norm)
        fluxes = self.calcium_flux(age, sex=s_norm)
        pet = self.predict_pet_kinetics(age, sex=s_norm, c_bone=c_bone)

        if is_scalar:
            return {
                "age": float(age),
                "sex": s_norm,
                "r_form": float(pet["r_form"]),
                "r_resorp": float(pet["r_resorp"]),
                "r_to": float(pet["r_to"]),
                "FBFR": float(rates["FBFR"]),
                "TFBFR": float(rates["TFBFR"]),
                "CFBFR": float(rates["CFBFR"]),
                "fbfr": float(rates["FBFR"]),
                "tfbfr": float(rates["TFBFR"]),
                "cfbfr": float(rates["CFBFR"]),
                "f_form": float(fluxes["f_form"]),
                "f_resorp": float(fluxes["f_resorp"]),
                "net_accretion": float(fluxes["net_accretion"]),
                "k3": float(pet["k3"]),
                "k4": float(pet["k4"]),
                "Ki": float(pet["Ki"]),
                "ki": float(pet["Ki"]),
            }

        arr_age = np.asarray(age, dtype=float)
        return {
            "age": arr_age,
            "sex": s_norm,
            "r_form": np.asarray(pet["r_form"], dtype=float),
            "r_resorp": np.asarray(pet["r_resorp"], dtype=float),
            "r_to": np.asarray(pet["r_to"], dtype=float),
            "FBFR": np.asarray(rates["FBFR"], dtype=float),
            "TFBFR": np.asarray(rates["TFBFR"], dtype=float),
            "CFBFR": np.asarray(rates["CFBFR"], dtype=float),
            "fbfr": np.asarray(rates["FBFR"], dtype=float),
            "tfbfr": np.asarray(rates["TFBFR"], dtype=float),
            "cfbfr": np.asarray(rates["CFBFR"], dtype=float),
            "f_form": np.asarray(fluxes["f_form"], dtype=float),
            "f_resorp": np.asarray(fluxes["f_resorp"], dtype=float),
            "net_accretion": np.asarray(fluxes["net_accretion"], dtype=float),
            "k3": np.asarray(pet["k3"], dtype=float),
            "k4": np.asarray(pet["k4"], dtype=float),
            "Ki": np.asarray(pet["Ki"], dtype=float),
            "ki": np.asarray(pet["Ki"], dtype=float),
        }

    def modifier_grid(
        self,
        age_min: float = 0.0,
        age_max: float = 85.0,
        step: float = 0.5,
        sex: str = "both",
        c_bone: float | None = None,
    ) -> pd.DataFrame:
        """Generate a dense lookup grid of bone turnover modifiers across lifespan.

        Parameters
        ----------
        age_min : float, default=0.0
            Minimum age in years. Clamped to >= 0.01 yr.
        age_max : float, default=85.0
            Maximum age in years.
        step : float, default=0.5
            Grid age step size in years.
        sex : str, default='both'
            Biological sex ('F', 'M', or 'both').
        c_bone : float | None, optional
            Trabecular bone fluoride concentration [mg/kg ash].

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: age, sex, body_weight, r_form, r_resorp,
            r_to, FBFR, TFBFR, CFBFR, f_form, f_resorp, net_accretion, k3, k4, Ki.
        """
        if step <= 0:
            raise ValueError("step must be strictly positive")
        if age_max <= age_min:
            raise ValueError("age_max must be strictly greater than age_min")

        sex_clean = sex.strip().lower()
        if sex_clean in ("both", "all"):
            df_f = self.modifier_grid(
                age_min=age_min, age_max=age_max, step=step, sex="F", c_bone=c_bone
            )
            df_m = self.modifier_grid(
                age_min=age_min, age_max=age_max, step=step, sex="M", c_bone=c_bone
            )
            return pd.concat([df_f, df_m], ignore_index=True)

        s_norm = normalize_sex(sex)
        ages = np.arange(age_min, age_max + 0.5 * step, step)
        ages[0] = max(ages[0], 0.01)

        mods = self.predict_modifiers(ages, sex=s_norm, c_bone=c_bone)
        bw = body_weight(ages, s_norm)

        df = pd.DataFrame(
            {
                "age": ages,
                "sex": s_norm,
                "body_weight": bw,
                "r_form": mods["r_form"],
                "r_resorp": mods["r_resorp"],
                "r_to": mods["r_to"],
                "FBFR": mods["FBFR"],
                "TFBFR": mods["TFBFR"],
                "CFBFR": mods["CFBFR"],
                "f_form": mods["f_form"],
                "f_resorp": mods["f_resorp"],
                "net_accretion": mods["net_accretion"],
                "k3": mods["k3"],
                "k4": mods["k4"],
                "Ki": mods["Ki"],
            }
        )
        return df
