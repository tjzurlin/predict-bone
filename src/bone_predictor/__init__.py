from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bone_predictor.biomarkers import BiomarkerModel
    from bone_predictor.turnover import TurnoverModel

from bone_predictor.constants import (
    ADULT_TURNOVER_BASELINE,
    ASH_FRACTION,
    C_ADULT,
    C_CONTROL,
    CA_BIRTH,
    CORTICAL_TURNOVER_BASELINE,
    CORTICAL_VOLUME_FRACTION,
    HAWKINS_K1,
    TRABECULAR_TURNOVER_BASELINE,
    TRABECULAR_VOLUME_FRACTION,
    HAWKINS_k2,
    HAWKINS_k3_ADULT,
    HAWKINS_k4_ADULT,
    HAWKINS_Ki_ADULT,
)
from bone_predictor.data import (
    PreparedData,
    load_boivin_data,
    load_ca_accretion_data,
    load_data,
    load_stepan_markers,
)
from bone_predictor.hawkins import HawkinsCalibrator
from bone_predictor.physiology import (
    body_weight,
    body_weight_derivative,
    bone_density,
    bone_mass,
    bone_volume,
    compartmental_bone_volumes,
    normalize_sex,
    skeletal_remodeling_balance,
)
from bone_predictor.units import (
    ash_to_molar,
    ash_to_wet,
    convert_concentration,
    molar_to_ash,
    molar_to_wet,
    wet_to_ash,
    wet_to_molar,
)

__version__ = "0.1.0"

__all__ = [
    "ADULT_TURNOVER_BASELINE",
    "ASH_FRACTION",
    "BiomarkerModel",
    "C_ADULT",
    "C_CONTROL",
    "CA_BIRTH",
    "CORTICAL_TURNOVER_BASELINE",
    "CORTICAL_VOLUME_FRACTION",
    "HAWKINS_K1",
    "HAWKINS_Ki_ADULT",
    "HAWKINS_k2",
    "HAWKINS_k3_ADULT",
    "HAWKINS_k4_ADULT",
    "HawkinsCalibrator",
    "PreparedData",
    "TRABECULAR_TURNOVER_BASELINE",
    "TRABECULAR_VOLUME_FRACTION",
    "TurnoverModel",
    "__version__",
    "ash_to_molar",
    "ash_to_wet",
    "body_weight",
    "body_weight_derivative",
    "bone_density",
    "bone_mass",
    "bone_volume",
    "compartmental_bone_volumes",
    "convert_concentration",
    "load_boivin_data",
    "load_ca_accretion_data",
    "load_data",
    "load_stepan_markers",
    "molar_to_ash",
    "molar_to_wet",
    "normalize_sex",
    "skeletal_remodeling_balance",
    "wet_to_ash",
    "wet_to_molar",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BiomarkerModel": ("bone_predictor.biomarkers", "BiomarkerModel"),
    "TurnoverModel": ("bone_predictor.turnover", "TurnoverModel"),
}


def __getattr__(name: str) -> Any:
    """Lazy import handler (PEP 562) to avoid eager PyMC compilation overhead."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

