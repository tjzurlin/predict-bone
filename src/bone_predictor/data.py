"""Data loading and schema validation infrastructure for bone-predictor.

Manages loading, invariant checking, and schema normalization for:
1. Boivin et al. histomorphometric bone turnover data (bone_TO.csv, N=15)
2. Historical calcium accretion tracer data (ca_accretion.csv, N=64)
3. Stepan et al. remodeling biomarker data (stepan_bone_markers.xlsx, N=2,944)
"""

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from bone_predictor.physiology import normalize_sex


def get_data_dir() -> Path:
    """Return the filesystem Path to the bundled data package directory."""
    pkg_data = Path(__file__).parent / "data"
    if pkg_data.is_dir():
        return pkg_data
    try:
        return Path(str(resources.files("bone_predictor.data")))
    except (TypeError, AttributeError):
        return pkg_data


def get_bundled_data_path(filename: str) -> Path:
    """Resolve the absolute path to a bundled dataset file.

    Parameters
    ----------
    filename : str
        Name of the file (e.g. 'bone_TO.csv').

    Returns
    -------
    Path
        Absolute path to the existing data file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist in the bundled data directory.
    """
    path = get_data_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Bundled dataset '{filename}' not found at {path}")
    return path


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute deterministic SHA-256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_boivin_data(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load and validate Boivin et al. histomorphometric bone turnover data.

    Parameters
    ----------
    filepath : Path or str, optional
        Custom path to bone_TO.csv. If None, loads the bundled dataset.

    Returns
    -------
    pd.DataFrame
        Validated dataframe with columns:
        - 'c_bone': Trabecular bone fluoride concentration [mg/kg ash]
        - 'cn_bfr': Cancellous bone formation rate [yr^-1]
        - 'cn_bfr_control': Formation rate as percentage of baseline control [%]
    """
    path = Path(filepath) if filepath is not None else get_bundled_data_path("bone_TO.csv")
    df = pd.read_csv(path)

    # Normalize column names to canonical snake_case
    col_map = {
        "C_bone (mg/kg)": "c_bone",
        "Cn-BFR": "cn_bfr",
        "Cn-BFR/control": "cn_bfr_control",
    }
    df = df.rename(columns=col_map)

    required_cols = {"c_bone", "cn_bfr", "cn_bfr_control"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in Boivin data: {missing}")

    # Schema & invariant validation
    if len(df) != 15:
        raise ValueError(f"Expected exactly 15 observations in Boivin data, found {len(df)}")
    if df.isna().any().any():
        raise ValueError("Boivin dataset contains null / NaN values")
    if (df["c_bone"] <= 0).any():
        raise ValueError("Bone fluoride concentration must be strictly positive")
    if (df["cn_bfr"] <= 0).any():
        raise ValueError("Cancellous bone formation rate must be strictly positive")
    if (df["cn_bfr_control"] <= 0).any():
        raise ValueError("Normalized formation rate must be strictly positive")

    # Verify baseline control observation exists
    baseline_match = np.isclose(df["c_bone"], 282.0, atol=1e-2)
    if not np.any(baseline_match):
        raise ValueError("Boivin dataset must contain baseline control concentration (282 mg/kg)")

    return df[["c_bone", "cn_bfr", "cn_bfr_control"]].copy()


def load_ca_accretion_data(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load and validate historical calcium accretion tracer cohort data.

    Parameters
    ----------
    filepath : Path or str, optional
        Custom path to ca_accretion.csv. If None, loads the bundled dataset.

    Returns
    -------
    pd.DataFrame
        Validated dataframe with columns:
        - 'study': Citation string
        - 'age': Chronological age in years [yr]
        - 'sex': Normalized biological sex ('M' or 'F')
        - 'f': Whole-body calcium accretion flux [kg Ca / year]
        - 'bw': Body weight [kg]
        - 'nbfr': Normalized bone formation rate [yr^-1] (f / bw)
    """
    path = Path(filepath) if filepath is not None else get_bundled_data_path("ca_accretion.csv")
    df = pd.read_csv(path)

    required_cols = {"study", "age", "sex", "f", "bw"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in accretion data: {required_cols - set(df.columns)}"
        )

    # Invariants
    if len(df) != 64:
        raise ValueError(f"Expected exactly 64 observations in accretion data, found {len(df)}")
    if df.isna().any().any():
        raise ValueError("Accretion dataset contains null / NaN values")

    # Normalize sex
    df["sex"] = df["sex"].apply(normalize_sex)

    if (df["age"] <= 0).any():
        raise ValueError("Subject age must be strictly positive")
    if (df["f"] <= 0).any():
        raise ValueError("Calcium accretion flux must be strictly positive")
    if (df["bw"] <= 0).any():
        raise ValueError("Body weight must be strictly positive")

    # Add normalized formation rate nBFR = f / bw [yr^-1]
    df["nbfr"] = df["f"] / df["bw"]

    return df[["study", "age", "sex", "f", "bw", "nbfr"]].copy()


def load_stepan_markers(
    filepath: Path | str | None = None,
    marker: str | None = None,
    sex: str | None = None,
) -> pd.DataFrame:
    """Load and validate Stepan et al. cross-sectional bone remodeling biomarkers.

    Consolidates data across 4 Excel sheets:
    - fig1-female-serum-alp (N=1098)
    - fig1-male-serum-alp (N=951)
    - fig4-female-urine-hp (N=465)
    - fig4-male-urine-hp (N=430)

    Clamps digitized negative ages to a biological lower bound of 0.01 years (~3.65 days).

    Parameters
    ----------
    filepath : Path or str, optional
        Custom path to stepan_bone_markers.xlsx. If None, loads bundled file.
    marker : str, optional
        Filter by biomarker ('alp' or 'hp', case-insensitive).
    sex : str, optional
        Filter by biological sex ('M' or 'F', case-insensitive).

    Returns
    -------
    pd.DataFrame
        Normalized dataframe with columns:
        - 'marker': Marker name ('alp' or 'hp')
        - 'sex': Biological sex ('M' or 'F')
        - 'age': Chronological age clamped to >= 0.01 [years]
        - 'value': Biomarker concentration value
        - 'units': Measurement units ('U/L' or 'mmol/mol')
    """
    path = (
        Path(filepath)
        if filepath is not None
        else get_bundled_data_path("stepan_bone_markers.xlsx")
    )

    sheet_specs = [
        ("fig1-female-serum-alp", "alp", "F", "alp (U/L)", "U/L", 1098),
        ("fig1-male-serum-alp", "alp", "M", "alp (U/L)", "U/L", 951),
        ("fig4-female-urine-hp", "hp", "F", "u-hp (mmol/mol-creat)", "mmol/mol", 465),
        ("fig4-male-urine-hp", "hp", "M", "u-hp (mmol/mol-creat)", "mmol/mol", 430),
    ]

    records: list[pd.DataFrame] = []
    with pd.ExcelFile(path) as excel_file:
        for sheet_name, m_tag, s_tag, val_col, unit_tag, expected_count in sheet_specs:
            raw_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
            if len(raw_sheet) != expected_count:
                raise ValueError(
                    f"Sheet '{sheet_name}' expected {expected_count} rows, found {len(raw_sheet)}"
                )

            age_col = [c for c in raw_sheet.columns if "age" in c.lower()]
            if not age_col:
                raise ValueError(f"No age column found in sheet '{sheet_name}'")

            sheet_df = pd.DataFrame(
                {
                    "marker": m_tag,
                    "sex": s_tag,
                    # Clamp negative digitized ages to 0.01 yr
                    "age": np.maximum(raw_sheet[age_col[0]].astype(float).to_numpy(), 0.01),
                    "value": raw_sheet[val_col].astype(float).to_numpy(),
                    "units": unit_tag,
                }
            )
            records.append(sheet_df)

    combined = pd.concat(records, ignore_index=True)

    if len(combined) != 2944:
        raise ValueError(f"Expected 2944 total Stepan observations, got {len(combined)}")
    if combined.isna().any().any():
        raise ValueError("Stepan markers dataset contains null / NaN values")
    if (combined["value"] <= 0).any():
        raise ValueError("Biomarker values must be strictly positive")
    if (combined["age"] < 0.01).any():
        raise ValueError("Ages must be clamped to >= 0.01 yr")

    # Optional filtering
    if marker is not None:
        m_filter = marker.strip().lower()
        if m_filter not in ("alp", "hp"):
            raise ValueError(f"Unknown marker filter '{marker}'. Expected 'alp' or 'hp'.")
        combined = combined[combined["marker"] == m_filter].reset_index(drop=True)

    if sex is not None:
        s_filter = normalize_sex(sex)
        combined = combined[combined["sex"] == s_filter].reset_index(drop=True)

    return combined


@dataclass(frozen=True)
class PreparedData:
    """Immutable data contract encapsulating all validated modeling datasets.

    Attributes
    ----------
    boivin_turnover : pd.DataFrame
        Validated Boivin histomorphometry dataset (N=15).
    calcium_accretion : pd.DataFrame
        Validated historical calcium accretion dataset (N=64).
    stepan_markers : pd.DataFrame
        Validated Stepan biomarker dataset (N=2,944).
    data_hash : str
        Combined SHA-256 hex digest ensuring data reproducibility.
    """

    boivin_turnover: pd.DataFrame
    calcium_accretion: pd.DataFrame
    stepan_markers: pd.DataFrame
    data_hash: str

    def __post_init__(self) -> None:
        """Enforce immutability and contract validation."""
        if not isinstance(self.boivin_turnover, pd.DataFrame):
            raise TypeError("boivin_turnover must be a pd.DataFrame")
        if not isinstance(self.calcium_accretion, pd.DataFrame):
            raise TypeError("calcium_accretion must be a pd.DataFrame")
        if not isinstance(self.stepan_markers, pd.DataFrame):
            raise TypeError("stepan_markers must be a pd.DataFrame")
        if not isinstance(self.data_hash, str) or not self.data_hash:
            raise ValueError("data_hash must be a non-empty string")


def load_data() -> PreparedData:
    """Load all bundled datasets and return a frozen, validated PreparedData container.

    Returns
    -------
    PreparedData
        Frozen container holding boivin_turnover, calcium_accretion,
        stepan_markers, and composite SHA-256 hash.
    """
    boivin_path = get_bundled_data_path("bone_TO.csv")
    accretion_path = get_bundled_data_path("ca_accretion.csv")
    stepan_path = get_bundled_data_path("stepan_bone_markers.xlsx")

    h = hashlib.sha256()
    h.update(compute_file_sha256(boivin_path).encode("utf-8"))
    h.update(compute_file_sha256(accretion_path).encode("utf-8"))
    h.update(compute_file_sha256(stepan_path).encode("utf-8"))
    composite_hash = h.hexdigest()

    boivin = load_boivin_data(boivin_path)
    accretion = load_ca_accretion_data(accretion_path)
    stepan = load_stepan_markers(stepan_path)

    return PreparedData(
        boivin_turnover=boivin,
        calcium_accretion=accretion,
        stepan_markers=stepan,
        data_hash=composite_hash,
    )
