"""Unit tests for the data loading and schema validation module."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bone_predictor.data import (
    PreparedData,
    compute_file_sha256,
    get_bundled_data_path,
    get_data_dir,
    load_boivin_data,
    load_ca_accretion_data,
    load_data,
    load_stepan_markers,
)


def test_bundled_paths_exist() -> None:
    """Verify bundled data directory and required data files exist."""
    data_dir = get_data_dir()
    assert data_dir.is_dir()

    boivin_path = get_bundled_data_path("bone_TO.csv")
    accretion_path = get_bundled_data_path("ca_accretion.csv")
    stepan_path = get_bundled_data_path("stepan_bone_markers.xlsx")

    assert boivin_path.is_file()
    assert accretion_path.is_file()
    assert stepan_path.is_file()

    with pytest.raises(FileNotFoundError):
        get_bundled_data_path("nonexistent_file.csv")


def test_compute_file_sha256() -> None:
    """Verify deterministic SHA-256 calculation."""
    boivin_path = get_bundled_data_path("bone_TO.csv")
    h1 = compute_file_sha256(boivin_path)
    h2 = compute_file_sha256(boivin_path)
    assert len(h1) == 64
    assert h1 == h2


def test_load_boivin_data() -> None:
    """Verify Boivin histomorphometry dataset loading and schema validation."""
    df = load_boivin_data()

    # Shape and columns
    assert len(df) == 15
    assert list(df.columns) == ["c_bone", "cn_bfr", "cn_bfr_control"]

    # Invariants
    assert not df.isna().any().any()
    assert (df["c_bone"] > 0).all()
    assert (df["cn_bfr"] > 0).all()
    assert (df["cn_bfr_control"] > 0).all()

    # Baseline physiological control point
    control_rows = df[np.isclose(df["c_bone"], 282.0, atol=1e-2)]
    assert len(control_rows) == 1
    assert control_rows["cn_bfr_control"].iloc[0] == pytest.approx(100.0)


def test_load_ca_accretion_data() -> None:
    """Verify historical calcium accretion tracer cohort data loading and schema."""
    df = load_ca_accretion_data()

    # Shape and columns
    assert len(df) == 64
    assert list(df.columns) == ["study", "age", "sex", "f", "bw", "nbfr"]

    # Invariants
    assert not df.isna().any().any()
    assert (df["age"] > 0).all()
    assert (df["f"] > 0).all()
    assert (df["bw"] > 0).all()
    assert set(df["sex"].unique()) == {"M", "F"}

    # Derived normalized formation rate
    assert np.allclose(df["nbfr"], df["f"] / df["bw"])

    # Historical studies
    studies = df["study"].unique()
    assert len(studies) == 6
    assert any("Bauer" in s for s in studies)
    assert any("Abrams" in s for s in studies)


def test_load_stepan_markers_full() -> None:
    """Verify Stepan et al. remodeling biomarker data loading and digitization clamping."""
    df = load_stepan_markers()

    # Total row count
    assert len(df) == 2944
    assert list(df.columns) == ["marker", "sex", "age", "value", "units"]

    # Invariants
    assert not df.isna().any().any()
    assert (df["value"] > 0).all()

    # Digitized negative ages must be clamped to >= 0.01 yr
    assert (df["age"] >= 0.01).all()
    assert df["age"].min() == pytest.approx(0.01)


def test_load_stepan_markers_filtering() -> None:
    """Verify filtering by biomarker and biological sex."""
    # Filter by marker
    df_alp = load_stepan_markers(marker="alp")
    assert len(df_alp) == 1098 + 951
    assert set(df_alp["marker"].unique()) == {"alp"}
    assert set(df_alp["units"].unique()) == {"U/L"}

    df_hp = load_stepan_markers(marker="hp")
    assert len(df_hp) == 465 + 430
    assert set(df_hp["marker"].unique()) == {"hp"}
    assert set(df_hp["units"].unique()) == {"mmol/mol"}

    # Filter by sex
    df_female = load_stepan_markers(sex="F")
    assert len(df_female) == 1098 + 465
    assert set(df_female["sex"].unique()) == {"F"}

    df_male = load_stepan_markers(sex="M")
    assert len(df_male) == 951 + 430
    assert set(df_male["sex"].unique()) == {"M"}

    # Filter both
    df_f_alp = load_stepan_markers(marker="alp", sex="F")
    assert len(df_f_alp) == 1098


def test_load_stepan_markers_invalid_filter() -> None:
    """Verify rejection of invalid marker or sex filter."""
    with pytest.raises(ValueError, match="Unknown marker"):
        load_stepan_markers(marker="invalid_marker")
    with pytest.raises(ValueError, match="Unknown sex"):
        load_stepan_markers(sex="invalid_sex")


def test_load_data_container_and_immutability() -> None:
    """Verify load_data() produces a frozen PreparedData dataclass with valid hash."""
    prepared = load_data()
    assert isinstance(prepared, PreparedData)
    assert len(prepared.boivin_turnover) == 15
    assert len(prepared.calcium_accretion) == 64
    assert len(prepared.stepan_markers) == 2944

    assert isinstance(prepared.data_hash, str)
    assert len(prepared.data_hash) == 64

    # Frozen dataclass immutability
    with pytest.raises(FrozenInstanceError):
        prepared.data_hash = "tampered_hash"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        prepared.boivin_turnover = pd.DataFrame()  # type: ignore[misc]


def test_boivin_validation_errors(tmp_path: Path) -> None:
    """Verify error detection for malformed Boivin data."""
    bad_csv = tmp_path / "bad_boivin.csv"

    # Missing column
    pd.DataFrame({"c_bone": [282.0], "cn_bfr": [0.076]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_boivin_data(bad_csv)

    # Wrong length
    pd.DataFrame(
        {"c_bone": [282.0] * 5, "cn_bfr": [0.076] * 5, "cn_bfr_control": [100.0] * 5}
    ).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Expected exactly 15 observations"):
        load_boivin_data(bad_csv)

    # Non-positive values
    pd.DataFrame(
        {"c_bone": [-10.0] * 15, "cn_bfr": [0.076] * 15, "cn_bfr_control": [100.0] * 15}
    ).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="strictly positive"):
        load_boivin_data(bad_csv)


def test_load_stepan_markers_file_handle_closed(tmp_path: Path) -> None:
    """Verify load_stepan_markers() closes ExcelFile handle on Windows."""
    import shutil

    temp_excel = tmp_path / "temp_stepan.xlsx"
    bundled_path = get_bundled_data_path("stepan_bone_markers.xlsx")
    shutil.copyfile(bundled_path, temp_excel)

    df = load_stepan_markers(temp_excel)
    assert len(df) == 2944

    # On Windows, deleting an open file raises PermissionError [WinError 32].
    # If the context manager properly closed the file, unlink succeeds immediately.
    temp_excel.unlink()
    assert not temp_excel.exists()
