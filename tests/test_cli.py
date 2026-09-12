"""Unit tests for bone-predictor CLI subcommands (cli.py).

Verifies:
1. create_parser configuration and subcommands.
2. Parameter parsing for fit-turnover, fit-biomarkers, predict, generate-grid, validate.
3. Execution of predict subcommand with table output and file export (CSV/JSON).
4. Execution of generate-grid subcommand with file export and schema validation.
5. Execution of validate subcommand.
6. Execution of fit-turnover and fit-biomarkers with mocked fitting.
7. Help and empty invocation handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bone_predictor.cli import create_parser, main


# ==============================================================================
# 1. Parser Structure & Arguments
# ==============================================================================
def test_create_parser_subcommands() -> None:
    """Verify create_parser registers all 5 required subcommands."""
    parser = create_parser()
    assert parser.prog == "bone-predictor"

    # Test subcommand dispatch
    subparsers_actions = [action for action in parser._actions if action.dest == "subcommand"]
    assert len(subparsers_actions) == 1
    subcommands = set(subparsers_actions[0].choices.keys())
    expected = {"fit-turnover", "fit-biomarkers", "predict", "generate-grid", "validate"}
    assert expected == subcommands


def test_parser_fit_turnover_args() -> None:
    """Verify arguments for fit-turnover subcommand."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "fit-turnover",
            "--model",
            "hormesis",
            "--output",
            "test_post.nc",
            "--chains",
            "2",
            "--draws",
            "500",
            "--tune",
            "200",
        ]
    )
    assert args.subcommand == "fit-turnover"
    assert args.model == "hormesis"
    assert args.output == "test_post.nc"
    assert args.chains == 2
    assert args.draws == 500
    assert args.tune == 200


def test_parser_fit_biomarkers_args() -> None:
    """Verify arguments for fit-biomarkers subcommand."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "fit-biomarkers",
            "--components",
            "4",
            "--marker",
            "alp",
            "--output",
            "test_bio.nc",
            "--chains",
            "3",
            "--draws",
            "600",
        ]
    )
    assert args.subcommand == "fit-biomarkers"
    assert args.components == 4
    assert args.marker == "alp"
    assert args.output == "test_bio.nc"
    assert args.chains == 3
    assert args.draws == 600


def test_parser_predict_args() -> None:
    """Verify arguments for predict subcommand."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "predict",
            "--age",
            "45.5",
            "--sex",
            "Female",
            "--c-bone",
            "1200.0",
            "--output",
            "pred.csv",
        ]
    )
    assert args.subcommand == "predict"
    assert args.age == 45.5
    assert args.sex == "Female"
    assert args.c_bone == 1200.0
    assert args.output == "pred.csv"


def test_parser_generate_grid_args() -> None:
    """Verify arguments for generate-grid subcommand."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "generate-grid",
            "--age-min",
            "5.0",
            "--age-max",
            "65.0",
            "--step",
            "1.0",
            "--sex",
            "M",
            "--c-bone",
            "500.0",
            "--output",
            "grid.csv",
        ]
    )
    assert args.subcommand == "generate-grid"
    assert args.age_min == 5.0
    assert args.age_max == 65.0
    assert args.step == 1.0
    assert args.sex == "M"
    assert args.c_bone == 500.0
    assert args.output == "grid.csv"


# ==============================================================================
# 2. Main Entry Point: Help & Default Execution
# ==============================================================================
def test_main_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify invoking main without arguments prints help and returns 0."""
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage: bone-predictor" in captured.out


def test_main_help_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify invoking main with --help prints help and exits cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Mechanistic modeling for bone turnover modifiers" in captured.out


# ==============================================================================
# 3. Subcommand Execution: Validate
# ==============================================================================
def test_main_validate_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify validate subcommand runs all self-diagnostics and returns 0."""
    ret = main(["validate"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SUCCESS: All models, schemas, contracts" in captured.out


# ==============================================================================
# 4. Subcommand Execution: Predict
# ==============================================================================
def test_main_predict_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify predict subcommand prints nicely formatted table to stdout."""
    ret = main(["predict", "--age", "30", "--sex", "M", "--c-bone", "282.0"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Bone Turnover & Hawkins PET Prediction (Age: 30.0 yr, Sex: M)" in captured.out
    assert "r_form   (RAI_ALP) : 1.0000" in captured.out
    assert "FBFR  (Whole-skeleton): 0.10000" in captured.out
    assert "Ki : 0.03588 mL/min/mL" in captured.out


def test_main_predict_csv_export(tmp_path: Path) -> None:
    """Verify predict subcommand exports results to CSV file."""
    out_file = tmp_path / "sub" / "prediction.csv"
    ret = main(["predict", "--age", "25", "--sex", "F", "--output", str(out_file)])
    assert ret == 0
    assert out_file.is_file()

    df = pd.read_csv(out_file)
    assert len(df) == 1
    assert "r_form" in df.columns
    assert "FBFR" in df.columns
    assert "Ki" in df.columns
    assert df["sex"].iloc[0] == "F"


def test_main_predict_json_export(tmp_path: Path) -> None:
    """Verify predict subcommand exports results to JSON file."""
    out_file = tmp_path / "prediction.json"
    ret = main(["predict", "--age", "12", "--sex", "M", "--output", str(out_file)])
    assert ret == 0
    assert out_file.is_file()

    df = pd.read_json(out_file)
    assert len(df) == 1
    assert "net_accretion" in df.columns


# ==============================================================================
# 5. Subcommand Execution: Generate-Grid
# ==============================================================================
def test_main_generate_grid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify generate-grid exports dense CSV table and prints row summary."""
    out_file = tmp_path / "grid.csv"
    ret = main(
        [
            "generate-grid",
            "--age-min",
            "10.0",
            "--age-max",
            "30.0",
            "--step",
            "5.0",
            "--sex",
            "both",
            "--c-bone",
            "282.0",
            "--output",
            str(out_file),
        ]
    )
    assert ret == 0
    assert out_file.is_file()

    captured = capsys.readouterr()
    assert "Saved turnover modifier grid to:" in captured.out

    df = pd.read_csv(out_file)
    # ages: 10, 15, 20, 25, 30 => 5 points * 2 sexes = 10 rows
    assert len(df) == 10
    assert set(df["sex"].unique()) == {"F", "M"}
    assert "Ki" in df.columns
    assert "FBFR" in df.columns


# ==============================================================================
# 6. Subcommand Execution: Fit-Turnover & Fit-Biomarkers (Mocked MCMC)
# ==============================================================================
def test_main_fit_turnover_mocked(tmp_path: Path) -> None:
    """Verify fit-turnover invokes model.fit and model.save with configured parameters."""
    out_nc = tmp_path / "mock_turnover.nc"

    with patch("bone_predictor.cli.TurnoverModel") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        ret = main(
            [
                "fit-turnover",
                "--model",
                "4p_hill",
                "--output",
                str(out_nc),
                "--chains",
                "2",
                "--draws",
                "50",
                "--tune",
                "25",
            ]
        )
        assert ret == 0
        mock_cls.assert_called_once_with(formulation="4p_hill", data=None)
        mock_instance.fit.assert_called_once_with(draws=50, chains=2, tune=25)
        mock_instance.save.assert_called_once_with(str(out_nc))


def test_main_fit_biomarkers_mocked(tmp_path: Path) -> None:
    """Verify fit-biomarkers invokes model.fit and model.save with configured parameters."""
    out_nc = tmp_path / "mock_biomarkers.nc"

    with patch("bone_predictor.cli.BiomarkerModel") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        ret = main(
            [
                "fit-biomarkers",
                "--components",
                "3",
                "--marker",
                "joint",
                "--output",
                str(out_nc),
                "--chains",
                "2",
                "--draws",
                "100",
                "--tune",
                "50",
            ]
        )
        assert ret == 0
        mock_cls.assert_called_once_with(n_components=3, marker="joint", data=None)
        mock_instance.fit.assert_called_once_with(draws=100, chains=2, tune=50)
        mock_instance.save.assert_called_once_with(str(out_nc))
