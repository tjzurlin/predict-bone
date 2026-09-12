"""Command line interface (CLI) for bone-predictor."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bone_predictor.biomarkers import BiomarkerModel
from bone_predictor.data import load_boivin_data, load_ca_accretion_data, load_stepan_markers
from bone_predictor.hawkins import HawkinsCalibrator
from bone_predictor.turnover import TurnoverModel

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create top-level argument parser with 5 subcommands."""
    parser = argparse.ArgumentParser(
        prog="bone-predictor",
        description="Mechanistic modeling for bone turnover modifiers and Hawkins PET kinetics.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. fit-turnover
    p_turnover = subparsers.add_parser(
        "fit-turnover", help="Fit bone turnover suppression model (r_TO from Boivin data)"
    )
    p_turnover.add_argument(
        "--data", type=str, default=None, help="Optional path to custom Boivin CSV data"
    )
    p_turnover.add_argument(
        "--model",
        choices=["4p_hill", "log_linear", "hormesis", "hill_2p"],
        default="4p_hill",
        help="Turnover model formulation (default: 4p_hill)",
    )
    p_turnover.add_argument(
        "--output", type=str, default="posterior/turnover_posterior.nc", help="Output path"
    )
    p_turnover.add_argument("--chains", type=int, default=4, help="MCMC chains (default: 4)")
    p_turnover.add_argument("--draws", type=int, default=1000, help="MCMC draws (default: 1000)")
    p_turnover.add_argument(
        "--tune", type=int, default=1000, help="MCMC tuning steps (default: 1000)"
    )

    # 2. fit-biomarkers
    p_bio = subparsers.add_parser(
        "fit-biomarkers", help="Fit biomarker spectral deconvolution (ALP/HP from Stepan data)"
    )
    p_bio.add_argument(
        "--data", type=str, default=None, help="Optional path to custom Stepan dataset"
    )
    p_bio.add_argument(
        "--components", type=int, choices=[2, 3, 4], default=3, help="Component count (default: 3)"
    )
    p_bio.add_argument(
        "--marker", choices=["alp", "hp", "joint"], default="joint", help="Target marker"
    )
    p_bio.add_argument(
        "--output", type=str, default="posterior/biomarker_posterior.nc", help="Output path"
    )
    p_bio.add_argument("--chains", type=int, default=4, help="MCMC chains (default: 4)")
    p_bio.add_argument("--draws", type=int, default=1000, help="MCMC draws (default: 1000)")
    p_bio.add_argument("--tune", type=int, default=1000, help="MCMC tuning steps (default: 1000)")

    # 3. predict
    p_pred = subparsers.add_parser(
        "predict", help="Predict Hawkins PET parameters and calcium fluxes for age/sex"
    )
    p_pred.add_argument("--age", type=float, default=30.0, help="Age in years (default: 30.0)")
    p_pred.add_argument(
        "--sex", choices=["M", "F", "Male", "Female", "m", "f"], default="M", help="Biological sex"
    )
    p_pred.add_argument(
        "--c-bone", type=float, default=282.0, help="Bone fluoride conc [mg/kg ash]"
    )
    p_pred.add_argument(
        "--output", type=str, default=None, help="Optional path to write prediction CSV/JSON"
    )

    # 4. generate-grid
    p_grid = subparsers.add_parser(
        "generate-grid", help="Generate dense lookup grid across lifespan"
    )
    p_grid.add_argument("--age-min", type=float, default=0.0, help="Minimum age [yr]")
    p_grid.add_argument("--age-max", type=float, default=85.0, help="Maximum age [yr]")
    p_grid.add_argument("--step", type=float, default=0.5, help="Age step [yr]")
    p_grid.add_argument(
        "--sex", choices=["both", "M", "F", "Male", "Female"], default="both", help="Sex"
    )
    p_grid.add_argument(
        "--c-bone", type=float, default=282.0, help="Bone fluoride conc [mg/kg ash]"
    )
    p_grid.add_argument(
        "--output", type=str, default="outputs/bone_turnover_grid.csv", help="Output path"
    )

    # 5. validate
    subparsers.add_parser("validate", help="Run prior predictive checks and schema validations")

    return parser


def handle_fit_turnover(args: argparse.Namespace) -> int:
    """Execute turnover model fitting."""
    data_df = pd.read_csv(args.data) if args.data else None
    model = TurnoverModel(formulation=args.model, data=data_df)
    print(
        f"Fitting TurnoverModel (formulation={args.model}) with "
        f"{args.chains} chains, {args.draws} draws..."
    )
    model.fit(draws=args.draws, chains=args.chains, tune=args.tune)
    model.save(args.output)
    print(f"TurnoverModel posterior successfully saved to: {args.output}")
    return 0


def handle_fit_biomarkers(args: argparse.Namespace) -> int:
    """Execute biomarker spectral deconvolution fitting."""
    data_df = None
    if args.data:
        p = Path(args.data)
        data_df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    model = BiomarkerModel(n_components=args.components, marker=args.marker, data=data_df)
    print(
        f"Fitting BiomarkerModel (components={args.components}, marker={args.marker}) "
        f"with {args.chains} chains, {args.draws} draws..."
    )
    model.fit(draws=args.draws, chains=args.chains, tune=args.tune)
    model.save(args.output)
    print(f"BiomarkerModel posterior successfully saved to: {args.output}")
    return 0


def handle_predict(args: argparse.Namespace) -> int:
    """Execute prediction of Hawkins kinetics and calcium fluxes."""
    calibrator = HawkinsCalibrator()
    mods = calibrator.predict_modifiers(age=args.age, sex=args.sex, c_bone=args.c_bone)

    print("=" * 60)
    print(f"Bone Turnover & Hawkins PET Prediction (Age: {args.age:.1f} yr, Sex: {mods['sex']})")
    print(f"Trabecular Fluoride (C_bone): {args.c_bone:.1f} mg/kg ash")
    print("-" * 60)
    print("Relative Activity Indices:")
    print(f"  r_form   (RAI_ALP) : {mods['r_form']:.4f}")
    print(f"  r_resorp (RAI_HP)  : {mods['r_resorp']:.4f}")
    print(f"  r_to     (Modifier): {mods['r_to']:.4f}")
    print("Fractional Turnover Rates [yr^-1]:")
    print(f"  FBFR  (Whole-skeleton): {mods['FBFR']:.5f}")
    print(f"  TFBFR (Trabecular)    : {mods['TFBFR']:.5f}")
    print(f"  CFBFR (Cortical)      : {mods['CFBFR']:.5f}")
    print("Calcium Fluxes [kg Ca/year]:")
    print(f"  f_form        : {mods['f_form']:.5f}")
    print(f"  f_resorp      : {mods['f_resorp']:.5f}")
    print(f"  net_accretion : {mods['net_accretion']:+.5f}")
    print("Hawkins [18F]Fluoride PET Microparameters:")
    print(f"  k3 : {mods['k3']:.5f} min^-1")
    print(f"  k4 : {mods['k4']:.5f} min^-1")
    print(f"  Ki : {mods['Ki']:.5f} mL/min/mL")
    print("=" * 60)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Export as clean row
        export_dict = {k: v for k, v in mods.items() if k not in ("fbfr", "tfbfr", "cfbfr", "ki")}
        df = pd.DataFrame([export_dict])
        if out_p.suffix.lower() == ".json":
            df.to_json(out_p, indent=2, orient="records")
        else:
            df.to_csv(out_p, index=False)
        print(f"Prediction results saved to: {out_p}")

    return 0


def handle_generate_grid(args: argparse.Namespace) -> int:
    """Generate dense turnover grid across lifespan."""
    calibrator = HawkinsCalibrator()
    df = calibrator.modifier_grid(
        age_min=args.age_min,
        age_max=args.age_max,
        step=args.step,
        sex=args.sex,
        c_bone=args.c_bone,
    )
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Generated {len(df)} grid rows across age [{args.age_min}, {args.age_max}] yr.")
    print(f"Saved turnover modifier grid to: {out_p}")
    return 0


def handle_validate() -> int:
    """Run model and contract self-diagnostics."""
    print("Running bone-predictor self-diagnostics and contract validation...")

    # 1. Validate bundled datasets
    print("  1/4 Validating bundled datasets...")
    df_boivin = load_boivin_data()
    if len(df_boivin) != 15:
        raise ValueError(f"Expected 15 Boivin observations, got {len(df_boivin)}")

    df_stepan = load_stepan_markers()
    if len(df_stepan) != 2944:
        raise ValueError(f"Expected 2944 Stepan observations, got {len(df_stepan)}")

    df_ca = load_ca_accretion_data()
    if len(df_ca) != 64:
        raise ValueError(f"Expected 64 calcium accretion observations, got {len(df_ca)}")

    # 2. Validate TurnoverModel compilation
    print("  2/4 Validating TurnoverModel compilation...")
    t_model = TurnoverModel(formulation="4p_hill", data=df_boivin)
    pm_t = t_model.build()
    if pm_t is None:
        raise RuntimeError("Turnover PyMC model compilation returned None")

    # 3. Validate BiomarkerModel compilation
    print("  3/4 Validating BiomarkerModel compilation...")
    b_model = BiomarkerModel(n_components=3, marker="joint", data=df_stepan.head(100))
    pm_b = b_model.build()
    if pm_b is None:
        raise RuntimeError("Biomarker PyMC model compilation returned None")

    # 4. Validate HawkinsCalibrator & PBKM bridge
    print("  4/4 Validating HawkinsCalibrator physiological baseline...")
    calibrator = HawkinsCalibrator()

    mods_adult_m = calibrator.predict_modifiers(30.0, "M", c_bone=282.0)
    if not np.isclose(mods_adult_m["r_form"], 1.0, atol=0.02):
        raise ValueError(f"Adult r_form={mods_adult_m['r_form']} expected ~1.0")
    if not np.isclose(mods_adult_m["r_resorp"], 1.0, atol=0.02):
        raise ValueError(f"Adult r_resorp={mods_adult_m['r_resorp']} expected ~1.0")
    if not np.isclose(mods_adult_m["FBFR"], 0.10, atol=0.01):
        raise ValueError(f"Adult FBFR={mods_adult_m['FBFR']} expected ~0.10")
    if not np.isclose(mods_adult_m["net_accretion"], 0.0, atol=0.03):
        raise ValueError(f"Adult net accretion={mods_adult_m['net_accretion']} expected ~0")

    mods_infant = calibrator.predict_modifiers(0.5, "F")
    if mods_infant["r_form"] <= 5.0:
        raise ValueError(f"Infant r_form={mods_infant['r_form']} expected > 5.0")
    if mods_infant["net_accretion"] <= 0.0:
        raise ValueError(f"Infant net accretion={mods_infant['net_accretion']} expected > 0")

    print("SUCCESS: All models, schemas, contracts, and physiological baselines validated.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help()
        return 0

    if args.subcommand == "fit-turnover":
        return handle_fit_turnover(args)
    if args.subcommand == "fit-biomarkers":
        return handle_fit_biomarkers(args)
    if args.subcommand == "predict":
        return handle_predict(args)
    if args.subcommand == "generate-grid":
        return handle_generate_grid(args)
    if args.subcommand == "validate":
        return handle_validate()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
