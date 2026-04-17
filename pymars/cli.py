from __future__ import annotations

"""Command line interface for pymars."""

import argparse
import importlib
import logging
import pickle
from pathlib import Path
from typing import Any, cast

from . import Earth, __version__

logger = logging.getLogger(__name__)


def main() -> None:
    """Command line interface for pymars."""
    parser = argparse.ArgumentParser(
        description="pymars: Pure Python Multivariate Adaptive Regression Splines"
    )
    parser.add_argument("--version", action="version", version=f"pymars {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit an Earth model")
    fit_parser.add_argument("--input", required=True, help="Input data file (CSV)")
    fit_parser.add_argument("--target", required=True, help="Target column name")
    fit_parser.add_argument(
        "--output-model", required=True, help="Output model file (pickle)"
    )
    fit_parser.add_argument(
        "--max-degree",
        type=int,
        default=1,
        help="Maximum degree of interaction terms (default: 1)",
    )
    fit_parser.add_argument(
        "--penalty",
        type=float,
        default=3.0,
        help="GCV penalty parameter (default: 3.0)",
    )
    fit_parser.add_argument(
        "--max-terms", type=int, help="Maximum number of terms (default: None)"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Make predictions with a fitted model"
    )
    predict_parser.add_argument(
        "--model", required=True, help="Input model file (pickle)"
    )
    predict_parser.add_argument("--input", required=True, help="Input data file (CSV)")
    predict_parser.add_argument(
        "--output", required=True, help="Output predictions file (CSV)"
    )

    # Score command
    score_parser = subparsers.add_parser("score", help="Score a fitted model")
    score_parser.add_argument(
        "--model", required=True, help="Input model file (pickle)"
    )
    score_parser.add_argument("--input", required=True, help="Input data file (CSV)")
    score_parser.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()

    if args.command == "fit":
        fit_model(args)
    elif args.command == "predict":
        make_predictions(args)
    elif args.command == "score":
        score_model(args)
    else:
        parser.print_help()


def fit_model(args: argparse.Namespace) -> None:
    """Fit an Earth model from command line arguments."""
    # Import pandas only when needed
    pd = cast(Any, importlib.import_module("pandas"))

    # Load data
    data = pd.read_csv(args.input)
    if args.target not in data.columns:
        raise ValueError(f"Target column '{args.target}' not found in data")

    X = data.drop(columns=[args.target]).values
    y = data[args.target].values

    # Create and fit model
    model_params = {
        "max_degree": args.max_degree,
        "penalty": args.penalty,
    }
    if args.max_terms is not None:
        model_params["max_terms"] = args.max_terms

    earth: Any = Earth(**model_params)
    earth.fit(X, y)

    # Save model
    with Path(args.output_model).open("wb") as f:
        pickle.dump(earth, f)

    print(f"Model fitted and saved to {args.output_model}")
    basis_functions = earth.basis_ or []
    print(f"Number of selected basis functions: {len(basis_functions)}")
    print(f"GCV score: {earth.gcv_:.4f}")
    print(f"R² score: {earth.score(X, y):.4f}")


def make_predictions(args: argparse.Namespace) -> None:
    """Make predictions with a fitted model."""
    # Import pandas only when needed
    pd = cast(Any, importlib.import_module("pandas"))

    # Load model
    with Path(args.model).open("rb") as f:
        model: Any = pickle.load(f)

    # Load input data
    data = pd.read_csv(args.input)
    X = data.values

    # Make predictions
    predictions = model.predict(X)

    # Save predictions
    pred_df = pd.DataFrame({"prediction": predictions})
    pred_df.to_csv(args.output, index=False)

    print(f"Predictions made and saved to {args.output}")


def score_model(args: argparse.Namespace) -> None:
    """Score a fitted model."""
    # Import pandas only when needed
    pd = cast(Any, importlib.import_module("pandas"))

    # Load model
    with Path(args.model).open("rb") as f:
        model: Any = pickle.load(f)

    # Load data
    data = pd.read_csv(args.input)
    if args.target not in data.columns:
        raise ValueError(f"Target column '{args.target}' not found in data")

    X = data.drop(columns=[args.target]).values
    y = data[args.target].values

    # Calculate score
    score = model.score(X, y)

    print(f"Model R² score: {score:.4f}")


if __name__ == "__main__":
    main()
