"""Deterministic numerical regression cases for Earth models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pymars import Earth

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "reference_regression_cases.json"


def _build_case_inputs(
    case_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if case_name == "piecewise_1d":
        X = np.array(
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
            dtype=float,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.0], dtype=float)
        probe = np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [6.5], [8.5]], dtype=float)
        return X, y, probe, None

    if case_name == "mixed_3d":
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = 2 * X[:, 0] + np.sin(np.pi * X[:, 1]) - X[:, 2] ** 2 + rng.randn(50) * 0.1
        probe = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.25, 0.5, 0.75],
                [0.9, 0.1, 0.4],
                [0.7, 0.8, 0.2],
            ],
            dtype=float,
        )
        return X, y, probe, None

    if case_name == "sample_weight_1d":
        X = np.array(
            [[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]],
            dtype=float,
        )
        y = np.array([0.0, 0.2, 0.1, 0.0, 20.0, 20.2, 19.8, 20.0], dtype=float)
        probe = np.array([[0.5], [2.5], [10.5], [12.5]], dtype=float)
        sample_weight = np.array(
            [1.0, 1.0, 1.0, 1.0, 40.0, 40.0, 40.0, 40.0],
            dtype=float,
        )
        return X, y, probe, sample_weight

    if case_name == "grid_2d":
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [4.0, 0.0],
                [4.0, 1.0],
            ],
            dtype=float,
        )
        y = np.array([0.0, 0.7, 1.0, 1.6, 2.0, 2.7, 3.4, 4.1, 4.8, 5.5], dtype=float)
        probe = np.array(
            [
                [0.5, 0.25],
                [1.5, 0.75],
                [2.5, 0.25],
                [3.5, 0.75],
            ],
            dtype=float,
        )
        return X, y, probe, None

    if case_name == "categorical_2d":
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=float,
        )
        y = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0], dtype=float)
        probe = np.array(
            [
                [0.0, 0.5],
                [1.0, 1.5],
                [2.0, 0.5],
            ],
            dtype=float,
        )
        return X, y, probe, None

    if case_name == "missingness_2d":
        X = np.array(
            [
                [1.0, 2.0],
                [np.nan, 3.0],
                [3.0, np.nan],
                [4.0, 5.0],
            ],
            dtype=float,
        )
        y = np.array([0.0, 100.0, -50.0, 0.0], dtype=float)
        probe = np.array(
            [
                [1.0, 2.0],
                [np.nan, 3.0],
                [3.0, np.nan],
                [2.0, 4.0],
            ],
            dtype=float,
        )
        return X, y, probe, None

    if case_name == "interaction_2d":
        x0 = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
            ],
            dtype=float,
        )
        x1 = np.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                0.0,
                1.0,
                2.0,
                3.0,
                0.0,
                1.0,
                2.0,
                3.0,
                0.0,
                1.0,
                2.0,
                3.0,
            ],
            dtype=float,
        )
        X = np.column_stack([x0, x1])
        y = (
            1.0
            + 0.5 * x0
            + 0.25 * x1
            + 1.5 * np.maximum(0.0, x0 - 1.5) * np.maximum(0.0, x1 - 1.0)
        )
        probe = np.array(
            [
                [0.5, 0.5],
                [1.5, 2.0],
                [2.5, 2.5],
                [3.0, 1.0],
            ],
            dtype=float,
        )
        return X, y, probe, None

    if case_name == "term_pressure_2d":
        x1 = np.linspace(-1.5, 1.5, 5)
        x2 = np.linspace(-1.0, 1.0, 5)
        X = np.array([(a, b) for a in x1 for b in x2], dtype=float)
        y = (
            1.0
            + 1.2 * np.maximum(0, X[:, 0] + 0.2)
            - 0.9 * np.maximum(0, X[:, 0] - 0.7)
            + 0.8 * np.maximum(0, X[:, 1] + 0.1)
        )
        probe = np.array(
            [
                [-1.25, -0.75],
                [0.0, 0.0],
                [1.0, 0.75],
            ],
            dtype=float,
        )
        return X, y, probe, None

    if case_name == "hinge_only_1d":
        X = np.array(
            [[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0], [3.0]],
            dtype=float,
        )
        y = np.array([5.0, 2.0, 1.0, 0.5, 1.0, 2.5, 5.0, 9.0], dtype=float)
        probe = np.array([[-1.5], [-0.25], [0.75], [2.5]], dtype=float)
        return X, y, probe, None

    if case_name == "penalty_sensitive_1d":
        x = np.linspace(-3.0, 3.0, 25)
        X = x.reshape(-1, 1)
        y = (
            0.8 * x
            + 1.2 * np.maximum(0.0, x + 1.0)
            - 1.5 * np.maximum(0.0, x - 0.5)
            + 0.9 * np.maximum(0.0, x - 1.8)
        )
        probe = np.array([[-2.5], [-0.25], [1.25], [2.5]], dtype=float)
        return X, y, probe, None

    raise KeyError(case_name)


def test_reference_regression_cases():
    """Lock down deterministic outputs for representative fitted models."""
    cases = json.loads(FIXTURE_PATH.read_text())

    for case_name, expected in cases.items():
        X, y, probe, sample_weight = _build_case_inputs(case_name)
        model = Earth(**expected["params"])
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X, y, **fit_kwargs)

        assert [str(bf) for bf in model.basis_] == expected["basis"]
        np.testing.assert_allclose(model.coef_, np.array(expected["coef"]), atol=1e-12)
        np.testing.assert_allclose(
            model.predict(probe), np.array(expected["predictions"]), atol=1e-12
        )
        assert np.isclose(model.gcv_, expected["gcv"], atol=1e-12)
        assert np.isclose(model.rss_, expected["rss"], atol=1e-12)
        assert np.isclose(model.mse_, expected["mse"], atol=1e-12)
