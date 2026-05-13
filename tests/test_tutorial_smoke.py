"""Tutorial smoke tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pymars as earth

if TYPE_CHECKING:
    from pathlib import Path


def _tutorial_sample() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-1.0, 0.0, 0.5],
            [-0.2, 0.3, 0.0],
            [0.1, -0.4, 0.2],
            [0.5, 0.2, -0.3],
            [1.0, -0.7, 0.8],
            [1.4, -1.1, 1.1],
        ],
        dtype=float,
    )
    y = np.array([0.8, 1.3, 2.0, 2.6, 4.9, 6.2], dtype=float)
    return X, y


def test_basic_tutorial_workflow(tmp_path: Path) -> None:
    """Exercise the basic tutorial workflow end to end."""
    X, y = _tutorial_sample()
    model = earth.Earth(max_degree=1, penalty=3.0)
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],)
    assert np.isfinite(model.score(X, y))

    spec_path = tmp_path / "model.json"
    saved_path = earth.save_model(model, spec_path)
    assert saved_path == spec_path
    assert spec_path.exists()

    validated_spec = earth.validate(spec_path)
    restored_model = earth.load_model(validated_spec)
    np.testing.assert_allclose(restored_model.predict(X), predictions)
    np.testing.assert_allclose(earth.predict(validated_spec, X), predictions)

    design_matrix = earth.design_matrix(validated_spec, X)
    assert design_matrix.shape[0] == X.shape[0]

    inspection = earth.inspect(validated_spec)
    assert isinstance(inspection, dict)
    assert inspection["n_features"] == X.shape[1]


def test_advanced_tutorial_workflow() -> None:
    """Exercise the advanced tutorial explanation workflow."""
    X, y = _tutorial_sample()
    model = earth.Earth(max_degree=1, penalty=3.0)
    model.fit(X, y)

    explanation = earth.get_model_explanation(model, X)
    assert "model_summary" in explanation
    assert explanation["model_summary"]["n_features"] == X.shape[1]
    assert isinstance(explanation["basis_functions"], list)
