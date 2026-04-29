"""Native Rust-backed Python training conformance tests."""

from __future__ import annotations

import numpy as np
import pytest

from pymars import Earth, runtime


def test_rust_training_bridge_uses_compiled_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python training path should hit the compiled Rust backend when available."""
    if runtime._rust_backend is None:
        pytest.skip("Compiled pymars_runtime extension is not available")

    monkeypatch.setenv("PYMARS_USE_RUST_TRAINING", "1")

    model = Earth(max_terms=5, max_degree=1, penalty=3.0)
    fitted = model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))

    assert fitted is model
    assert model.fitted_

    exported = model.export_model()
    assert "basis_terms" in exported
    assert "coefficients" in exported

    predictions = model.predict(np.array([[0.0], [1.0], [2.0]]))
    np.testing.assert_allclose(predictions, np.array([1.0, 3.0, 5.0]), atol=1e-12)
