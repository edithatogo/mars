"""Distributed runtime replay tests."""

from __future__ import annotations

import numpy as np
import pytest

from pymars import Earth, runtime


@pytest.fixture
def linear_spec() -> dict:
    """Fit a deterministic model used for replay comparison tests."""
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
            [6.0, 12.0],
            [7.0, 14.0],
        ],
        dtype=float,
    )
    y = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0], dtype=float)
    model = Earth(max_degree=1, max_terms=10, penalty=3.0)
    model.fit(X, y)
    return model.get_model_spec()


def test_predict_distributed_matches_serial_output(linear_spec: dict) -> None:
    """Distributed preview path must be output-identical to serial predict."""
    x = np.array([[0.5, 1.0], [2.5, 5.0], [4.1, 8.2], [10.0, 20.0]], dtype=float)
    serial = runtime.predict(linear_spec, x)
    parallel = runtime.predict_distributed(linear_spec, x, workers=1)
    assert np.allclose(parallel, serial)


def test_predict_distributed_preserves_order_with_multiple_workers(
    linear_spec: dict,
) -> None:
    """Chunked local execution must preserve contiguous row order."""
    x = np.linspace(0.0, 1.0, 64 * 2).reshape(64, 2)
    ordered = runtime.predict(linear_spec, x)
    distributed = runtime.predict_distributed(
        linear_spec, x, workers=4, chunk_size=7, preserve_order=True
    )
    assert np.allclose(distributed, ordered)


def test_design_matrix_distributed_matches_design_matrix(linear_spec: dict) -> None:
    """Distributed design matrix evaluation must match the serial baseline."""
    x = np.array(
        [[0.0, 1.0], [1.5, 3.0], [2.0, 4.0], [9.0, 18.0], [11.0, 22.0]],
        dtype=float,
    )
    serial = runtime.design_matrix(linear_spec, x)
    distributed = runtime.design_matrix_distributed(
        linear_spec, x, workers=3, chunk_size=2
    )
    assert distributed.shape == serial.shape
    assert np.allclose(distributed, serial)


def test_cpu_cluster_predict_matches_serial_output(linear_spec: dict) -> None:
    """CPU cluster replay must match the serial prediction path."""
    x = np.array([[0.5, 1.0], [2.5, 5.0], [4.1, 8.2], [10.0, 20.0]], dtype=float)
    serial = runtime.predict(linear_spec, x)
    cluster = runtime.predict_cpu_cluster(linear_spec, x, workers=2, chunk_size=2)
    assert np.allclose(cluster, serial)


def test_cpu_cluster_design_matrix_matches_serial(linear_spec: dict) -> None:
    """CPU cluster replay must match the serial design-matrix path."""
    x = np.array(
        [[0.0, 1.0], [1.5, 3.0], [2.0, 4.0], [9.0, 18.0], [11.0, 22.0]],
        dtype=float,
    )
    serial = runtime.design_matrix(linear_spec, x)
    cluster = runtime.design_matrix_cpu_cluster(linear_spec, x, workers=2, chunk_size=2)
    assert cluster.shape == serial.shape
    assert np.allclose(cluster, serial)


def test_distributed_runtime_validation_blocks_invalid_sizes(linear_spec: dict) -> None:
    """Invalid worker and chunk hints should be rejected early."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="workers must be >= 1"):
        runtime.predict_distributed(linear_spec, x, workers=0)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        runtime.predict_distributed(linear_spec, x, chunk_size=0)


def test_distributed_runtime_rejects_non_2d_inputs(linear_spec: dict) -> None:
    """Replay preview must fail fast on invalid row shapes."""
    invalid = np.array([1.0, 2.0, 3.0], dtype=float)
    with pytest.raises(ValueError, match="2D array-like"):
        runtime.predict_distributed(linear_spec, invalid)
    with pytest.raises(ValueError, match="2D array-like"):
        runtime.design_matrix_distributed(linear_spec, invalid)
