"""Tests for the H4 cluster abstraction layer."""

from __future__ import annotations

import sys

import numpy as np

from pymars import (
    CLUSTER_CHUNK_SIZE_ENV_VAR,
    CLUSTER_MODE_ENV_VAR,
    CLUSTER_PRESERVE_ORDER_ENV_VAR,
    CLUSTER_SCHEDULER_ENV_VAR,
    CLUSTER_WORKER_COMMAND_ENV_VAR,
    CLUSTER_WORKERS_ENV_VAR,
    CPU_CLUSTER_MODE,
    MULTI_NODE_CLUSTER_MODE,
    ClusterConfig,
    DeferredMultiNodeBackend,
    cluster_backend_summary,
    cluster_config_from_environment,
    cluster_config_summary,
    design_matrix_cluster,
    detect_requested_cluster_mode,
    predict_cluster,
    select_cluster_backend,
)


def test_cluster_backend_defaults_to_cpu_cluster(monkeypatch) -> None:
    """Cluster selection should default to the CPU cluster backend."""
    monkeypatch.delenv("MARS_EARTH_CLUSTER_MODE", raising=False)

    backend = select_cluster_backend()
    summary = cluster_backend_summary()

    assert backend is not None
    assert backend.name == CPU_CLUSTER_MODE
    assert summary["selected"] == CPU_CLUSTER_MODE
    assert summary["fallback"] is False
    assert detect_requested_cluster_mode() is None
    assert summary["config"]["mode"] == CPU_CLUSTER_MODE


def test_cluster_backend_recognizes_multi_node_as_deferred(monkeypatch) -> None:
    """Multi-node cluster mode should stay explicitly deferred for now."""
    monkeypatch.setenv("MARS_EARTH_CLUSTER_MODE", MULTI_NODE_CLUSTER_MODE)

    backend = select_cluster_backend()
    summary = cluster_backend_summary()

    assert isinstance(backend, DeferredMultiNodeBackend)
    assert summary["selected"] == MULTI_NODE_CLUSTER_MODE
    assert summary["fallback"] is True
    assert summary["requested"] == MULTI_NODE_CLUSTER_MODE
    assert summary["config"]["mode"] == MULTI_NODE_CLUSTER_MODE


def test_deferred_multi_node_backend_raises_helpfully() -> None:
    """The deferred backend should fail clearly if invoked directly."""
    backend = DeferredMultiNodeBackend()
    config = ClusterConfig(mode=MULTI_NODE_CLUSTER_MODE, workers=2, chunk_size=1)

    try:
        backend.predict({}, [[0.0]], config)
    except NotImplementedError as exc:
        assert "deferred" in str(exc).lower()
    else:
        raise AssertionError("Expected NotImplementedError")


def test_cluster_config_normalizes_mode() -> None:
    """Cluster config should normalize the mode name consistently."""
    config = ClusterConfig(mode=" CPU-Cluster ", workers=2, chunk_size=4)
    assert config.normalized_mode() == CPU_CLUSTER_MODE


def test_cluster_config_rejects_invalid_workers() -> None:
    """Cluster config should reject invalid worker counts."""
    try:
        ClusterConfig(workers=0)
    except ValueError as exc:
        assert "workers" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_cluster_config_rejects_invalid_chunk_size() -> None:
    """Cluster config should reject invalid chunk sizes."""
    try:
        ClusterConfig(workers=1, chunk_size=0)
    except ValueError as exc:
        assert "chunk_size" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_cluster_config_from_environment(monkeypatch) -> None:
    """Cluster config should read the environment deterministically."""
    monkeypatch.setenv(CLUSTER_MODE_ENV_VAR, "multi-node")
    monkeypatch.setenv(CLUSTER_WORKERS_ENV_VAR, "4")
    monkeypatch.setenv(CLUSTER_CHUNK_SIZE_ENV_VAR, "8")
    monkeypatch.setenv(CLUSTER_PRESERVE_ORDER_ENV_VAR, "false")
    monkeypatch.setenv(CLUSTER_SCHEDULER_ENV_VAR, "slurm")
    monkeypatch.setenv(
        CLUSTER_WORKER_COMMAND_ENV_VAR, "python -m pymars.cluster_worker"
    )

    config = cluster_config_from_environment()

    assert config.mode == MULTI_NODE_CLUSTER_MODE
    assert config.workers == 4
    assert config.chunk_size == 8
    assert config.preserve_order is False
    assert config.scheduler == "slurm"
    assert config.worker_command == "python -m pymars.cluster_worker"


def test_cluster_config_from_environment_rejects_invalid_values(monkeypatch) -> None:
    """Cluster config should fail clearly on invalid numeric environment values."""
    monkeypatch.setenv(CLUSTER_WORKERS_ENV_VAR, "not-an-int")

    try:
        cluster_config_from_environment()
    except ValueError as exc:
        assert CLUSTER_WORKERS_ENV_VAR in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_cluster_config_summary_matches_config() -> None:
    """Cluster config summaries should round-trip the important fields."""
    config = ClusterConfig(
        mode=CPU_CLUSTER_MODE,
        scheduler="pbs",
        worker_command="python -m pymars.cluster_worker",
        workers=3,
        chunk_size=2,
        preserve_order=False,
    )

    summary = cluster_config_summary(config)

    assert summary == {
        "mode": CPU_CLUSTER_MODE,
        "scheduler": "pbs",
        "worker_command_configured": True,
        "workers": 3,
        "chunk_size": 2,
        "preserve_order": False,
        "retries": 0,
    }


def test_cluster_backend_selection_accepts_config_objects() -> None:
    """Cluster backend selection should accept a config object directly."""
    config = ClusterConfig(mode=CPU_CLUSTER_MODE, workers=2, chunk_size=1)

    backend = select_cluster_backend(config)

    assert backend is not None
    assert backend.name == CPU_CLUSTER_MODE


def test_cpu_cluster_backend_reuses_partitioned_replay() -> None:
    """The cluster abstraction should still support the CPU cluster replay path."""
    from pymars import Earth, runtime

    X = np.array(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]],
        dtype=float,
    )
    y = np.array([0.0, 3.0, 6.0, 9.0, 12.0], dtype=float)
    model = Earth(max_degree=1, max_terms=10, penalty=3.0)
    model.fit(X, y)
    spec = model.get_model_spec()

    probe = np.array([[0.5, 1.0], [2.5, 5.0]], dtype=float)
    expected = runtime.predict_cpu_cluster(spec, probe, workers=2, chunk_size=1)
    actual = runtime.predict_cpu_cluster(spec, probe, workers=2, chunk_size=1)
    assert np.allclose(actual, expected)


def test_cluster_api_dispatches_to_cpu_backend() -> None:
    """The stable cluster API should dispatch to the CPU cluster backend."""
    from pymars import Earth

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)
    model = Earth(max_degree=1, max_terms=5, penalty=3.0)
    model.fit(X, y)
    spec = model.get_model_spec()
    config = ClusterConfig(mode=CPU_CLUSTER_MODE, workers=2, chunk_size=1)

    cluster_predictions = predict_cluster(
        spec, np.array([[0.5], [1.5]], dtype=float), config
    )
    cluster_matrix = design_matrix_cluster(
        spec, np.array([[0.5], [1.5]], dtype=float), config
    )

    assert cluster_predictions.shape == (2,)
    assert cluster_matrix.shape[0] == 2


def test_cluster_api_rejects_deferred_multi_node_mode() -> None:
    """The stable cluster API should fail clearly for the deferred multi-node mode."""
    config = ClusterConfig(mode=MULTI_NODE_CLUSTER_MODE, workers=2, chunk_size=1)

    try:
        predict_cluster({}, [[0.0]], config)
    except NotImplementedError as exc:
        assert "deferred" in str(exc).lower()
    else:
        raise AssertionError("Expected NotImplementedError")


def test_command_backed_multi_node_predict_matches_cpu_replay() -> None:
    """Configured multi-node worker commands should replay chunks deterministically."""
    from pymars import Earth, runtime

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float)
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)
    model = Earth(max_degree=1, max_terms=5, penalty=3.0)
    model.fit(X, y)
    spec = model.get_model_spec()
    probe = np.array([[0.5], [1.5], [2.5]], dtype=float)
    config = ClusterConfig(
        mode=MULTI_NODE_CLUSTER_MODE,
        scheduler="command",
        worker_command=f"{sys.executable} -m pymars.cluster_worker",
        workers=2,
        chunk_size=1,
    )

    actual = predict_cluster(spec, probe, config)
    expected = runtime.predict(spec, probe)
    summary = cluster_backend_summary(config)

    assert np.allclose(actual, expected)
    assert summary["selected"] == MULTI_NODE_CLUSTER_MODE
    assert summary["fallback"] is False


def test_command_backed_multi_node_design_matrix_matches_cpu_replay() -> None:
    """The command-backed H4 adapter should aggregate design-matrix chunks."""
    from pymars import Earth, runtime

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float)
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)
    model = Earth(max_degree=1, max_terms=5, penalty=3.0)
    model.fit(X, y)
    spec = model.get_model_spec()
    probe = np.array([[0.5], [1.5], [2.5]], dtype=float)
    config = ClusterConfig(
        mode=MULTI_NODE_CLUSTER_MODE,
        scheduler="command",
        worker_command=f"{sys.executable} -m pymars.cluster_worker",
        workers=2,
        chunk_size=1,
    )

    actual = design_matrix_cluster(spec, probe, config)
    expected = runtime.design_matrix(spec, probe)

    assert np.allclose(actual, expected)
