"""Cluster execution helpers for H4 replay and future multi-node backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .runtime import design_matrix_cpu_cluster, predict_cpu_cluster

CLUSTER_MODE_ENV_VAR = "MARS_EARTH_CLUSTER_MODE"
CLUSTER_WORKERS_ENV_VAR = "MARS_EARTH_CLUSTER_WORKERS"
CLUSTER_CHUNK_SIZE_ENV_VAR = "MARS_EARTH_CLUSTER_CHUNK_SIZE"
CLUSTER_PRESERVE_ORDER_ENV_VAR = "MARS_EARTH_CLUSTER_PRESERVE_ORDER"
CLUSTER_SCHEDULER_ENV_VAR = "MARS_EARTH_CLUSTER_SCHEDULER"
CPU_CLUSTER_MODE = "cpu-cluster"
MULTI_NODE_CLUSTER_MODE = "multi-node"


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Describe the requested H4 cluster execution mode."""

    mode: str = CPU_CLUSTER_MODE
    scheduler: str | None = None
    workers: int = 1
    chunk_size: int | None = None
    preserve_order: bool = True

    def __post_init__(self) -> None:
        """Validate cluster execution settings eagerly."""
        if self.workers < 1:
            raise ValueError("ClusterConfig.workers must be at least 1.")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("ClusterConfig.chunk_size must be at least 1.")

    def normalized_mode(self) -> str:
        """Return a canonical, lower-case cluster mode name."""
        return self.mode.strip().lower() or CPU_CLUSTER_MODE

    @classmethod
    def from_environment(cls) -> ClusterConfig:
        """Build a cluster config from the process environment."""
        mode = os.environ.get(CLUSTER_MODE_ENV_VAR, CPU_CLUSTER_MODE)
        workers = _parse_positive_int_env(CLUSTER_WORKERS_ENV_VAR, default=1)
        chunk_size = _parse_positive_int_env(CLUSTER_CHUNK_SIZE_ENV_VAR)
        preserve_order_raw = os.environ.get(CLUSTER_PRESERVE_ORDER_ENV_VAR, "1").strip()
        scheduler = os.environ.get(CLUSTER_SCHEDULER_ENV_VAR)

        preserve_order = preserve_order_raw.lower() not in {"0", "false", "no"}
        return cls(
            mode=mode,
            scheduler=scheduler.strip() if scheduler else None,
            workers=workers or 1,
            chunk_size=chunk_size,
            preserve_order=preserve_order,
        )


@runtime_checkable
class ClusterBackend(Protocol):
    """Protocol for H4 cluster execution backends."""

    name: str

    def is_available(self) -> bool:
        """Return whether the backend is usable in the current process."""

    def predict(self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig):
        """Predict through the cluster backend."""

    def design_matrix(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
        """Build a design matrix through the cluster backend."""


@dataclass(frozen=True, slots=True)
class CpuClusterBackend:
    """Process-based CPU cluster replay backend."""

    name: str = CPU_CLUSTER_MODE

    def is_available(self) -> bool:
        """CPU cluster replay is always available."""
        return True

    def predict(self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig):
        """Predict through the CPU cluster replay path."""
        return predict_cpu_cluster(
            spec_or_path,
            X,
            workers=config.workers,
            chunk_size=config.chunk_size,
            preserve_order=config.preserve_order,
        )

    def design_matrix(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
        """Build a design matrix through the CPU cluster replay path."""
        return design_matrix_cpu_cluster(
            spec_or_path,
            X,
            workers=config.workers,
            chunk_size=config.chunk_size,
            preserve_order=config.preserve_order,
        )


@dataclass(frozen=True, slots=True)
class DeferredMultiNodeBackend:
    """Placeholder for the not-yet-implemented multi-node H4 backend."""

    name: str = MULTI_NODE_CLUSTER_MODE

    def is_available(self) -> bool:
        """The multi-node backend is intentionally deferred."""
        return False

    def predict(self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig):
        """Fail clearly when callers request the deferred multi-node path."""
        del spec_or_path, X, config
        raise NotImplementedError(
            "Multi-node H4 execution is deferred; use the CPU cluster replay path."
        )

    def design_matrix(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
        """Fail clearly when callers request the deferred multi-node path."""
        del spec_or_path, X, config
        raise NotImplementedError(
            "Multi-node H4 execution is deferred; use the CPU cluster replay path."
        )


def detect_requested_cluster_mode() -> str | None:
    """Return the requested cluster mode from the environment."""
    value = os.environ.get(CLUSTER_MODE_ENV_VAR)
    if value is None:
        return None
    mode = value.strip().lower()
    return mode or None


def _parse_positive_int_env(var_name: str, default: int | None = None) -> int | None:
    """Parse a positive integer environment variable."""
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{var_name} must be at least 1.")
    return value


def select_cluster_backend(
    preferred: str | ClusterConfig | None = None,
) -> ClusterBackend | None:
    """Return the available cluster backend for the requested mode."""
    if isinstance(preferred, ClusterConfig):
        requested = preferred.normalized_mode()
    else:
        requested = (preferred or detect_requested_cluster_mode() or "").strip().lower()
    if requested in ("", CPU_CLUSTER_MODE):
        return CpuClusterBackend()
    if requested == MULTI_NODE_CLUSTER_MODE:
        return DeferredMultiNodeBackend()
    return None


def cluster_backend_summary(
    config: ClusterConfig | None = None,
) -> dict[str, Any]:
    """Return a normalized summary of cluster backend selection."""
    resolved_config = config or cluster_config_from_environment()
    requested = resolved_config.normalized_mode()
    selected = select_cluster_backend(requested)
    if selected is None:
        return {
            "requested": requested or MULTI_NODE_CLUSTER_MODE,
            "selected": None,
            "fallback": True,
            "available": [CPU_CLUSTER_MODE],
            "config": cluster_config_summary(resolved_config),
        }
    return {
        "requested": requested or selected.name,
        "selected": selected.name,
        "fallback": not selected.is_available(),
        "available": [CPU_CLUSTER_MODE],
        "config": cluster_config_summary(resolved_config),
    }


def cluster_config_from_environment() -> ClusterConfig:
    """Return the current cluster configuration from environment variables."""
    return ClusterConfig.from_environment()


def cluster_config_summary(config: ClusterConfig | None = None) -> dict[str, Any]:
    """Return a normalized summary of the active cluster configuration."""
    resolved = config or cluster_config_from_environment()
    return {
        "mode": resolved.normalized_mode(),
        "scheduler": resolved.scheduler,
        "workers": resolved.workers,
        "chunk_size": resolved.chunk_size,
        "preserve_order": resolved.preserve_order,
    }


def predict_cluster(
    spec_or_path: dict[str, Any] | str,
    X: Any,
    config: ClusterConfig | None = None,
) -> Any:
    """Predict through the selected H4 cluster backend."""
    resolved_config = config or ClusterConfig()
    backend = select_cluster_backend(resolved_config)
    if backend is None:
        raise NotImplementedError(
            f"Cluster mode {resolved_config.mode!r} is not available."
        )
    return backend.predict(spec_or_path, X, resolved_config)


def design_matrix_cluster(
    spec_or_path: dict[str, Any] | str,
    X: Any,
    config: ClusterConfig | None = None,
) -> Any:
    """Build a design matrix through the selected H4 cluster backend."""
    resolved_config = config or ClusterConfig()
    backend = select_cluster_backend(resolved_config)
    if backend is None:
        raise NotImplementedError(
            f"Cluster mode {resolved_config.mode!r} is not available."
        )
    return backend.design_matrix(spec_or_path, X, resolved_config)
