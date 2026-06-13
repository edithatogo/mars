"""Cluster execution helpers for H4 replay and future multi-node backends."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from ._model_spec import spec_to_json
from .runtime import design_matrix_cpu_cluster, load_model_spec, predict_cpu_cluster

CLUSTER_MODE_ENV_VAR = "MARS_EARTH_CLUSTER_MODE"
CLUSTER_WORKERS_ENV_VAR = "MARS_EARTH_CLUSTER_WORKERS"
CLUSTER_CHUNK_SIZE_ENV_VAR = "MARS_EARTH_CLUSTER_CHUNK_SIZE"
CLUSTER_PRESERVE_ORDER_ENV_VAR = "MARS_EARTH_CLUSTER_PRESERVE_ORDER"
CLUSTER_SCHEDULER_ENV_VAR = "MARS_EARTH_CLUSTER_SCHEDULER"
CLUSTER_WORKER_COMMAND_ENV_VAR = "MARS_EARTH_CLUSTER_WORKER_COMMAND"
CLUSTER_RETRIES_ENV_VAR = "MARS_EARTH_CLUSTER_RETRIES"
CPU_CLUSTER_MODE = "cpu-cluster"
MULTI_NODE_CLUSTER_MODE = "multi-node"


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Describe the requested H4 cluster execution mode."""

    mode: str = CPU_CLUSTER_MODE
    scheduler: str | None = None
    worker_command: str | None = None
    workers: int = 1
    chunk_size: int | None = None
    preserve_order: bool = True
    retries: int = 0

    def __post_init__(self) -> None:
        """Validate cluster execution settings eagerly."""
        if self.workers < 1:
            raise ValueError("ClusterConfig.workers must be at least 1.")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("ClusterConfig.chunk_size must be at least 1.")
        if self.retries < 0:
            raise ValueError("ClusterConfig.retries must be at least 0.")

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
        worker_command = os.environ.get(CLUSTER_WORKER_COMMAND_ENV_VAR)
        retries = _parse_non_negative_int_env(CLUSTER_RETRIES_ENV_VAR, default=0)

        preserve_order = preserve_order_raw.lower() not in {"0", "false", "no"}
        return cls(
            mode=mode,
            scheduler=scheduler.strip() if scheduler else None,
            worker_command=worker_command.strip() if worker_command else None,
            workers=workers or 1,
            chunk_size=chunk_size,
            preserve_order=preserve_order,
            retries=retries or 0,
        )


@runtime_checkable
class ClusterBackend(Protocol):
    """Protocol for H4 cluster execution backends."""

    name: str

    def is_available(self) -> bool:
        """Return whether the backend is usable in the current process."""

    def predict(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
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

    def predict(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
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

    def predict(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
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


@dataclass(frozen=True, slots=True)
class CommandMultiNodeBackend:
    """Command-backed H4 backend for explicit scheduler or node worker commands."""

    command: str
    name: str = MULTI_NODE_CLUSTER_MODE

    def is_available(self) -> bool:
        """Return whether a worker command is configured."""
        return bool(self.command.strip())

    def predict(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
        """Predict by dispatching row chunks to the configured worker command."""
        return np.asarray(
            self._dispatch("predict", spec_or_path, X, config),
            dtype=float,
        )

    def design_matrix(
        self, spec_or_path: dict[str, Any] | str, X: Any, config: ClusterConfig
    ):
        """Build a design matrix by dispatching chunks to the worker command."""
        return np.asarray(
            self._dispatch("design_matrix", spec_or_path, X, config),
            dtype=float,
        )

    def _dispatch(
        self,
        operation: str,
        spec_or_path: dict[str, Any] | str,
        X: Any,
        config: ClusterConfig,
    ) -> list[Any]:
        """Partition rows, run worker commands, and aggregate chunk results."""
        spec = load_model_spec(spec_or_path)
        rows = np.asarray(X, dtype=float)
        if rows.ndim != 2:
            raise ValueError("X must be a 2D array-like input for multi-node replay.")
        row_payload = cast("list[list[float]]", rows.tolist())
        chunk_size = _normalize_cluster_chunk_size(
            config.chunk_size, len(row_payload), config.workers
        )
        indices = _chunked_row_indices(len(row_payload), chunk_size)
        if not indices:
            return []
        command_parts = shlex.split(self.command)
        if not command_parts:
            raise NotImplementedError("Multi-node worker command is not configured.")

        with tempfile.TemporaryDirectory(prefix="mars-h4-worker-") as tmpdir:
            chunk_outputs: list[tuple[int, list[Any]]] = []
            for chunk_index, (start, end) in enumerate(indices):
                payload_path = Path(tmpdir) / f"chunk-{chunk_index}.json"
                payload_path.write_text(
                    json.dumps(
                        {
                            "operation": operation,
                            "spec_json": spec_to_json(spec),
                            "rows": row_payload[start:end],
                        }
                    )
                )
                chunk_outputs.append(
                    (
                        start,
                        _run_worker_command(
                            command_parts,
                            payload_path,
                            retries=config.retries,
                        ),
                    )
                )

        if config.preserve_order:
            chunk_outputs.sort(key=lambda item: item[0])
        aggregated: list[Any] = []
        for _, values in chunk_outputs:
            aggregated.extend(values)
        return aggregated


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


def _parse_non_negative_int_env(
    var_name: str, default: int | None = None
) -> int | None:
    """Parse a non-negative integer environment variable."""
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
    if value < 0:
        raise ValueError(f"{var_name} must be at least 0.")
    return value


def _normalize_cluster_chunk_size(
    chunk_size: int | None, row_count: int, workers: int
) -> int:
    """Return a positive chunk size for command-backed cluster dispatch."""
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("ClusterConfig.chunk_size must be at least 1.")
        return chunk_size
    return max(1, row_count // workers) if row_count >= workers else 1


def _chunked_row_indices(n_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return row index ranges for deterministic chunk dispatch."""
    if n_rows <= 0:
        return []
    return [
        (start, min(n_rows, start + chunk_size))
        for start in range(0, n_rows, chunk_size)
    ]


def _run_worker_command(
    command_parts: list[str], payload_path: Path, *, retries: int
) -> list[Any]:
    """Run a worker command with bounded retries and parse JSON output."""
    last_error: Exception | None = None
    for _attempt in range(retries + 1):
        try:
            completed = subprocess.run(  # noqa: S603 - explicit opt-in worker command.
                [*command_parts, str(payload_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            return _worker_result_list(payload)
        except (subprocess.CalledProcessError, json.JSONDecodeError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(
        f"Multi-node worker command failed: {last_error}"
    ) from last_error


def _worker_result_list(payload: dict[str, Any]) -> list[Any]:
    """Return the worker result list or raise a typed parsing error."""
    result = payload.get("result")
    if not isinstance(result, list):
        raise TypeError("Worker output must contain a list result.")
    return result


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
        if isinstance(preferred, ClusterConfig) and preferred.worker_command:
            return CommandMultiNodeBackend(command=preferred.worker_command)
        return DeferredMultiNodeBackend()
    return None


def cluster_backend_summary(
    config: ClusterConfig | None = None,
) -> dict[str, Any]:
    """Return a normalized summary of cluster backend selection."""
    resolved_config = config or cluster_config_from_environment()
    requested = resolved_config.normalized_mode()
    selected = select_cluster_backend(resolved_config)
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
        "worker_command_configured": bool(resolved.worker_command),
        "workers": resolved.workers,
        "chunk_size": resolved.chunk_size,
        "preserve_order": resolved.preserve_order,
        "retries": resolved.retries,
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
