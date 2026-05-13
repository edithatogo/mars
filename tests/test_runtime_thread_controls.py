"""Runtime thread override tests."""

from __future__ import annotations

import os

import pytest

from pymars import runtime


def test_set_runtime_threads_removes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ``None`` should remove the runtime thread override variable."""
    env_var = runtime._RUNTIME_THREAD_ENV_VAR
    monkeypatch.setenv(env_var, "4")

    runtime.set_runtime_threads(None)

    assert env_var not in os.environ


def test_set_runtime_threads_validate_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero or negative thread counts are rejected."""
    monkeypatch.delenv(runtime._RUNTIME_THREAD_ENV_VAR, raising=False)

    with pytest.raises(ValueError, match="thread_count must be >= 1"):
        runtime.set_runtime_threads(0)


def test_set_runtime_threads_updates_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid thread counts should be written to the environment hint."""
    env_var = runtime._RUNTIME_THREAD_ENV_VAR
    monkeypatch.delenv(env_var, raising=False)

    runtime.set_runtime_threads(4)

    assert os.environ.get(env_var) == "4"


def test_runtime_threads_context_restores_previous_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The previous thread override should be restored after the context exits."""
    env_var = runtime._RUNTIME_THREAD_ENV_VAR
    monkeypatch.setenv(env_var, "4")

    with runtime.runtime_threads(2):
        assert os.environ.get(env_var) == "2"

    assert os.environ.get(env_var) == "4"


def test_runtime_threads_context_removes_missing_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing variable before entering the context should be removed after exit."""
    env_var = runtime._RUNTIME_THREAD_ENV_VAR
    monkeypatch.delenv(env_var, raising=False)

    with runtime.runtime_threads(1):
        assert os.environ.get(env_var) == "1"

    assert env_var not in os.environ


def test_runtime_threads_context_none_clears_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing ``None`` should clear the runtime override while active."""
    env_var = runtime._RUNTIME_THREAD_ENV_VAR
    monkeypatch.setenv(env_var, "3")

    with runtime.runtime_threads(None):
        assert env_var not in os.environ

    assert os.environ.get(env_var) == "3"
