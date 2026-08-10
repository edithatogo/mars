"""Executable contract for the canonical assurance harness."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest import MonkeyPatch

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "assurance.py"

pytestmark = pytest.mark.xfail(
    not HARNESS.is_file(),
    reason="red-phase contract: tools/assurance.py is implemented by the next task",
    strict=False,
)


def _module():
    """Import the harness only after the red-phase marker has been evaluated."""
    return importlib.import_module("tools.assurance")


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the harness with the current interpreter and no interactive input."""
    return subprocess.run(  # noqa: S603 - interpreter and repository path are fixed.
        [sys.executable, str(HARNESS), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=20,
    )


def test_profiles_are_discoverable_and_classified() -> None:
    """Discovery identifies all profiles and required versus preview suites."""
    result = _run("list", "--json")

    assert result.returncode == 0, result.stderr
    catalog = json.loads(result.stdout)
    assert set(catalog) == {
        "fast",
        "full",
        "preview",
        "release-rehearsal",
        "security",
    }
    assert catalog["preview"]["required"] is False
    assert all(catalog[name]["required"] for name in catalog if name != "preview")
    assert all(catalog[name]["commands"] for name in catalog)


def test_unknown_profile_is_a_usage_error() -> None:
    """Unknown profiles fail before executing commands."""
    result = _run("run", "unknown", "--dry-run")

    assert result.returncode == 2
    assert "unknown" in result.stderr.lower()


def test_dry_run_receipt_is_deterministic_and_replayable(tmp_path: Path) -> None:
    """Dry runs record stable commands, inputs, hosted parity, and replay data."""
    receipt = tmp_path / "receipt.json"
    result = _run(
        "run",
        "fast",
        "--dry-run",
        "--seed",
        "1729",
        "--receipt",
        str(receipt),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["profile"] == "fast"
    assert payload["seed"] == 1729
    assert payload["outcome"] == "dry-run"
    assert payload["source_commit"]
    assert payload["dirty_state_digest"]
    assert payload["platform"]
    assert payload["toolchains"]
    assert payload["hosted_parity"]["workflows"]
    assert payload["replay"]["argv"]
    assert payload["replay"]["environment"]["PYTHONHASHSEED"] == "1729"
    assert payload["replay"]["environment"]["HYPOTHESIS_SEED"] == "1729"
    assert all(command["outcome"] == "not-run" for command in payload["commands"])
    assert "started_at" not in payload


def test_execution_is_noninteractive_and_propagates_seed(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Execution is shell-free, noninteractive, and receives deterministic seeds."""
    harness = _module()
    observed: list[dict[str, object]] = []

    def fake_run(argv, **kwargs):
        observed.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    receipt = tmp_path / "receipt.json"
    result = harness.run_profile(
        "fast", seed=2468, receipt=receipt, dry_run=False, keep_going=False
    )

    assert result == 0
    assert observed
    assert all(call["shell"] is False for call in observed)
    assert all(call["stdin"] is subprocess.DEVNULL for call in observed)
    assert all(call["env"]["CI"] == "true" for call in observed)
    assert all(call["env"]["PYTHONHASHSEED"] == "2468" for call in observed)
    assert all(call["env"]["HYPOTHESIS_SEED"] == "2468" for call in observed)


def test_fail_fast_preserves_partial_failure_receipt(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """The first failure stops admission while preserving atomic partial evidence."""
    harness = _module()
    calls = 0

    def fake_run(argv, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    receipt = tmp_path / "receipt.json"
    result = harness.run_profile(
        "fast", seed=1729, receipt=receipt, dry_run=False, keep_going=False
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))

    assert result == 1
    assert calls == 1
    assert payload["outcome"] == "failed"
    assert payload["commands"][0]["exit_code"] == 7
    assert payload["commands"][0]["outcome"] == "failed"
    assert not receipt.with_suffix(".json.tmp").exists()


def test_keep_going_collects_independent_failures(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Diagnostic continuation retains a failing exit and every command result."""
    harness = _module()
    outcomes = iter((1, 0, 2, 0))

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, next(outcomes))

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    receipt = tmp_path / "receipt.json"
    result = harness.run_profile(
        "fast", seed=1729, receipt=receipt, dry_run=False, keep_going=True
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))

    assert result == 1
    assert len(payload["commands"]) == 4
    assert [item["exit_code"] for item in payload["commands"]] == [1, 0, 2, 0]
    assert payload["outcome"] == "failed"


def test_receipts_do_not_capture_unrestricted_environment(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Receipts contain an allowlisted replay environment and no ambient secrets."""
    harness = _module()
    monkeypatch.setenv("MARS_TEST_SECRET", "must-not-be-recorded")
    receipt = tmp_path / "receipt.json"
    result = harness.run_profile(
        "fast", seed=1729, receipt=receipt, dry_run=True, keep_going=False
    )

    assert result == 0
    serialized = receipt.read_text(encoding="utf-8")
    assert "must-not-be-recorded" not in serialized
    assert "MARS_TEST_SECRET" not in serialized
    assert os.environ["MARS_TEST_SECRET"] == "must-not-be-recorded"  # noqa: S105
