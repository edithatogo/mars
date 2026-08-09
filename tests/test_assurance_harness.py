"""Contract tests for the canonical assurance harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "assurance.py"


def run_harness(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the harness without synchronizing the project environment."""
    return subprocess.run(  # noqa: S603 - executable and harness path are fixed.
        [sys.executable, str(HARNESS), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_profiles_are_discoverable_as_json() -> None:
    """Every governed profile should be machine discoverable."""
    result = run_harness("list", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "fast",
        "full",
        "preview",
        "release-rehearsal",
        "security",
    }
    assert all(payload[name]["commands"] for name in payload)
    assert "benchmark" in payload["full"]["commands"]
    assert {"bandit", "pip-audit"} <= set(payload["security"]["commands"])


def test_dry_run_writes_a_deterministic_receipt(tmp_path: Path) -> None:
    """Dry runs should record stable commands without executing tools."""
    receipt = tmp_path / "receipt.json"
    result = run_harness(
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
    assert payload["commands"]
    assert all(command["outcome"] == "not-run" for command in payload["commands"])
    assert "started_at" not in payload


def test_unknown_profile_is_a_usage_error() -> None:
    """Unknown profiles should fail before any command executes."""
    result = run_harness("run", "unknown", "--dry-run")

    assert result.returncode == 2
    assert "unknown" in result.stderr.lower()
