"""Canonical local assurance runner with machine-readable receipts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "build" / "assurance" / "receipt.json"


@dataclass(frozen=True)
class Command:
    """One non-shell command in an assurance profile."""

    name: str
    argv: tuple[str, ...]


FAST = (
    Command("policy", ("uv", "run", "python", "tools/validate_ci_policy.py")),
    Command("lint", ("uv", "run", "ruff", "check", "pymars", "tests", "tools")),
    Command("types", ("uv", "run", "ty", "check", "pymars")),
    Command(
        "tests",
        (
            "uv",
            "run",
            "pytest",
            "tests",
            "-q",
            "--junitxml=build/assurance/junit.xml",
            "--cov=pymars",
            "--cov-report=xml:build/assurance/coverage.xml",
        ),
    ),
)

PROFILES: dict[str, tuple[Command, ...]] = {
    "fast": FAST,
    "full": (
        *FAST,
        Command(
            "rust", ("cargo", "test", "--manifest-path", "rust-runtime/Cargo.toml")
        ),
        Command("go", ("go", "test", "./...")),
        Command("typescript", ("npm", "test", "--prefix", "bindings/typescript")),
        Command("docs", ("pnpm", "--dir", "docs/astro-site", "build")),
    ),
    "security": (
        Command("bandit", ("uv", "run", "bandit", "-r", "pymars", "-q")),
        Command("pip-audit", ("uv", "run", "pip-audit")),
        Command(
            "cargo-deny",
            ("cargo", "deny", "--manifest-path", "rust-runtime/Cargo.toml", "check"),
        ),
    ),
    "release-rehearsal": (
        Command("python-dist", ("uv", "build")),
        Command(
            "rust-package",
            (
                "cargo",
                "package",
                "--manifest-path",
                "rust-runtime/Cargo.toml",
                "--allow-dirty",
            ),
        ),
        Command(
            "typescript-pack",
            ("npm", "pack", "--prefix", "bindings/typescript", "--dry-run"),
        ),
    ),
    "preview": (
        Command(
            "python-prerelease",
            ("uv", "sync", "--extra", "dev", "--prerelease=allow", "--dry-run"),
        ),
        Command(
            "rust-nightly",
            ("cargo", "+nightly", "test", "--manifest-path", "rust-runtime/Cargo.toml"),
        ),
    ),
}


def profile_catalog() -> dict[str, dict[str, object]]:
    """Return the stable, serializable profile catalog."""
    return {
        name: {
            "commands": [command.name for command in commands],
            "required": name in {"fast", "full", "security", "release-rehearsal"},
        }
        for name, commands in sorted(PROFILES.items())
    }


def write_receipt(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace a JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_profile(
    profile: str,
    *,
    seed: int,
    receipt: Path,
    dry_run: bool,
    keep_going: bool,
) -> int:
    """Execute a profile and return its process-compatible outcome."""
    commands = PROFILES[profile]
    results: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile": profile,
        "seed": seed,
        "commands": results,
    }
    if dry_run:
        results.extend(
            {"name": command.name, "argv": list(command.argv), "outcome": "not-run"}
            for command in commands
        )
        payload["outcome"] = "dry-run"
        write_receipt(receipt, payload)
        return 0

    started = datetime.now(UTC)
    payload["started_at"] = started.isoformat()
    environment = os.environ.copy()
    environment.update(
        {"CI": "true", "PYTHONHASHSEED": str(seed), "HYPOTHESIS_SEED": str(seed)}
    )
    failed = False
    for command in commands:
        command_started = time.monotonic()
        completed = subprocess.run(  # noqa: S603 - argv comes from the static catalog.
            command.argv,
            cwd=ROOT,
            env=environment,
            check=False,
        )
        result = {
            "name": command.name,
            "argv": list(command.argv),
            "exit_code": completed.returncode,
            "duration_seconds": round(time.monotonic() - command_started, 3),
            "outcome": "passed" if completed.returncode == 0 else "failed",
        }
        results.append(result)
        if completed.returncode != 0:
            failed = True
            if not keep_going:
                break
    payload["finished_at"] = datetime.now(UTC).isoformat()
    payload["outcome"] = "failed" if failed else "passed"
    write_receipt(receipt, payload)
    return 1 if failed else 0


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="action", required=True)
    listing = subparsers.add_parser("list", help="list assurance profiles")
    listing.add_argument("--json", action="store_true", dest="as_json")
    running = subparsers.add_parser("run", help="run one assurance profile")
    running.add_argument("profile", choices=sorted(PROFILES))
    running.add_argument("--seed", type=int, default=1729)
    running.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    running.add_argument("--dry-run", action="store_true")
    running.add_argument("--keep-going", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    """Run the assurance CLI."""
    arguments = parser().parse_args(argv)
    if arguments.action == "list":
        catalog = profile_catalog()
        if arguments.as_json:
            print(json.dumps(catalog, indent=2, sort_keys=True))
        else:
            for name, details in catalog.items():
                print(f"{name}: {', '.join(details['commands'])}")
        return 0
    return run_profile(
        arguments.profile,
        seed=arguments.seed,
        receipt=arguments.receipt,
        dry_run=arguments.dry_run,
        keep_going=arguments.keep_going,
    )


if __name__ == "__main__":
    sys.exit(main())
