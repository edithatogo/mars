"""Tests for repository CI/CD policy validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tools.validate_ci_policy import PolicyViolation, validate_repository

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_repository(tmp_path: Path) -> None:
    _write(
        tmp_path / ".github" / "workflows" / "ci.yml",
        """name: CI
on:
  pull_request:
  merge_group:
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    timeout-minutes: 20
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: actions/checkout@0123456789012345678901234567890123456789 # v7
      - run: pytest --cov=pymars --cov-report=xml:coverage.xml
      - uses: codecov/codecov-action@abcdefabcdefabcdefabcdefabcdefabcdefabcd # v5
        with:
          files: ./coverage.xml
          flags: python
          fail_ci_if_error: true
          use_oidc: true
""",
    )
    _write(
        tmp_path / "codecov.yml",
        """coverage:
  status:
    project:
      python:
        target: 90%
    patch:
      python:
        target: 90%
""",
    )
    _write(
        tmp_path / ".github" / "renovate.json",
        json.dumps(
            {
                "$schema": "https://docs.renovatebot.com/renovate-schema.json",
                "extends": ["config:recommended"],
                "enabledManagers": [
                    "cargo",
                    "github-actions",
                    "git-submodules",
                    "gomod",
                    "npm",
                    "pep621",
                ],
                "packageRules": [],
            }
        ),
    )
    _write(
        tmp_path / ".github" / "assurance-controls.json",
        json.dumps(
            {
                "schemaVersion": 1,
                "lifecycleStates": [
                    "configured",
                    "enabled",
                    "executed",
                    "passing",
                    "deferred",
                    "blocked",
                ],
                "controls": [
                    {
                        "id": "workflow-policy",
                        "title": "Workflow policy",
                        "tier": "pull-request",
                        "trigger": ["pull_request", "merge_group"],
                        "commands": ["actionlint"],
                        "workflows": ["CI"],
                        "evidence": ["job summary"],
                        "failurePolicy": "block-merge",
                        "remediation": "Fix and rerun.",
                        "state": "configured",
                    }
                ],
            }
        ),
    )
    _write(
        tmp_path / "docs" / "astro-site" / "package.json",
        json.dumps({"dependencies": {"astro": "^7.0.0"}}),
    )


def test_minimal_hardened_repository_passes(tmp_path: Path) -> None:
    """A minimal repository satisfying every policy has no findings."""
    _minimal_repository(tmp_path)
    assert validate_repository(tmp_path) == []


def test_mutable_actions_missing_timeout_and_merge_group_are_rejected(
    tmp_path: Path,
) -> None:
    """Mutable actions and missing execution guards are rejected."""
    _minimal_repository(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8")
        .replace("  merge_group:\n", "")
        .replace("    timeout-minutes: 20\n", "")
        .replace(
            "actions/checkout@0123456789012345678901234567890123456789",
            "actions/checkout@v7",
        ),
        encoding="utf-8",
    )
    codes = {violation.code for violation in validate_repository(tmp_path)}
    assert {
        "action-not-sha-pinned",
        "job-timeout-missing",
        "merge-group-missing",
    } <= codes


def test_codecov_must_be_blocking_oidc_and_generate_xml(tmp_path: Path) -> None:
    """Codecov must consume XML, fail closed, and authenticate with OIDC."""
    _minimal_repository(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8")
        .replace(" --cov-report=xml:coverage.xml", "")
        .replace(
            "          fail_ci_if_error: true\n", "          fail_ci_if_error: false\n"
        )
        .replace(
            "          use_oidc: true\n",
            "          token: ${{ secrets.CODECOV_TOKEN }}\n",
        ),
        encoding="utf-8",
    )
    codes = {violation.code for violation in validate_repository(tmp_path)}
    assert {
        "codecov-not-blocking",
        "codecov-oidc-missing",
        "coverage-xml-not-generated",
    } <= codes


def test_permissions_concurrency_and_artifact_retention_are_required(
    tmp_path: Path,
) -> None:
    """Workflows declare permissions, concurrency, and artifact retention."""
    _minimal_repository(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8")
        .replace("permissions:\n  contents: read\n", "")
        .replace(
            "concurrency:\n  group: ci-${{ github.ref }}\n  cancel-in-progress: true\n",
            "",
        )
        .replace(
            "      - run: pytest --cov=pymars --cov-report=xml:coverage.xml\n",
            "      - run: pytest --cov=pymars --cov-report=xml:coverage.xml\n"
            "      - uses: actions/upload-artifact@0123456789012345678901234567890123456789 # v4\n"
            "        with:\n"
            "          name: reports\n"
            "          path: coverage.xml\n",
        ),
        encoding="utf-8",
    )
    codes = {violation.code for violation in validate_repository(tmp_path)}
    assert {
        "workflow-permissions-missing",
        "workflow-concurrency-missing",
        "artifact-retention-missing",
    } <= codes


def test_concurrency_and_retention_are_scoped_to_their_yaml_blocks(
    tmp_path: Path,
) -> None:
    """Unrelated groups and retained uploads cannot mask incomplete controls."""
    _minimal_repository(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8")
        .replace(
            "concurrency:\n  group: ci-${{ github.ref }}\n  cancel-in-progress: true\n",
            "concurrency:\n  cancel-in-progress: true\n",
        )
        .replace(
            "    steps:\n",
            "    strategy:\n"
            "      group: unrelated-job-value\n"
            "    steps:\n"
            "      - uses: actions/upload-artifact@0123456789012345678901234567890123456789 # v4\n"
            "        with:\n"
            "          name: retained\n"
            "          path: coverage.xml\n"
            "          retention-days: 7\n"
            "      - uses: actions/upload-artifact@abcdefabcdefabcdefabcdefabcdefabcdefabcd # v4\n"
            "        with:\n"
            "          name: unbounded\n"
            "          path: coverage.xml\n",
        ),
        encoding="utf-8",
    )
    codes = {violation.code for violation in validate_repository(tmp_path)}
    assert {"workflow-concurrency-missing", "artifact-retention-missing"} <= codes


def test_control_declarations_require_lifecycle_and_evidence_contract(
    tmp_path: Path,
) -> None:
    """Controls declare their schema, lifecycle, and evidence contract."""
    _minimal_repository(tmp_path)
    controls = tmp_path / ".github" / "assurance-controls.json"
    controls.write_text(
        json.dumps(
            {
                "schemaVersion": 0,
                "lifecycleStates": ["configured"],
                "controls": [{"id": "workflow-policy"}],
            }
        ),
        encoding="utf-8",
    )
    codes = {violation.code for violation in validate_repository(tmp_path)}
    assert {
        "assurance-controls-schema-version",
        "assurance-lifecycle-incomplete",
        "assurance-control-incomplete",
    } <= codes


def test_local_file_dependency_and_incomplete_monorepo_managers_are_rejected(
    tmp_path: Path,
) -> None:
    """Dependencies stay portable and Renovate covers monorepo managers."""
    _minimal_repository(tmp_path)
    package = tmp_path / "docs" / "astro-site" / "package.json"
    package.write_text(
        json.dumps({"dependencies": {"local-package": "file:/Users/example/pkg"}}),
        encoding="utf-8",
    )
    renovate = tmp_path / ".github" / "renovate.json"
    config = json.loads(renovate.read_text(encoding="utf-8"))
    config["enabledManagers"] = ["npm"]
    renovate.write_text(json.dumps(config), encoding="utf-8")
    violations = validate_repository(tmp_path)
    assert (
        PolicyViolation(
            code="nonportable-local-dependency",
            path="docs/astro-site/package.json",
            message="Dependency local-package uses non-portable specifier file:/Users/example/pkg",
        )
        in violations
    )
    assert "renovate-manager-coverage" in {item.code for item in violations}


def test_missing_and_malformed_policy_files_are_reported(tmp_path: Path) -> None:
    """Missing, malformed, and empty policy inputs fail deterministically."""
    _minimal_repository(tmp_path)
    controls = tmp_path / ".github" / "assurance-controls.json"
    controls.unlink()
    renovate = tmp_path / ".github" / "renovate.json"
    renovate.unlink()
    for workflow in (tmp_path / ".github" / "workflows").iterdir():
        workflow.unlink()
    codes = {violation.code for violation in validate_repository(tmp_path)}
    assert {
        "assurance-controls-missing",
        "codecov-upload-missing",
        "renovate-config-missing",
    } <= codes

    controls.write_text("{", encoding="utf-8")
    assert "assurance-controls-invalid" in {
        item.code for item in validate_repository(tmp_path)
    }
    controls.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "lifecycleStates": [
                    "configured",
                    "enabled",
                    "executed",
                    "passing",
                    "deferred",
                    "blocked",
                ],
                "controls": [],
            }
        ),
        encoding="utf-8",
    )
    assert "assurance-controls-empty" in {
        item.code for item in validate_repository(tmp_path)
    }
