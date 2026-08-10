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
        json.dumps({"controls": [{"id": "workflow-policy"}]}),
    )
    _write(
        tmp_path / "docs" / "astro-site" / "package.json",
        json.dumps({"dependencies": {"astro": "^7.0.0"}}),
    )


def test_minimal_hardened_repository_passes(tmp_path: Path) -> None:
    """Accept a repository satisfying the minimum deterministic policy."""
    _minimal_repository(tmp_path)
    assert validate_repository(tmp_path) == []


def test_mutable_actions_missing_timeout_and_merge_group_are_rejected(
    tmp_path: Path,
) -> None:
    """Reject mutable actions and missing workflow execution controls."""
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
    """Require blocking OIDC Codecov uploads backed by XML coverage."""
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


def test_local_file_dependency_and_incomplete_monorepo_managers_are_rejected(
    tmp_path: Path,
) -> None:
    """Reject local dependencies and incomplete Renovate manager coverage."""
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


def test_go_workflow_versions_must_match_module_floor(tmp_path: Path) -> None:
    """Reject setup-go and consumer fixtures below the governed module floor."""
    _minimal_repository(tmp_path)
    _write(tmp_path / "go.mod", "module example.com/root\n\ngo 1.26.5\n")
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8")
        + """
      - uses: actions/setup-go@0123456789012345678901234567890123456789 # v6
        with:
          go-version: "1.22"
      - run: |
          cat > go.mod <<EOF
          go 1.22
          EOF
""",
        encoding="utf-8",
    )
    codes = {violation.code for violation in validate_repository(tmp_path)}
    assert "go-version-drift" in codes
