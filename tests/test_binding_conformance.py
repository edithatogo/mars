"""Tests for the shared binding conformance harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bindings.conformance.runner import (
    DEFAULT_MANIFEST,
    load_manifest,
    validate_binding_output,
    validate_expected_fixtures,
)


def test_manifest_references_valid_runtime_fixtures() -> None:
    cases, _, _ = load_manifest(DEFAULT_MANIFEST)

    assert {case.name for case in cases} == {
        "v1",
        "categorical",
        "combined",
        "interaction",
        "missingness",
    }
    validate_expected_fixtures(cases)


def test_manifest_declares_runtime_and_training_binding_modes() -> None:
    payload = json.loads(DEFAULT_MANIFEST.read_text())

    assert payload["binding_modes"] == {
        "runtime_mvp": {"status": "current"},
        "runtime_rust_backed": {"status": "planned"},
        "training_rust_backed": {"status": "planned"},
    }


@pytest.mark.parametrize("removed_mode", ["runtime_rust_backed", "training_rust_backed"])
def test_manifest_requires_future_binding_modes(
    tmp_path: Path, removed_mode: str
) -> None:
    payload = json.loads(DEFAULT_MANIFEST.read_text())
    del payload["binding_modes"][removed_mode]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(AssertionError, match=removed_mode):
        load_manifest(manifest_path)


def test_binding_output_validation_accepts_expected_fixture_values(
    tmp_path: Path,
) -> None:
    cases, atol, rtol = load_manifest(DEFAULT_MANIFEST)
    output = {
        "binding": "test",
        "fixtures": [],
    }
    for case in cases:
        expected = json.loads(case.runtime_fixture.read_text())
        output["fixtures"].append(
            {
                "name": case.name,
                "design_matrix": expected["design_matrix"],
                "predict": expected["predict"],
            }
        )

    output_path = tmp_path / "binding-output.json"
    output_path.write_text(json.dumps(output))

    validate_binding_output(output_path, cases, atol=atol, rtol=rtol)
