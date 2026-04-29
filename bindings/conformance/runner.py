"""Shared conformance checks for portable runtime bindings."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "bindings" / "conformance" / "manifest.json"


@dataclass(frozen=True)
class FixtureCase:
    """Resolved conformance fixture pair."""

    name: str
    model_spec: Path
    runtime_fixture: Path


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[list[FixtureCase], float, float]:
    """Load and validate the conformance manifest."""
    raw = json.loads(path.read_text())
    binding_modes = raw.get("binding_modes", {})
    if "runtime_mvp" not in binding_modes:
        raise AssertionError("Conformance manifest must declare runtime_mvp binding mode")
    if "runtime_rust_backed" not in binding_modes:
        raise AssertionError(
            "Conformance manifest must declare runtime_rust_backed binding mode"
        )
    if "training_rust_backed" not in binding_modes:
        raise AssertionError(
            "Conformance manifest must declare training_rust_backed binding mode"
        )
    tolerances = raw.get("tolerances", {})
    atol = float(tolerances.get("absolute", 1e-12))
    rtol = float(tolerances.get("relative", 1e-12))
    cases: list[FixtureCase] = []
    for item in raw.get("fixtures", []):
        name = item["name"]
        model_spec = (path.parent / item["model_spec"]).resolve()
        runtime_fixture = (path.parent / item["runtime_fixture"]).resolve()
        if not model_spec.exists():
            raise AssertionError(f"Missing model spec fixture for {name}: {model_spec}")
        if not runtime_fixture.exists():
            raise AssertionError(
                f"Missing runtime fixture for {name}: {runtime_fixture}"
            )
        cases.append(
            FixtureCase(
                name=name,
                model_spec=model_spec,
                runtime_fixture=runtime_fixture,
            )
        )
    if not cases:
        raise AssertionError("Conformance manifest must include at least one fixture")
    return cases, atol, rtol


def validate_expected_fixtures(cases: list[FixtureCase]) -> None:
    """Validate expected runtime fixture shape."""
    for case in cases:
        payload = json.loads(case.runtime_fixture.read_text())
        for field in ("probe", "design_matrix", "predict"):
            if field not in payload:
                raise AssertionError(f"{case.name} fixture missing {field}")
        if len(payload["probe"]) != len(payload["design_matrix"]):
            raise AssertionError(f"{case.name} probe/design_matrix row mismatch")
        if len(payload["probe"]) != len(payload["predict"]):
            raise AssertionError(f"{case.name} probe/predict length mismatch")


def validate_binding_output(
    output_path: Path,
    cases: list[FixtureCase],
    *,
    atol: float,
    rtol: float,
) -> None:
    """Validate a binding-emitted parity output file."""
    raw = json.loads(output_path.read_text())
    outputs = {item["name"]: item for item in raw.get("fixtures", [])}
    missing = {case.name for case in cases} - set(outputs)
    if missing:
        raise AssertionError(f"Binding output missing fixtures: {sorted(missing)}")

    for case in cases:
        expected = json.loads(case.runtime_fixture.read_text())
        actual = outputs[case.name]
        assert_nested_close(
            actual.get("design_matrix"),
            expected["design_matrix"],
            label=f"{case.name}.design_matrix",
            atol=atol,
            rtol=rtol,
        )
        assert_nested_close(
            actual.get("predict"),
            expected["predict"],
            label=f"{case.name}.predict",
            atol=atol,
            rtol=rtol,
        )


def assert_nested_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    atol: float,
    rtol: float,
) -> None:
    """Compare nested JSON numeric values with NaN parity."""
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(f"{label} expected list, got {type(actual).__name__}")
        if len(actual) != len(expected):
            raise AssertionError(
                f"{label} length mismatch: {len(actual)} != {len(expected)}"
            )
        for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_nested_close(
                actual_item,
                expected_item,
                label=f"{label}[{idx}]",
                atol=atol,
                rtol=rtol,
            )
        return

    actual_float = _json_number_to_float(actual)
    expected_float = _json_number_to_float(expected)
    if math.isnan(expected_float):
        if not math.isnan(actual_float):
            raise AssertionError(f"{label} expected NaN, got {actual_float}")
        return
    if not math.isclose(actual_float, expected_float, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(
            f"{label} mismatch: {actual_float} != {expected_float}"
        )


def _json_number_to_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    raise AssertionError(f"Expected numeric JSON value or null, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases, atol, rtol = load_manifest(args.manifest)
    validate_expected_fixtures(cases)
    if args.output is not None:
        validate_binding_output(args.output, cases, atol=atol, rtol=rtol)


if __name__ == "__main__":
    main()
