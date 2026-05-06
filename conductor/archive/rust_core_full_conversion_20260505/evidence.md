# Rust Core Full Conversion Evidence

## Phase 0: Inventory and Boundary Definition

## Remaining Python-Owned Core Behavior

The current repository still keeps several core responsibilities in Python:

- the public `Earth` estimator surface in `pymars/earth.py`
- Python compatibility wrappers around runtime replay and training
- Python-side serialization and model-spec conversion helpers in
  `pymars/_model_spec.py`
- fallback logic that preserves the existing public API while delegating to
  Rust when the native runtime path is available
- adapter-side feature-importance and diagnostics preservation

## Rust-Owned Core Behavior Already in Place

Rust already owns:

- portable `ModelSpec` validation
- basis evaluation, design-matrix generation, and prediction replay
- a substantial portion of the training path and export logic
- the CLI bridge used by non-Python bindings
- Rust-native benchmark and observability scaffolding

## Boundary Notes

The conversion track should treat Python as an adapter boundary and retire
duplicate execution only after Rust parity is proven for each slice. The
public `pymars` import surface should remain unchanged while ownership moves.

## Phase 1: Rust Ownership Migration Slice Evidence

The first landed phase-1 slice now covers the Rust-backed portable-spec
loading, validation, replay, inspection, export-normalization, and training
paths that are safe to route through Rust first. The repository routes JSON
string and file-path spec inputs through Rust canonical loading, prefers
Rust-backed validation for compatible specs, uses the Rust runtime for
compatible `inspect` summaries, routes portable replay/export normalization
through Rust first, and defaults to the Rust training bridge before falling
back to Python only for unsupported or incompatible cases.

### Slice Records

- `source`: `rust-runtime/src/runtime.rs`, `rust-runtime/src/python.rs`
  - `claim`: Rust owns canonical portable spec loading for JSON strings and
    file paths, the portable `inspect` summary slice for compatible specs,
    and the Rust-first validation / replay / export-normalization / training
    helpers for compatible specs.
  - `evidence`: `load_model_spec_json_or_path` dispatches JSON/path inputs to
    Rust canonical loading; `validate_model_spec_json`,
    `predict_json`, `design_matrix_json`, `inspect_model_spec_json`, and
    `export_model_json` are exported by the Python module and exercised by
    the Rust unit tests in `rust-runtime/src/python.rs`; the Rust training
    bridge is exercised by the training tests and the Python routing tests.
  - `status`: `confirmed`
  - `priority`: `parity-critical`
  - `next_action`: `Phase 2 boundary retirement and fallback removal`

- `source`: `pymars/runtime.py`
  - `claim`: Python still acts as the adapter for dict inputs and explicit
    fallback cases, but routes portable JSON/path inputs through Rust first
    when the compiled backend is available and routes compatible validation,
    replay, export, and training operations through Rust before falling back.
  - `evidence`: `load_model_spec()` now calls
    `_rust_backend.load_model_spec_canonical_json(...)` before falling back to
    Python parsing; `validate()` checks the Rust path first for compatible
    specs; `predict()` and `design_matrix()` call the Rust bridge for portable
    specs first; `inspect()` prefers `_rust_backend.inspect_model_spec_json`
    for compatible specs; `export_model_json()` routes compatible exports
    through Rust normalization first; and `fit_model()` now defaults to the
    Rust training bridge when it is available.
  - `status`: `confirmed`
  - `priority`: `parity-critical`
  - `next_action`: `Retire the remaining adapter-only fallback slices`

- `source`: `tests/test_python_routing.py`
  - `claim`: The migrated slice is covered by tests for Rust routing, fallback
    behavior, Rust-backed inspect, validation, replay, export, and training.
  - `evidence`: `test_runtime_load_model_spec_routes_through_rust`,
    `test_runtime_load_model_spec_falls_back_when_rust_loader_fails`, and
    `test_runtime_validate_uses_rust_backend_for_portable_specs`,
    `test_runtime_inspect_uses_rust_backend_for_supported_specs`,
    `test_runtime_export_model_json_routes_through_rust`,
    `test_runtime_save_model_routes_through_rust_export`, and
    `test_earth_predict_uses_rust_backend_when_available`,
    `test_rust_training_bridge_can_fit_without_environment_gate`,
    `test_rust_training_bridge_sends_routing_flags`, and
    `test_rust_training_bridge_preserves_diagnostics` validate the current
    contract.
  - `status`: `confirmed`
  - `priority`: `parity-critical`
  - `next_action`: `Use as the baseline for the next migration slice review`

## Phase 1 Review Position

Phase 1 is now complete. The landed Rust-backed spec-loading, validation,
replay, inspect, export-normalization, and supported training slice has moved
the supported ownership boundary into Rust-first execution. The remaining
open work now belongs to phase 2: reducing Python to adapter-only behavior and
retiring deliberate fallback paths only where parity is proven.

## Phase 2: Python Boundary Retirement Slice Evidence

The phase-2 boundary retirement slice is now also in place. Supported runtime
paths are Rust-first and no longer silently fall back to Python when the Rust
backend fails on a compatible request. Python remains available only for
explicitly unsupported or missing-native cases, plus the adapter glue needed
for invalid payloads and reconstruction helpers.

### Slice Records

- `source`: `pymars/runtime.py`
  - `claim`: Supported portable validation, inspect, export, predict, design
    matrix, and training paths route through Rust first, while unsupported or
    missing-native cases retain explicit Python fallback.
  - `evidence`: the runtime now lets Rust errors surface on compatible
    validation, inspect, export, predict, design-matrix, and training calls;
    Python fallback remains only for incompatible specs, missing native
    support, or invalid-payload validation cases.
  - `status`: `confirmed`
  - `priority`: `parity-critical`
  - `next_action`: `Phase 3 documentation and release guidance alignment`

- `source`: `pymars/_model_spec.py`
  - `claim`: The portable-model reconstruction helpers now tolerate Rust
    training payloads that omit optional score fields or carry Rust-only
    training parameters such as `threshold` and `feature_names`.
  - `evidence`: `spec_to_model` ignores Rust-only training parameters and
    normalizes missing score values to keep the adapter boundary stable.
  - `status`: `confirmed`
  - `priority`: `important`
  - `next_action`: `Phase 3 docs and release guidance should describe the
    narrower Python adapter surface`

- `source`: `tests/test_python_routing.py`, `tests/test_python_training_conformance.py`
  - `claim`: The supported Rust-first paths are covered by regression tests
    that expect errors to propagate on compatible specs and preserve explicit
    fallback only for unsupported or incompatible cases.
  - `evidence`: training, validation, export, predict, inspect, and
    unsupported-case fallback paths are covered by the Python routing and
    training-conformance suites.
  - `status`: `confirmed`
  - `priority`: `parity-critical`
  - `next_action`: `Use as the baseline for phase-3 documentation sync`

## Final Position

The Rust core full conversion track is now complete. Supported replay,
validation, inspect, export-normalization, and training paths are owned by the
Rust runtime first; Python remains as compatibility glue plus explicit
unsupported or missing-native fallback. The track has moved from migration to
archival close-out.
