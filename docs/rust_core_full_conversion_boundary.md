# Rust Core Full Conversion Boundary

This note records the current Phase 0 inventory for the Rust core full
conversion track. It is the working boundary map for the remaining Python-owned
core behavior.

## Current Python-Owned Core Behavior

Python still owns these core-adjacent responsibilities:

- runtime helper glue in `pymars/runtime.py`
  - Rust-first portable model export serialization for fitted specs, with
    Python fallback only for unsupported or missing-native cases
  - Rust-first `validate`, `inspect`, `predict`, and `design_matrix` dispatch
    for portable specs, with Python fallback only for unsupported or
    missing-native cases
  - the Rust-first training bridge, with Python fallback only for unsupported
    or missing-native cases
  - row coercion for Rust-backed runtime evaluation
- training orchestration and export logic in `pymars/earth.py`
  - input scrubbing and missing-value handling
  - categorical preprocessing and imputer setup
  - the Python fallback model path used when Rust fitting is unavailable
  - `export_model`, `from_model`, and diagnostics helpers
  - prediction fallback only when Rust-backed portable replay is unavailable
- portable-spec validation and reconstruction helpers in `pymars/_model_spec.py`
  - schema validation
  - conversion between `Earth` objects and portable specs
  - basis-term serialization and reconstruction

## Rust Ownership Boundary

Rust is already the authoritative owner for:

- portable spec loading through the canonical Rust-backed
  `load_model_spec_json_or_path` helper for JSON strings and file paths, with
  `load_model_spec_str` and `load_model_spec_path` as the lower-level Rust
  entry points
- portable model export serialization for compatible specs through the Rust
  runtime bridge
- portable spec validation through the Rust runtime bridge, with Python used
  only as a fallback for unsupported or missing-native cases
- portable `inspect` metadata summaries for compatible specs through the
  native runtime
- `design_matrix`
- `predict`
- the supported Rust-backed training slices exposed through `fit`
- CLI-driven runtime bridging for the non-Python bindings
- benchmark and observability scaffolding for the Rust core

Python remains an adapter layer for:

- import compatibility and the public `pymars.Earth` surface
- lightweight JSON/spec glue
- portable model export fallback for backend errors
- portable spec loading handoff for dict inputs and backend fallback cases;
  `pymars.runtime.load_model_spec` routes JSON strings and file paths through
  Rust first
- portable spec validation and reconstruction helpers in `pymars/_model_spec.py`
- Rust-first training fallback during the conversion when the backend rejects
  a request or is unavailable
- runtime replay fallback for unsupported or backend-rejected specs
- feature-importance and summary helpers that still read the Python model
  object state

## Transitional Fallback Points

The deliberate fallback points that remain documented in code are:

- `pymars.runtime.validate` falls back to Python validation if Rust cannot
  validate the spec or the spec is not Rust-compatible
- `pymars.inspect` uses the Rust-backed metadata slice first and falls back to
  Python adapter behavior if the spec is not Rust-compatible or the backend is
  unavailable
- `pymars.runtime.predict` and `design_matrix` fall back to Python replay for
  unsupported inputs or missing runtime support
- `pymars.runtime.fit_model` returns `None` when Rust training is unavailable
  and otherwise lets supported Rust training failures surface
- `pymars.Earth.fit` uses Python orchestration when the Rust bridge is
  unavailable
- `pymars.Earth.predict` uses the Rust-backed portable replay path first and
  falls back to the reconstructed Python model only when needed
- host-language bindings use their own CLI fallback paths only when the Rust
  runtime binary is missing or incompatible

These fallbacks are narrow and explicit. The removal condition for each one is
fixture-backed parity from the Rust-owned path.

## Host-Language Bridge Dependencies

The current R, Julia, Go, C#, and TypeScript bindings depend on the shared
portable `ModelSpec` contract and the Rust CLI bridge, not on Python core
execution. Their remaining Python dependency is mostly indirect: the portable
spec schema and docs must stay stable while Rust becomes the authoritative core.

## Evidence

The inventory above is supported by the current repository state in:

- `pymars/runtime.py`
- `pymars/earth.py`
- `pymars/_model_spec.py`
- `bindings/r/R/runtime.R`
- `bindings/julia/src/MarsEarth.jl`
- `bindings/go/runtime.go`
- `bindings/csharp/Runtime.cs`
- `bindings/typescript/src/runtime.js`
- `docs/training_orchestration_inventory.md`
- `docs/training_core_migration.md`
- `docs/rust_core.md`

## How This Track Uses the Boundary

Phase 1 should migrate remaining behavior only where the Rust boundary above is
already proven or can be proved by fixtures. Phase 2 should remove any Python
core behavior that is no longer needed once the Rust path is authoritative.
