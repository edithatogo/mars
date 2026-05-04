# Rust Core Full Conversion Boundary

This note records the current Phase 0 inventory for the Rust core full
conversion track. It is the working boundary map for the remaining Python-owned
core behavior.

## Current Python-Owned Core Behavior

Python still owns these core-adjacent responsibilities:

- runtime helper glue in `pymars/runtime.py`
  - loading portable specs from JSON text, paths, and mapping objects
  - saving fitted models to portable JSON
  - the opt-in Rust training bridge guarded by `PYMARS_USE_RUST_TRAINING`
  - Rust-runtime compatibility checks and row coercion
  - replay fallbacks for `predict` and `design_matrix`
- training orchestration and export logic in `pymars/earth.py`
  - input scrubbing and missing-value handling
  - categorical preprocessing and imputer setup
  - the Python fallback model path used when Rust fitting is unavailable
  - `predict`, `export_model`, `from_model`, and diagnostics helpers
- portable-spec validation and reconstruction helpers in `pymars/_model_spec.py`
  - schema validation
  - conversion between `Earth` objects and portable specs
  - basis-term serialization and reconstruction

## Rust Ownership Boundary

Rust is already the authoritative owner for:

- portable spec validation through the native runtime when compatible
- `design_matrix`
- `predict`
- the supported Rust-backed training slices exposed through `fit`
- CLI-driven runtime bridging for the non-Python bindings
- benchmark and observability scaffolding for the Rust core

Python remains an adapter layer for:

- import compatibility and the public `pymars.Earth` surface
- lightweight JSON/spec glue
- environment-gated training fallback during the conversion
- runtime replay fallback for unsupported or incompatible specs
- feature-importance and summary helpers that still read the Python model
  object state

## Transitional Fallback Points

The deliberate fallback points that remain documented in code are:

- `pymars.runtime.validate` falls back to Python validation if Rust cannot
  validate the spec
- `pymars.runtime.predict` and `design_matrix` fall back to Python replay for
  unsupported specs or missing runtime support
- `pymars.runtime.fit_model` is opt-in and returns `None` when Rust training is
  disabled or unavailable
- `pymars.Earth.fit` uses Python orchestration when the Rust bridge does not
  produce a trained spec
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
- `bindings/julia/src/MarsRuntime.jl`
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
