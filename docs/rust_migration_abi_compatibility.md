# Rust Migration and ABI Compatibility

This page defines the non-breaking migration posture for future Rust-first
work. The project should keep the Python API stable while moving more shared
runtime behavior behind explicit Rust-owned boundaries.

## Current Boundary

Rust is already authoritative for portable runtime behavior that can be
represented through `ModelSpec`:

- portable spec loading for JSON strings and paths
- compatible spec validation
- design-matrix evaluation
- prediction/replay
- inspect/export normalization
- supported baseline training slices
- CLI bridge surfaces for host-language bindings

Python remains the stable public adapter:

- `import pymars as earth`
- `earth.Earth(...)`
- scikit-learn estimator behavior
- unsupported-case fallback
- diagnostics that depend on Python estimator state

## ABI Recommendation

Adopt a narrow ABI only for stable runtime primitives. Do not freeze training
internals yet.

Recommended first ABI scope:

- ABI version query
- load/validate `ModelSpec`
- design-matrix evaluation
- prediction/replay
- inspect summary
- export normalization
- error allocation and free functions

Do not include initially:

- pruning internals
- full training search internals
- diagnostics formatting
- plotting
- host-language convenience APIs

## API-Preserving Migration

The user-facing API remains unchanged:

```python
import pymars as earth

model = earth.Earth()
model.fit(X, y)
model.predict(X)
```

Adapters can route through Rust, CLI, FFI, or a future ABI without changing the
public surface. That requires:

- host wrappers own native type conversion
- Rust owns computation and portable spec semantics
- every ABI call has explicit ownership and lifetime rules
- every allocated buffer has a matching free function
- errors return structured codes plus a readable message
- fallback remains explicit and tested

## ABI Design Rules

| Rule | Rationale |
| --- | --- |
| Opaque handles only | avoids exposing Rust internals across ABI boundaries |
| Versioned symbols or version query | allows host wrappers to fail clearly on incompatible runtime versions |
| Caller-owned inputs, runtime-owned outputs | keeps memory ownership understandable |
| Explicit free functions | required for safe C ABI use from R, Julia, C#, Go, and Python |
| JSON-compatible errors | keeps error reporting portable across bindings |
| Fixture-backed conformance | prevents ABI drift from changing model semantics |

## Future Migration Slices

1. ABI proof-of-concept for validate/predict/export.
2. Host-wrapper experiment in one non-Python binding.
3. Cross-language conformance tests through the ABI path.
4. Training ABI decision after parity and ownership rules are stable.
5. Optional Arrow input/output helpers once memory ownership is proven.

## Review Gates

No future Rust migration slice should land without:

- fixture parity tests
- negative tests for unsupported cases
- public API compatibility check
- binding conformance update when applicable
- docs update for ownership and fallback behavior

## References

- Binding ABI contract: [Binding ABI and API Contract](binding_abi_contract.md)
- Rust ownership boundary: [Rust Core Ownership Boundary](rust_core_ownership.md)
- Rust full conversion boundary: [Rust Core Full Conversion Boundary](rust_core_full_conversion_boundary.md)
