# Language Binding Plan

The target `mars` surface spans Python, R, Julia, Rust, C#, Go, and TypeScript.
All bindings should share the Rust core and the same fixture-backed conformance
requirements.

## Python

Python remains the compatibility baseline.

- Preserve `pymars.Earth`, `EarthRegressor`, `EarthClassifier`, `EarthCV`, and
  `GLMEarth` import behavior.
- Keep scikit-learn compatibility in Python wrappers.
- Route portable runtime helpers through the Rust extension first when the
  model spec is supported by the shared ABI, and fall back to the Python
  implementation for unsupported replay shapes.
- Move fitting internals behind the same public estimator API once the Rust core
  reaches parity.
- Package the extension with `maturin`/PyO3.

## Rust

Rust is the core implementation surface.

- Publish the core as a crate with documented replay and eventually fitting APIs.
- Treat `ModelSpec` compatibility as a public contract.
- Keep fixture parity tests in the crate.

## R

The R package should expose an R-native modeling interface while sharing the
Rust core.

- Provide fit/predict helpers with formula/data-frame ergonomics where feasible.
- Preserve `ModelSpec` import/export.
- Use shared conformance fixtures for runtime behavior.

## Julia

The Julia package should expose a type-stable wrapper around the Rust core.

- Provide direct matrix/table prediction APIs.
- Preserve `ModelSpec` import/export.
- Keep numerical parity with the shared fixtures.

## C#

The C# package should target .NET consumers that need embedded prediction and
eventual fitting.

- Current runtime replay is bridged through the Rust CLI while native interop
  is prepared.
- Expose strongly typed model and prediction APIs.
- Preserve error categories from the Rust core.

## Go

The Go package should prioritize service embedding and simple deployment.

- Expose model loading, validation, design-matrix evaluation, and prediction.
- Current runtime replay is bridged through the Rust CLI while native interop
  is prepared.
- Keep data conversion explicit and allocation behavior predictable.
- Preserve fixture-backed parity with Rust.

## TypeScript

TypeScript should support JavaScript and web runtimes through a WASM-first path
unless native Node packaging proves necessary.

- Provide model loading, validation, and prediction APIs.
- Support browser-compatible artifact replay when possible.
- Preserve fixture-backed parity with Rust.

## Binding Priority

Recommended implementation order:

1. Rust core crate boundary.
2. Python runtime integration.
3. Rust fitting APIs.
4. R and Julia bindings.
5. Go and C# bindings.
6. TypeScript/WASM binding.

The order can change for user demand, but every binding must pass the shared
conformance harness before being treated as supported.
