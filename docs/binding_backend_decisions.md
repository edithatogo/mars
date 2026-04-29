# Binding Backend Decisions

The Rust-backed binding track uses a mixed strategy so the language surfaces can
stay idiomatic while sharing one computational core.

## Selected Mechanisms

- Python: `PyO3` with `maturin` via the `pymars_runtime` extension
- R: currently bridged through the Rust runtime CLI
- Julia: currently bridged through the Rust runtime CLI
- Rust: direct crate API
- C#: currently bridged through the Rust runtime CLI
- Go: currently bridged through the Rust runtime CLI
- TypeScript: `wasm-bindgen` and WebAssembly

## Why This Split

- Python needs a native-extension story that preserves sklearn behavior.
- Rust remains the direct implementation host, so the crate API stays primary.
- R, Julia, Go, and C# are currently using a CLI bridge because it is the
  fastest stable path while native interop layers are still being prepared.
- TypeScript benefits from a portable runtime artifact that can be loaded in
  Node and, if required later, in browser contexts.

## Contract Rules

- The Rust core owns validation, evaluation, prediction, and training semantics.
- Host packages own input conversion, packaging, and idiomatic errors.
- All bindings must preserve the shared error categories and fixture contract.
- Runtime-only packages must reject training with explicit unsupported-feature
  errors until training support is implemented.
- The detailed ABI and API rules live in [Binding ABI and API Contract](binding_abi_contract.md).
- Python runtime helpers should dispatch to the Rust extension when the model
  spec is compatible with the shared foreign interface, and otherwise fall back
  to the pure-Python path until the training surface also migrates.
- R, Julia, C#, and Go currently route runtime replay through the Rust CLI
  bridge while the native host bindings are being prepared.
