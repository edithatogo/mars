# Binding Backend Decisions

The Rust-backed binding track uses a mixed strategy so the language surfaces can
stay idiomatic while sharing one computational core.

## Selected Mechanisms

- Python: `PyO3` with `maturin`
- R: `extendr`
- Julia: C ABI via `ccall`
- Rust: direct crate API
- C#: C ABI via P/Invoke
- Go: C ABI via `cgo`
- TypeScript: `wasm-bindgen` and WebAssembly

## Why This Split

- Python needs a native-extension story that preserves sklearn behavior.
- Rust remains the direct implementation host, so the crate API stays primary.
- R, Julia, Go, and C# can all consume a stable C ABI boundary cleanly.
- TypeScript benefits from a portable runtime artifact that can be loaded in
  Node and, if required later, in browser contexts.

## Contract Rules

- The Rust core owns validation, evaluation, prediction, and training semantics.
- Host packages own input conversion, packaging, and idiomatic errors.
- All bindings must preserve the shared error categories and fixture contract.
- Runtime-only packages must reject training with explicit unsupported-feature
  errors until training support is implemented.
- The detailed ABI and API rules live in [Binding ABI and API Contract](binding_abi_contract.md).
