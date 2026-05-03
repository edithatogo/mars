# Rust Core

The Rust runtime is the shared computational boundary for portable replay and
training primitives.

It currently provides:

- portable `ModelSpec` validation
- `design_matrix`
- `predict`
- Rust-backed training entrypoints for the supported baseline cases

See [Training Core Migration](training_core_migration.md) for the remaining
parity gaps.
