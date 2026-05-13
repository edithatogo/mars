# CI and Quality Policy

This repo enforces a conservative quality gate:

- Python tests and coverage thresholds
- `ruff` on package code, `ruff format` on package code and tests, `ty`, and
  `codespell`
- Vale prose lint
- `mkdocs --strict`
- Rust core dependency and test hygiene, with `cargo deny` and `cargo nextest`
  as the preferred Rust-side quality tools and the same commands recommended
  for local validation before pushing
- Rust core observability and profiling guidance as documented in
  [Rust Core Observability and Profiling](rust_core_observability.md), with
  benchmarks and profiling artifacts kept out of the public API surface
- R package build/check/manual validation when the R package is part of a
  release rehearsal
- release rehearsal and artifact inspection
- package alignment checks so docs, manifests, and `release_metadata.json` stay in sync

The primary CI and local validation paths are strict. The intentionally
advisory jobs are limited to exploratory or release-adjacent workflows:

- profiling
- benchmark collection
- mutation testing
- release publish rehearsal gates where external registries or approvals are the
  blocking factor

Recommended local Rust commands:

```bash
cargo deny --manifest-path rust-runtime/Cargo.toml check
cargo nextest run --manifest-path rust-runtime/Cargo.toml
cargo test --manifest-path rust-runtime/Cargo.toml
cargo bench --manifest-path rust-runtime/Cargo.toml --bench runtime_bench --no-run
```

See [Supply Chain Security](supply_chain.md) for provenance and release
automation policy.
