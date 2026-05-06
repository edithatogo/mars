# Rust Core Profiling and Observability Evidence

## Phase 0: Inventory and Baseline

- The Rust runtime boundaries are centered in `rust-runtime/src/runtime.rs`,
  `rust-runtime/src/python.rs`, `rust-runtime/src/training.rs`, and the CLI in
  `rust-runtime/src/bin/mars-runtime-cli.rs`.
- Runtime paths now include validation, design-matrix construction,
  prediction, and training-adjacent flows through criterion benches in
  `rust-runtime/benches/runtime_bench.rs`.
- Observability is opt-in and low-overhead. Structured spans are present in the
  Rust runtime/CLI path, while production defaults remain quiet.
- CI now exercises the Rust quality gates separately from the release-rehearsal
  path via `rust-core-quality.yml` and `rust-core-benchmarks.yml`.
- Documentation pages exist in `docs/rust_core_observability.md`,
  `docs/core_transition_evidence.md`, `docs/remaining_roadmap.md`, and
  `conductor/tech-stack.md`.

## Phase 1: Rust-Native Profiling and Benchmarking

- The benchmark strategy is defined around stable runtime scenarios and
  repeatable fixture-backed runs.
- `criterion` is the primary benchmark harness.
- The benchmark boundary is intentionally limited to runtime replay and
  training-adjacent Rust paths; GPU profiling remains out of scope until GPU
  kernels exist.

## Phase 2: CI and Code-Quality Gates

- `cargo nextest` is the fast Rust test runner for regression checks.
- `cargo deny` is used for dependency and license hygiene.
- Benchmark and observability runs remain separate from fast correctness gates.
- Release-rehearsal guidance points to the deeper profiling checks without
  requiring them on every commit.

## Phase 3: Documentation and Release Guidance

- Maintainer-facing profiling guidance is documented in
  `docs/rust_core_observability.md`.
- The roadmap and release-facing docs now point maintainers to the Rust core
  profiling workflow without promising unimplemented GPU or heavyweight
  release-only tooling.
- The observability guidance preserves the public API boundary while documenting
  the internal diagnostics contract for maintainers.

## Phase 4: Final Validation and Handoff Readiness

- Validation completed successfully with:
  - `cargo test --manifest-path rust-runtime/Cargo.toml --test training_tests --test foreign_tests --test fixture_tests`
  - `cargo nextest run --manifest-path rust-runtime/Cargo.toml --workspace --all-features`
  - `cd rust-runtime && cargo deny check --config deny.toml`
  - `uv run mkdocs build --strict`
  - `vale --config .vale.ini docs/rust_core_observability.md docs/ci_quality.md docs/remaining_roadmap.md conductor/tech-stack.md`
  - `git diff --check`
- Outstanding follow-ups are intentionally out of scope for this track:
  - GPU profiling unless GPU kernels are introduced
  - heavyweight profiling gates in the fast CI path
