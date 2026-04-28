# Remaining Roadmap

The retired `ROADMAP.md` and `TODO.md` now delegate planning to Conductor.
This page summarizes the remaining tracks and the cross-track gates that must
hold for the final product goal: a central Rust MARS core surfaced through
Python, R, Julia, Rust, C#, Go, and TypeScript packages.

## Target State

- Runtime replay and training semantics live in the Rust core.
- Python preserves the current scikit-learn-compatible estimator API.
- Each supported language exposes idiomatic bindings over the Rust core.
- Package publication is gated by conformance, package dry-runs, registry
  readiness, and maintainer approval.

## Track Order

1. Replace MVP replay bindings with Rust-backed language bindings.
2. Harden CI/CD, quality gates, security checks, and automation.
3. Prepare release readiness and registry governance.
4. Complete Rust training orchestration and Python integration.
5. Expose Rust training APIs through all language bindings.
6. Publish packages to language package managers with release governance.

## Dependency Gates

Stable runtime packages must not be published until:

- Rust-backed runtime bindings pass conformance in every supported language.
- Duplicated MVP replay logic is removed, isolated as a tested fallback, or the
  package is explicitly labeled experimental/pre-release.
- Install-from-built-artifact smoke tests pass where the ecosystem supports
  local package installation.
- Registry ownership, package names, credentials, and protected release
  environments are confirmed.

Stable training packages must not be published until:

- Rust training orchestration exports conforming `ModelSpec` artifacts.
- Python estimator routing through Rust preserves sklearn-facing behavior.
- Training-capable language bindings pass shared training conformance fixtures.
- Runtime-only packages clearly reject training with stable unsupported-feature
  errors.

## CI/CD and Quality Requirements

Required automation should cover:

- Python tests, docs build, package build, lint/type/security checks where
  configured.
- Rust `cargo fmt --check`, `cargo clippy`, `cargo test`, `cargo package`, and
  dependency/security checks where practical.
- Go, TypeScript, R, Julia, and C# build/test/package checks.
- Cross-platform validation for native artifacts.
- Conformance fixtures run against source packages and built artifacts.
- Least-privilege workflow permissions, concurrency controls, cache hygiene,
  artifact retention, and protected release environments.
- Dependency update automation and explicit maintainer review policy.

## Release Policy

Release readiness is separate from publication. The readiness track must produce
registry ownership status, blocker records, dry-run artifacts, version policy,
and approval gates. The publication track should only execute approved releases
or record unresolved external blockers with owner, action, and date.
