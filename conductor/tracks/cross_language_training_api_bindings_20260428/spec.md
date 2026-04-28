# Specification: Cross-Language Training API Bindings

## Overview

Rust-backed runtime bindings make inference portable. The final product goal
also requires the central Rust MARS core to be accessible from Python, R, Julia,
Rust, C#, Go, and TypeScript for fitting/training where each ecosystem can
support it safely.

This track exposes the completed Rust training orchestrator through the existing
language package surfaces after the Rust training core and Rust-backed runtime
bindings are stable.

## Dependency Notes

- Depends on Rust-backed runtime language bindings.
- Depends on Rust training orchestration and Python integration.
- Blocks stable package publication for packages that claim training support.

## Functional Requirements

- Define a shared training API over the Rust core for normalized inputs,
  hyperparameters, optional sample weights, feature metadata, and fitted
  `ModelSpec` export.
- Surface idiomatic training/fit APIs for Python, R, Julia, Rust, C#, Go, and
  TypeScript where the package target supports training.
- Preserve Python sklearn compatibility and current import conventions.
- Translate Rust training errors into host-language error categories.
- Add training conformance fixtures that validate fitted model structure and
  predictions across languages.
- Document which packages support training, runtime replay only, or
  experimental/pre-release training.

## Non-Functional Requirements

- Host bindings must not reimplement training semantics.
- Fitted outputs must remain portable `ModelSpec` artifacts.
- Numeric parity must be bounded and fixture-backed.
- Training APIs must fail with clear unsupported-feature errors where a language
  surface intentionally ships runtime-only support.

## Acceptance Criteria

- Python, Rust, and at least one non-Python binding can fit through the Rust core
  and export a conforming `ModelSpec`.
- All supported training-capable bindings pass shared training conformance
  fixtures.
- Runtime-only packages are clearly labeled and tested as runtime-only.
- Package docs include idiomatic train/predict/export examples.
- CI separates runtime conformance from training conformance.

## Out of Scope

- Publishing packages.
- Changing `ModelSpec` semantics without schema-version migration.
- Adding new algorithms beyond MARS.
