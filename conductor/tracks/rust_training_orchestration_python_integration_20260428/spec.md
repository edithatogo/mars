# Specification: Complete Rust Training Orchestration and Python Integration

## Overview

The current Rust runtime owns portable replay and the first training primitives:
basis evaluation, weighted least-squares/RSS, GCV scoring, candidate scoring,
and pruning subset scoring. This track completes the next step: a Rust training
orchestrator that can fit MARS models end to end and export the same portable
`ModelSpec` contract used by the existing Python package.

The public Python API remains the compatibility baseline. `Earth.fit`,
`Earth.predict`, estimator parameters, sklearn behavior, and import conventions
must remain stable while Rust is introduced behind that surface.

## Dependency Notes

- Should start after the Rust-backed runtime binding boundary is selected and
  CI quality gates are defined.
- Blocks cross-language training API bindings.
- Does not by itself publish packages; publication remains gated by release
  readiness and registry approval.

## Functional Requirements

- Implement a Rust training API that accepts normalized training inputs,
  hyperparameters, optional sample weights, and feature metadata.
- Move full forward-pass orchestration into Rust, including candidate
  generation, knot selection, interaction-degree constraints, stopping rules,
  and deterministic tie handling.
- Move pruning orchestration into Rust, including subset search, GCV scoring,
  coefficient refits, and final basis selection.
- Export fitted models from Rust as versioned `ModelSpec` artifacts compatible
  with Python, Rust, and all binding conformance fixtures.
- Add Python integration behind the existing estimator API without changing
  sklearn-facing behavior.
- Preserve a controlled fallback path to the current Python implementation until
  Rust parity is proven.
- Keep runtime replay conformance passing while training internals migrate.

## Non-Functional Requirements

- Training parity must be fixture-backed against the current Python behavior.
- Numerical differences must be bounded, documented, and tested with explicit
  tolerances.
- The Rust core must keep deterministic behavior for identical inputs.
- Python-specific code must remain scikit-learn compatible and must not expose
  Rust implementation details as required user-facing concepts.
- Any new binding or build technology must be documented in `tech-stack.md`
  before implementation.

## Acceptance Criteria

- Rust can fit representative regression models end to end and export
  `ModelSpec` artifacts that pass replay conformance.
- Python `Earth.fit` can route through Rust under an explicit internal flag,
  environment flag, or feature gate while preserving current public API behavior.
- Existing Python tests and Rust tests pass.
- New parity fixtures cover linear terms, hinge terms, interactions, pruning,
  sample weights, missingness behavior where supported, and deterministic ties.
- Documentation explains the training-core migration boundary and any remaining
  Python-owned estimator responsibilities.

## Out of Scope

- Publishing language packages to external registries.
- Replacing all language bindings with Rust-backed native packages.
- Changing the public Python estimator API or dropping sklearn compatibility.
- Removing the Python fallback before Rust parity is complete.
