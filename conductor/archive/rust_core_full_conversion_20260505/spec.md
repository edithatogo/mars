# Specification: Rust Core Full Conversion and Python Boundary Retirement

## Overview

Move the remaining mars computational core toward a Rust-first implementation
so validation, replay, basis evaluation, training orchestration, export, and
core diagnostics are owned by Rust wherever the current architecture allows it.
The Python surface must remain stable during the transition, but Python should
become a thin compatibility layer rather than the place where core behavior
continues to live.

This is a conversion track, not a user-facing API expansion track. It should
not change the `pymars` import surface, package branding, or serialized model
contract unless a separate API migration track exists.

## Context

- The repository already has a Rust runtime core, Rust-backed replay, Rust
  training primitives, and binding shims for the other languages.
- A number of runtime and training paths still have Python-side fallback or
  orchestration logic, but supported training and replay now route through
  Rust first rather than waiting on a feature gate.
- The project already has dedicated tracks for profiling/observability and
  parity audit work, so this track should focus on ownership transfer and
  retirement of Python core responsibilities.

## Phase 0 Working Boundary

The current evidence base indicates:

- Python still owns runtime helper glue, adapter-only preprocessing, and some
  fallback logic in `pymars/runtime.py` and `pymars/earth.py` for unsupported
  or Rust-rejected cases.
- Python still owns portable-spec serialization, validation, and model
  reconstruction helpers in `pymars/_model_spec.py`.
- Rust already owns portable replay, supported training slices, and the CLI
  bridge used by the other language packages.
- Rust now also owns the default training bridge for supported requests.
- The remaining host-language packages depend on the shared portable
  `ModelSpec` contract and the Rust CLI, not on Python core execution.
- The conversion work should therefore focus on shrinking Python to adapter
  behavior and retiring fallback paths only after fixture-backed parity proof.

See [Rust Core Full Conversion Boundary](../../docs/rust_core_full_conversion_boundary.md)
for the working inventory and boundary map captured from the repository.

## Functional Requirements

- Inventory the remaining Python-owned core responsibilities:
  - runtime replay helpers
  - training orchestration glue
  - model export and serialization helpers
  - diagnostics or fallback logic that still changes core behavior
- Define the final Rust ownership boundary for core behavior:
  - what Rust must own before Python can become a thin adapter
  - what can remain in Python as compatibility or import glue
  - what can be removed only after parity proof
- Migrate remaining core logic into Rust in staged slices:
  - preserve behavior, error handling, and fixture parity
  - keep the Python public API stable during each slice
  - prefer explicit feature flags or environment gates for transitional paths
- Retire Python fallback logic once Rust parity is proven:
  - remove duplicate execution where Rust is authoritative
  - keep temporary fallback only where a parity gap is documented and
    deliberate
- Align bindings and host-language bridges to the Rust-owned core:
  - Python should call Rust by default for supported flows
  - other language bindings should remain compatible with the same Rust core
- Update documentation and release guidance to reflect the ownership shift:
  - describe what Rust owns
  - describe what Python still owns
  - explain any deliberate transitional fallback behavior

## Non-Functional Requirements

- Preserve current public API names and import conventions.
- Keep transitional fallbacks explicit, narrow, and time-bound.
- Keep the conversion deterministic and fixture-backed.
- Avoid introducing new model semantics while moving ownership.
- Keep the plan parallel-friendly across six workers with disjoint ownership.

## Parallel Work Model

- Agent 1: inventory of remaining Python-owned core paths and boundaries.
- Agent 2: Rust ownership design and migration targets.
- Agent 3: Python adapter simplification and fallback retirement.
- Agent 4: binding alignment and transitional compatibility checks.
- Agent 5: documentation and release guidance for the ownership shift.
- Agent 6: validation, coherence review, and evidence consolidation.

## Acceptance Criteria

- The remaining Python-owned core paths are inventoried and bounded.
- A Rust-first ownership map exists for the remaining core responsibilities.
- Staged migration work exists for each remaining Python-owned slice.
- Transitional Python fallback behavior is explicit and documented.
- The Python surface remains stable for supported users.
- The track is decomposed so six workers can operate independently.
- Every phase ends with a Conductor review checkpoint.
- Final validation confirms the conversion plan does not require public API
  changes to proceed.

## Out of Scope

- Changing package names or import names.
- Adding new model families or user-facing features.
- Reworking the release registries.
- Replacing the Rust runtime with a different engine.
- Auditing upstream parity libraries in depth; that belongs to the parity audit
  track.
