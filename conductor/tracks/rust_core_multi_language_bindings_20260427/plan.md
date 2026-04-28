# Implementation Plan

## Phase 0: Architecture Alignment

- [x] Task: Define the Rust core ownership model
    - [x] Decide which modules belong in the Rust core first
    - [x] Define the relationship between `ModelSpec`, Rust structs, and host-language wrappers
    - [x] Define error semantics shared across bindings
    - [x] Define the minimum stable Rust crate API
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Architecture Alignment' (Protocol in workflow.md)

## Phase 1: Core Runtime Consolidation

- [x] Task: Promote the Rust replay prototype into a core crate boundary
    - [x] Separate reusable core logic from test-only fixture harnesses
    - [x] Keep validation, `design_matrix`, and `predict` parity with Python fixtures
    - [x] Add Rust crate-level API documentation
    - [x] Preserve JSON fixture compatibility
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Core Runtime Consolidation' (Protocol in workflow.md)

## Phase 2: Python Binding Strategy

- [x] Task: Plan and prototype Python integration with the Rust core
    - [x] Choose the Python binding mechanism
    - [x] Preserve `pymars.Earth` and sklearn-compatible wrappers
    - [x] Route portable runtime helpers through Rust where feasible
    - [x] Define fallback and packaging behavior for source installs
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Python Binding Strategy' (Protocol in workflow.md)

## Phase 3: Multi-Language Binding Plans

- [x] Task: Define package surfaces for additional languages
    - [x] Define R package surface
    - [x] Define Julia package surface
    - [x] Define Rust crate public surface
    - [x] Define C# package surface
    - [x] Define Go package surface
    - [x] Define TypeScript package surface, including WASM/native decision points
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Multi-Language Binding Plans' (Protocol in workflow.md)

## Phase 4: Shared Conformance Harness

- [x] Task: Create binding-agnostic conformance requirements
    - [x] Define fixture discovery and expected-output format
    - [x] Define runtime validation cases every binding must pass
    - [x] Define numerical tolerances by operation
    - [x] Define CI expectations for each binding family
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Shared Conformance Harness' (Protocol in workflow.md)
