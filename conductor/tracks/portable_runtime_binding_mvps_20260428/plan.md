# Implementation Plan

## Phase 0: Binding Layout

- [x] Task: Create binding package layout
    - [x] Add Go binding package
    - [x] Add TypeScript binding package
    - [x] Add R binding package surface
    - [x] Add Julia binding package surface
    - [x] Add C# binding package surface
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Binding Layout' (Protocol in workflow.md)

## Phase 1: Implement Locally Runnable Bindings

- [x] Task: Implement Go and TypeScript replay bindings
    - [x] Implement `validate`
    - [x] Implement `design_matrix`
    - [x] Implement `predict`
    - [x] Add fixture parity tests
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Implement Locally Runnable Bindings' (Protocol in workflow.md)

## Phase 2: Add Remaining Binding Surfaces

- [x] Task: Add R, Julia, and C# replay package surfaces
    - [x] Add runtime replay source files
    - [x] Add package metadata
    - [x] Add conformance test entrypoints or documented commands
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Add Remaining Binding Surfaces' (Protocol in workflow.md)

## Phase 3: Binding Documentation and Validation

- [x] Task: Document binding status and run available validation
    - [x] Update binding docs with package status
    - [x] Run Go conformance tests
    - [x] Run TypeScript conformance tests
    - [x] Run Rust runtime conformance tests
    - [x] Record unavailable local validation tooling explicitly
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Binding Documentation and Validation' (Protocol in workflow.md)

## Phase 4: Binding CI/CD and Publication Paths

- [x] Task: Add binding CI and package publication scaffolding
    - [x] Add CI jobs for Python, Rust, Go, TypeScript, R, Julia, and C#
    - [x] Add package metadata for npm, NuGet, Julia, R, Go modules, crates.io, and PyPI release paths
    - [x] Add manual or tag-gated publishing workflow scaffolding
    - [x] Document required registry credentials and release approval gates
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Binding CI/CD and Publication Paths' (Protocol in workflow.md)
