# Implementation Plan

## Phase 0: Binding Architecture Decision

- [x] Task: Choose Rust-backed binding mechanisms
    - [ ] Evaluate Python, R, Julia, C#, Go, and TypeScript binding mechanisms
    - [ ] Decide whether the shared boundary is C ABI, PyO3/maturin, WebAssembly, or a mixed strategy
    - [ ] Update `tech-stack.md` with selected tools before implementation
- [x] Task: Define ABI/API compatibility rules
    - [ ] Define allocation, release, and ownership rules
    - [ ] Define null, NaN, missingness, and categorical encoding behavior
    - [ ] Define stable error codes, error payloads, and SemVer compatibility expectations
- [x] Task: Add failing cross-language Rust-core conformance checks
    - [x] Extend the manifest to distinguish Rust-backed tests from MVP replay tests
    - [x] Add failure expectations for languages still using duplicated replay logic
    - [x] Document required local toolchains
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Binding Architecture Decision' (Protocol in workflow.md)

## Phase 1: Stable Rust Foreign Interface

- [x] Task: Implement the shared Rust foreign interface
    - [x] Expose validate, design-matrix, and predict operations through the selected boundary
    - [x] Define stable error codes and message payloads
    - [x] Define ownership, allocation, and release rules
- [x] Task: Validate Rust interface safety
    - [x] Add Rust tests for malformed artifacts, unsupported versions, feature mismatches, and NaN/null behavior
    - [x] Add sanitizer or leak-oriented checks where feasible
    - [x] Document the ABI/API contract for binding authors
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Stable Rust Foreign Interface' (Protocol in workflow.md)

## Phase 2: Python and Rust Package Integration

- [x] Task: Wire Python to the Rust runtime/core
    - [x] Add build configuration for the selected Python binding mechanism
    - [x] Route runtime replay calls through Rust where available
    - [x] Preserve sklearn estimator behavior and fallback support
- [x] Task: Harden the Rust crate as the native binding source
    - [x] Keep the crate API idiomatic for Rust users
    - [x] Verify `cargo test`, `cargo package`, and conformance tests
    - [x] Update Rust package documentation
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Python and Rust Package Integration' (Protocol in workflow.md)

## Phase 3: R, Julia, C#, and Go Bindings

- [x] Task: Replace R and Julia replay logic with Rust-backed calls
    - [x] Preserve current public function names
    - [x] Translate Rust errors into idiomatic host-language errors
    - [x] Run shared fixtures from each package test entrypoint
- [x] Task: Replace C# and Go replay logic with Rust-backed calls
    - [x] Preserve package APIs and type safety
    - [x] Validate ownership and memory-release rules
    - [x] Run shared fixtures from each package test entrypoint
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: R, Julia, C#, and Go Bindings' (Protocol in workflow.md)

## Phase 4: TypeScript Binding

- [ ] Task: Replace TypeScript replay logic with a Rust-backed package
    - [ ] Implement the selected Rust-backed TypeScript mechanism
    - [ ] Support Node-based fixture tests
    - [ ] Document browser support only if the chosen mechanism supports it
- [ ] Task: Validate package ergonomics
    - [ ] Run `npm test`
    - [ ] Run `npm pack --dry-run`
    - [ ] Verify exported module shape and README examples
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: TypeScript Binding' (Protocol in workflow.md)

## Phase 5: Binding CI Consolidation

- [ ] Task: Update CI for Rust-backed bindings
    - [ ] Build the Rust core before language test jobs
    - [ ] Cache language toolchains and Rust artifacts safely
    - [ ] Run conformance for every supported language
- [ ] Task: Add built-artifact installation smoke tests
    - [ ] Build local package artifacts for each supported language
    - [ ] Install artifacts into clean temporary environments where feasible
    - [ ] Run a shared conformance smoke fixture from installed packages
- [ ] Task: Remove or isolate duplicated replay logic
    - [ ] Delete replaced host-language replay implementations where practical
    - [ ] Mark any retained fallback paths as temporary and tested
    - [ ] Update binding docs with the new architecture
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 5: Binding CI Consolidation' (Protocol in workflow.md)
