# Implementation Plan

## Phase 0: Binding Architecture Decision

- [x] Task: Choose Rust-backed binding mechanisms
    - [ ] Evaluate Python, R, Julia, C#, Go, and TypeScript binding mechanisms
    - [ ] Decide whether the shared boundary is C ABI, PyO3/maturin, WebAssembly, or a mixed strategy
    - [ ] Update `tech-stack.md` with selected tools before implementation
- [~] Task: Define ABI/API compatibility rules
    - [ ] Define allocation, release, and ownership rules
    - [ ] Define null, NaN, missingness, and categorical encoding behavior
    - [ ] Define stable error codes, error payloads, and SemVer compatibility expectations
- [ ] Task: Add failing cross-language Rust-core conformance checks
    - [ ] Extend the manifest to distinguish Rust-backed tests from MVP replay tests
    - [ ] Add failure expectations for languages still using duplicated replay logic
    - [ ] Document required local toolchains
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Binding Architecture Decision' (Protocol in workflow.md)

## Phase 1: Stable Rust Foreign Interface

- [ ] Task: Implement the shared Rust foreign interface
    - [ ] Expose validate, design-matrix, and predict operations through the selected boundary
    - [ ] Define stable error codes and message payloads
    - [ ] Define ownership, allocation, and release rules
- [ ] Task: Validate Rust interface safety
    - [ ] Add Rust tests for malformed artifacts, unsupported versions, feature mismatches, and NaN/null behavior
    - [ ] Add sanitizer or leak-oriented checks where feasible
    - [ ] Document the ABI/API contract for binding authors
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Stable Rust Foreign Interface' (Protocol in workflow.md)

## Phase 2: Python and Rust Package Integration

- [ ] Task: Wire Python to the Rust runtime/core
    - [ ] Add build configuration for the selected Python binding mechanism
    - [ ] Route runtime replay calls through Rust where available
    - [ ] Preserve sklearn estimator behavior and fallback support
- [ ] Task: Harden the Rust crate as the native binding source
    - [ ] Keep the crate API idiomatic for Rust users
    - [ ] Verify `cargo test`, `cargo package`, and conformance tests
    - [ ] Update Rust package documentation
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Python and Rust Package Integration' (Protocol in workflow.md)

## Phase 3: R, Julia, C#, and Go Bindings

- [ ] Task: Replace R and Julia replay logic with Rust-backed calls
    - [ ] Preserve current public function names
    - [ ] Translate Rust errors into idiomatic host-language errors
    - [ ] Run shared fixtures from each package test entrypoint
- [ ] Task: Replace C# and Go replay logic with Rust-backed calls
    - [ ] Preserve package APIs and type safety
    - [ ] Validate ownership and memory-release rules
    - [ ] Run shared fixtures from each package test entrypoint
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: R, Julia, C#, and Go Bindings' (Protocol in workflow.md)

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
