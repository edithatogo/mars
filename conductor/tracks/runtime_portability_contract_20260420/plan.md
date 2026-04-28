# Implementation Plan

## Phase 0: Current State Inventory

- [x] Task: Inventory the existing serialized model and runtime assumptions
    - [x] Review current `ModelSpec` shape and runtime entry points
    - [x] Identify Python-specific assumptions that would block foreign-language consumers
    - [x] Identify existing version fields, compatibility assumptions, and migration gaps
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Current State Inventory' (Protocol in workflow.md)

## Phase 1: Versioning and Compatibility Rules

- [x] Task: Define schema migration and compatibility policy
    - [x] Specify semantic versioning expectations for portable model artifacts
    - [x] Define required-versus-optional fields and forward/backward compatibility rules
    - [x] Define failure behavior for missing, unknown, or incompatible versions
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Versioning and Compatibility Rules' (Protocol in workflow.md)

## Phase 2: Runtime Boundary Definition

- [x] Task: Define the stable runtime surface for portable evaluation
    - [x] Specify the language-neutral evaluation API around `ModelSpec` and basis-matrix evaluation
    - [x] Specify data-shape and dtype expectations for runtime consumers
    - [x] Define the C-ABI-neutral or embedded boundary constraints
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Runtime Boundary Definition' (Protocol in workflow.md)

## Phase 3: Packaging Decision and Publishable Contract

- [x] Task: Decide runtime packaging and publish the contract
    - [x] Evaluate whether the runtime remains in `pymars` or moves to a thin split package
    - [x] Document the decision and its maintenance implications
    - [x] Produce a concise portability contract for downstream implementers
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Packaging Decision and Publishable Contract' (Protocol in workflow.md)
