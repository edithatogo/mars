# Specification: Rust Migration and ABI Compatibility

## Overview

The repository has already moved a large part of the portable runtime into Rust,
but the migration story still needs a narrow, explicit compatibility track. This
track defines the remaining migration boundary, the ABI posture, and the
non-breaking path for future Rust-first work.

## Dependency Notes

- Depends on the current Rust core ownership boundary and the binding backend
  decisions.
- Must preserve the current Python API and the current language-specific
  bindings.
- Should align with the HPC/ABI roadmap without duplicating it.

## Functional Requirements

- Inventory the remaining Rust migration and compatibility concerns:
  - adapter boundaries
  - fallback policy
  - portable `ModelSpec` ownership
  - host-language bridge behavior
- Decide whether a narrow ABI should be introduced or whether the current
  bridge model remains sufficient.
- Define how to keep future Rust migration work API-compatible.
- Define what documentation, tests, and acceptance criteria are required before
  any new Rust migration slice can land.

## Non-Functional Requirements

- No breaking API changes.
- Keep `import pymars as earth` and `earth.Earth(...)` intact.
- Avoid duplicating the performance-oriented roadmap.

## Acceptance Criteria

- The migration boundary and ABI recommendation are explicit.
- The repo documentation names the remaining Rust migration concerns.
- The track defines how future migration slices stay non-breaking.
- The roadmap references the migration track alongside the HPC/ABI roadmap.

## Out of Scope

- Implementing new accelerator features.
- Changing the public package names.
- Replacing the current binding strategy wholesale.
