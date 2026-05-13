# Specification: Release Readiness and Registry Governance

## Overview

Publication should be separated from release readiness. This track prepares the
repository, CI, registry ownership, credentials, package metadata, dry-run
artifacts, and approval process without publishing stable packages.

The result should be a release candidate process maintainers can inspect and
approve before any package is pushed to an external registry.

## Dependency Notes

- Depends on CI/CD quality automation hardening.
- Stable release readiness depends on Rust-backed runtime binding CI passing.
- Can run before full Rust training orchestration, but stable package release
  must state whether the release is runtime-only or includes training APIs.

## Functional Requirements

- Confirm package names, registry ownership, maintainers, and release channels
  for PyPI, crates.io, npm, r-universe/CRAN, Julia General, NuGet, and Go
  modules.
- Confirm package contents through dry-run or local pack commands for each
  ecosystem.
- Confirm version reporting, changelog, and package metadata parity across the
  language surfaces.
- Confirm logging defaults, verbosity toggles, and error-surfacing conventions
  are documented for each binding.
- Add an explicit external blocker table with owner, action, date, and status.
- Define protected release environments, required secrets, and trusted
  publishing options where available.
- Define a no-stable-publish rule while duplicated MVP replay logic remains in a
  package, unless the package is explicitly marked experimental/pre-release.
- Define release notes, changelog, version-bump, rollback/yank/deprecation, and
  smoke-test policy.

## Non-Functional Requirements

- Registry credentials must not be committed.
- Dry-run artifacts must be inspectable before publication.
- Release readiness should fail closed when required metadata, tests, or
  credentials are missing.
- Maintainers must be able to run a rehearsal without publishing.

## Acceptance Criteria

- Every package manager has documented registry ownership and credential status.
- Every package has a successful build/dry-run artifact or a documented blocker.
- CI can produce inspectable release candidate artifacts.
- A release checklist exists with explicit approval and rollback steps.
- The publication track can consume this track's outputs without rediscovering
  registry state.

## Out of Scope

- Actual package publication.
- Implementing Rust-backed bindings or training APIs.
- Changing model semantics to satisfy registry checks.
