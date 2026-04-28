# Specification: Package Publication Execution

## Overview

The project must publish supported language packages through each ecosystem's
package manager once Rust-backed binding CI, release readiness, registry
ownership, credentials, and maintainer approval are complete. This track is the
execution track for actual publication and post-publish verification.

## Dependency Notes

- Depends on CI/CD quality automation hardening.
- Depends on release readiness and registry governance.
- Stable runtime package publication depends on Rust-backed runtime binding CI
  passing.
- Stable training package publication depends on Rust training orchestration and
  cross-language training API binding tracks where the package claims training
  support.

## Functional Requirements

- Publish approved packages to:
  - Python: PyPI
  - Rust: crates.io
  - TypeScript: npm
  - R: r-universe first, then CRAN when mature
  - Julia: Julia General registry
  - C#: NuGet
  - Go: Go modules through repository tags
- Consume the release-readiness blocker table and halt publication for any
  package whose blockers remain unresolved.
- Require explicit maintainer approval before each publish action.
- Install each published package from its registry and run a smoke fixture.
- Update release notes, package documentation links, and registry status.
- Record any registry that cannot be published with owner/action/date rather
  than silently treating it as complete.

## Non-Functional Requirements

- No package may publish from ordinary pull-request CI.
- Publish jobs must be manual, tag-gated, or protected-environment gated.
- Release jobs must not silently skip failed conformance checks.
- Package contents must already have passed dry-run inspection before release.
- Registry credentials must never be committed to the repository.
- Stable package publication must not use duplicated MVP replay logic unless
  the package is explicitly labeled experimental/pre-release.

## Acceptance Criteria

- Actual publication is completed for registries with available credentials,
  passing release gates, and approval.
- Published artifacts can be installed from their registries and pass smoke
  fixtures.
- Unpublished registries have explicit blocker records with owner/action/date.
- Release notes and package documentation are updated for the published
  versions.

## Out of Scope

- Implementing new model behavior.
- Running first-time release readiness discovery.
- Changing runtime or training semantics.
- Committing registry tokens or bypassing protected release approvals.
