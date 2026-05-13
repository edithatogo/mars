# Canonical Versioning and Release Metadata Governance

## Overview

This track defines a single canonical source of truth for release metadata,
package version policy, and cross-language version parity rules for the
`mars-earth` package family. The goal is to keep package manifests, release
docs, and CI validation aligned without forcing artificial version coupling
where ecosystems require independent versions.

## Goals

- Define a canonical release metadata source for the repo.
- Document which version fields are authoritative for each package.
- Preserve intentional version skew where an ecosystem requires it.
- Generate or validate release inventory and handoff docs from the canonical
  metadata.
- Enforce package-manifest and release-doc alignment in CI.

## Functional Requirements

1. Capture the package family version policy in one canonical metadata file.
2. Describe the repository-wide versioning model for Python, Rust, npm, NuGet,
   Go, R, and Julia.
3. Preserve the current `mars-earth` brand while allowing ecosystem-native
   package names where required.
4. Keep release inventory, publication handoff, and release checklist content
   synchronized with the canonical metadata.
5. Extend the existing alignment check so drift in version sources or package
   identities is detected early.
6. Record the logging and runtime/core parity contract alongside versioning so
   release docs stay SOTA and coherent.

## Non-Goals

- Forcing a single numeric version across every ecosystem.
- Changing public package names that are already ecosystem-native.
- Changing runtime behavior or public APIs.
- Replacing the current release workflows with a new publication system.

## Acceptance Criteria

- A canonical release-metadata source exists in the repo.
- The metadata source can drive or validate the release inventory and handoff
  docs.
- CI checks verify manifests, docs, and release metadata stay aligned.
- The repo documents when version skew is intentional and acceptable.
- The release docs continue to show the current live registry state accurately.
