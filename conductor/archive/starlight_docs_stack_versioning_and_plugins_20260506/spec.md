# Starlight Docs Stack Versioning and Plugin Governance

## Overview

This track defines a version-pinned, plugin-complete Starlight documentation
stack for `mars`. The goal is to evaluate the Starlight core version and the
supporting plugin set needed to preserve the current documentation experience,
then record a clear migration or coexistence path from the current mkdocs site
without changing the public docs surface prematurely.

## Goals

- Inventory the documentation capabilities that must be preserved during a
  potential Starlight migration.
- Identify the Starlight version target and the required plugin set.
- Record version pinning, upgrade policy, and compatibility constraints.
- Define validation and CI expectations for any Starlight-based docs stack.
- Keep the current mkdocs documentation site stable until a Starlight change is
  explicitly approved.

## Functional Requirements

1. Map the current docs site features that matter for users and maintainers,
   including navigation, search, code blocks, admonitions, release pages, track
   pages, and content organization.
2. Identify Starlight core version candidates and the supporting plugins needed
   to match or improve the current docs behavior.
3. Document which plugins are required, optional, or out of scope, and why.
4. Define how Starlight version updates and plugin updates should be reviewed,
   pinned, and released.
5. Specify the CI checks and local validation commands needed to keep the docs
   stack reproducible.
6. Capture a fallback or rollback path so the current mkdocs site remains the
   source of truth until migration is approved.

## Non-Goals

- Rewriting docs content for a new information architecture.
- Switching the live documentation site to Starlight before the version and
  plugin policy is approved.
- Changing public docs URLs or navigation solely for aesthetics.
- Unrelated application or package code changes.

## Acceptance Criteria

- A Starlight version and plugin matrix is documented in Conductor and the
  roadmap.
- The docs stack decision includes a clear upgrade and rollback policy.
- The validation strategy for the Starlight docs stack is recorded.
- The track is linked from the Conductor registry and the remaining roadmap.
- The current mkdocs site remains intact unless a later phase explicitly
  approves migration.
