# Specification: Docs-Stack Governance Alignment

## Overview

Keep the docs-stack governance state explicit and accurate.

The current live docs site remains mkdocs Material. Starlight remains a
governed option, not a live migration.

## Functional Requirements

- Keep the docs-stack governance page accurate and non-committal.
- Preserve the distinction between the live docs site and any future migration
  evaluation.
- Update navigation and cross-links so the docs map points to the governance
  pages, tutorial pages, and relevant release/HPC docs.
- Keep any future migration requirements documented before adoption.

## Non-Functional Requirements

- Preserve the current mkdocs-based site until a migration is explicitly
  approved.
- Keep the governance docs factual and claim-safe.

## Acceptance Criteria

- The docs describe the current docs-stack state without implying that
  Starlight is live when it is not.
- The main docs navigation points to the governance pages and their child
  content.
- `./scripts/check_hpc_claims.sh --strict` still passes after the updates.

## Out of Scope

- Building or migrating to Starlight.
- Tutorial content expansion.
- External submission or registration synchronization.
