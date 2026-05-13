# Specification: Documentation, Tutorials, and Governance Coordination

## Overview

Coordinate the remaining repo-side work that is still open after the core
runtime, binding, and packaging work.

This umbrella track exists to keep the narrower workstreams aligned:

- documentation and tutorial coverage;
- docs-stack governance; and
- external submission and registration synchronization.

It must not change the runtime contract levels or introduce unsupported HPC
claims.

## Functional Requirements

- Maintain the docs/tutorial track as a separate workstream that covers Python
  walkthroughs and binding usage docs.
- Maintain the docs-stack governance track as a separate workstream that keeps
  mkdocs Material as the live site and Starlight as a governed, non-committal
  evaluation.
- Maintain the submission/registration synchronization track as a separate
  workstream that records the live state for Spack, EasyBuild, conda-forge,
  HPSF/TAC, and Julia General.
- Keep decision points that require user or maintainer action clearly separated
  from repo-side synchronization.
- Update the umbrella Conductor registry so the child tracks are easy to find
  and their dependencies are explicit.

## Non-Functional Requirements

- Preserve the existing project brand and ecosystem-specific package names.
- Keep documentation factual and claim-safe.
- Ensure docs examples are reproducible and avoid unsupported accelerator or
  distributed-execution claims.
- Maintain consistency with the current mkdocs-based site until a migration is
  explicitly approved.
- Keep the coordination shell lightweight so the child tracks can execute in
  parallel without overlapping edits.

## Acceptance Criteria

- The umbrella track points to the narrower child tracks and does not duplicate
  their implementation scope.
- The child tracks cover tutorial content, docs-stack governance, and external
  sync separately.
- The docs describe the current docs-stack state without implying that
  Starlight is live when it is not.
- The open packaging/governance lanes have current URLs, statuses, and follow-up
  notes in the trackers and release docs.
- Any remaining external decision points are explicitly called out rather than
  hidden.
- `./scripts/check_hpc_claims.sh --strict` still passes after the documentation
  and tracker updates.

## Out of Scope

- Implementing H3 accelerator portability.
- Implementing any new runtime kernels or ABI changes.
- Closing external review threads that require maintainer action beyond keeping
  their status current.
- Replacing mkdocs Material with Starlight without an approved migration track.
