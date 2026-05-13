# Specification: Julia General Registration Submission

## Overview

Prepare and submit the `MarsEarth` Julia package to Julia General now that the
`MarsRuntime` identity is treated as a superseded legacy package.

This track must not change runtime implementation; it is registry-submission
governance only and aligns with [HPC contract constraints](../../docs/hpc_contracts.md)
that limit external claims to H0 unless compute contracts are implemented.

## Functional Requirements

- Register a new Julia package identity named `MarsEarth`.
- Keep the current package UUID and source layout in `bindings/julia` as-is for the
  first submission attempt.
- Record registry submission ownership and PR/status details for follow-up.
- Confirm package README, license, dependency policy, and source URL in the final
  registry entry.
- Confirm no duplicate identity confusion remains in release/docs with `MarsRuntime`.

## Non-Functional Requirements

- Registration submission should include clear ownership/contact information.
- Registration should not claim H2-H4 runtime capabilities until evidence exists.
- The track must record any blocker where registrator tooling or maintainer review
  is unavailable.

## Acceptance Criteria

- `MarsEarth` registration artifacts (PR draft, registry link, maintainer notes)
  are attached to the release/blocker docs.
- Blocker state (URL, owner, date, next action) is recorded if submission cannot
  be completed in this environment.
- H1-H4 contract text in downstream tracks is unaffected by this submission.

## Out of Scope

- H2/H3/H4 implementation.
- Publishing or re-architecting Rust runtime internals from this track.
