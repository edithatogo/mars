# Specification: HPC Contract Governance

## Overview

Create and maintain the contract layer that defines what the HPC version of
`mars-earth` means before implementation or submission tracks claim HPC
readiness.

## Functional Requirements

- Keep `docs/hpc_contracts.md` as the source of truth for HPC contract levels.
- Map each HPC implementation or submission track to the minimum contract level
  it must satisfy.
- Add checks or documentation review steps that prevent unsupported GPU, MPI,
  accelerator, or distributed claims.
- Add a claim-check script or documented command that can be run before
  implementation, packaging, or submission phases are completed.
- Maintain the implementation dependency graph and parallel subagent ownership
  table for all HPC tracks.
- Keep the contract aligned with release inventory, package paths, and community
  submission readiness docs.

## Non-Functional Requirements

- Preserve the current public Python and binding APIs.
- Keep the contract additive and versioned.
- Avoid implying that feasibility artifacts are implemented compute features.

## Acceptance Criteria

- The contract document defines H0-H4 levels and current state.
- Every active HPC implementation/submission track references the contract.
- Every active HPC implementation/submission track has dependency and ownership
  boundaries suitable for parallel subagents.
- The claim-check command flags unsupported HPC claims.
- Release/community docs distinguish packaging readiness from HPC compute
  readiness.

## Out of Scope

- Implementing kernels, ABI, packaging submissions, or foundation submissions.
