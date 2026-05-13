# Specification: EasyBuild Upstream Submission

## Overview

Move the local EasyBuild feasibility sketch to an upstream-ready easyconfig
once the H0 contract in `docs/hpc_contracts.md` is satisfied.

This track owns `packaging/easybuild/**` and EasyBuild-specific submission
notes. It must not edit Spack, conda-forge, or Rust runtime implementation
files.

## Functional Requirements

- Replace placeholder source metadata and module assumptions.
- Rename stale `pymars` packaging identifiers where EasyBuild policy allows the
  published package identity `mars-earth`; keep `pymars` only as the Python
  import/module name.
- Validate the easyconfig with EasyBuild consistency and dry-run checks.
- Add or document smoke tests for installed Python and Rust runtime behavior.
- Prepare an upstream EasyBuild PR with accurate H0 claims.
- Record upstream review feedback in release/community docs.
- Run the HPC claim-check gate before submission text is finalized.

## Non-Functional Requirements

- Do not imply accelerator, MPI, or distributed support.
- Keep the easyconfig free of private registry credentials.

## Acceptance Criteria

- The easyconfig passes local checks where tooling is available.
- Upstream PR or blocker state is recorded with owner, URL, and date.
- Docs state H0 packaging readiness only.
- No placeholder source, checksum, or module assumption remains in
  upstream-bound files.

## Out of Scope

- Implementing runtime acceleration or scheduler integration.
