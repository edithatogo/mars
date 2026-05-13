# Specification: Spack Upstream Submission

## Overview

Move the local Spack feasibility sketch to an upstream-ready submission once
the H0 contract in `docs/hpc_contracts.md` is satisfied.

This track owns `packaging/spack/**` and Spack-specific submission notes. It
must not edit EasyBuild, conda-forge, or Rust runtime implementation files.

## Functional Requirements

- Replace placeholder source metadata and checksums in the local Spack recipe.
- Rename stale `pymars` packaging identifiers where Spack policy allows the
  published package identity `mars-earth`; keep `pymars` only as the Python
  import/module name.
- Validate the recipe in a local Spack environment.
- Add install smoke tests that do not require private tokens.
- Prepare an upstream Spack PR with accurate package claims.
- Record upstream review feedback in release/community docs.
- Run the HPC claim-check gate before submission text is finalized.

## Non-Functional Requirements

- Do not claim GPU, MPI, or distributed support unless higher HPC contracts are
  implemented.
- Keep source install behavior aligned with the published Python/Rust package
  state.

## Acceptance Criteria

- The Spack recipe passes local syntax/spec checks.
- The upstream PR or deferred-submission blocker is recorded with owner, URL,
  and date.
- Docs state the package satisfies H0 only unless later tracks complete.
- No placeholder URL, checksum, or version remains in upstream-bound files.

## Out of Scope

- Implementing accelerator or distributed runtime features.
