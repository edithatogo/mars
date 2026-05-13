# Specification: conda-forge Feedstock Submission

## Overview

Convert the conda-forge feasibility note into a staged-recipes submission once
the H0 contract in `docs/hpc_contracts.md` is satisfied.

This track owns `packaging/conda-forge/**` and conda-forge recipe drafts. It
must not edit Spack, EasyBuild, or Rust runtime implementation files.

## Functional Requirements

- Create a conda recipe for the published package identity.
- Use `mars-earth` as the distribution/feedstock identity unless conda-forge
  policy requires normalization; keep `pymars` only as the Python import/module
  name.
- Validate source URL, checksums, dependencies, license metadata, and tests.
- Run local recipe lint/build checks where tooling is available.
- Prepare a staged-recipes PR with H0-only claims.
- Record review feedback and feedstock status in release/community docs.
- Run the HPC claim-check gate before submission text is finalized.

## Non-Functional Requirements

- Do not add accelerator or distributed dependencies.
- Keep build/test commands reproducible and token-free.

## Acceptance Criteria

- The recipe is ready for staged-recipes or a blocker is recorded.
- The submission URL, owner, and review status are documented.
- The package claim remains H0 unless higher contracts are implemented.
- No placeholder source URL, checksum, dependency, or test command remains in
  upstream-bound files.

## Out of Scope

- Feedstock bot maintenance after merge unless a later track is created.
