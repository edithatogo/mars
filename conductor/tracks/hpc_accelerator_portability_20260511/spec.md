# Specification: H3 Accelerator Portability

## Overview

Implement the H3 contract from `docs/hpc_contracts.md`: optional accelerator
execution for replay workloads with safe CPU fallback and documented numerical
tolerance.

This track depends on H1 benchmark data. Backend selection must be based on the
observed replay kernel shape, packaging constraints, and CPU fallback design.

## Functional Requirements

- Choose an accelerator backend strategy only after reviewing H1 benchmark data
  and packaging constraints; document the first supported device/runtime target.
- Add device discovery and safe fallback behavior.
- Add replay parity tests against CPU fixtures within documented tolerances.
- Add benchmarks comparing CPU and accelerator paths.
- Document unsupported basis terms, data layouts, and device capabilities.

## Non-Functional Requirements

- Accelerator dependencies must be optional.
- Users without accelerator runtimes must still be able to install and use CPU
  packages.
- Importing the package must not initialize devices or spawn hidden workers.
- Accelerator support must be feature-gated or otherwise optional.

## Non-Claim Deferral Policy

- This track is not yet implemented for accelerator execution in the current revision
  set. Replay remains CPU-only by default.
- Maintainable deferral statement:
  - "`H3` accelerator-ready execution is not yet implemented; this track is deferred."
- This is a safe release wording because it combines:
  - an explicit contract name (`H3`) and
  - an explicit present-tense state (`not yet implemented`).
- Any release-facing text that references the track while deferred must include one of:
  - `not yet implemented`
  - `non-goal`
  - `not currently in scope`
  - `planned for a later phase`

## Acceptance Criteria

- H3 requirements in `docs/hpc_contracts.md` are satisfied for at least one
  supported accelerator path.
- CPU fallback and no-device environments are tested.
- Docs clearly state supported and unsupported accelerator behavior.
- H3 docs pass the claim-check gate.

## Track Non-claim Checkpoints

- A maintained non-claim checkpoint is required until implementation starts.
- The non-claim checkpoint must record:
  - current supported runtime (`CPU replay only`)
  - explicit deferral rationale (e.g., backend selection pending H1 evidence)
  - the file path that documents deferral
  - whether optional accelerator support is still blocked by default install/test
    constraints
- The non-claim checkpoint is required before marking any release or
  submission documentation as complete for H3.

## Out of Scope

- Distributed execution.
- Mandatory CUDA, ROCm, Metal, TPU, or vendor-specific dependencies.
