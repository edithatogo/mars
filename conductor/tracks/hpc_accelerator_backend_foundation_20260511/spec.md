# H3 Accelerator Backend Foundation

## Overview

Define the shared accelerator contract for H3, including backend selection,
capability detection, optional dependency rules, CPU fallback, and claim
hygiene.

## Functional Requirements

- Define a common backend interface for accelerator replay workloads.
- Define how the runtime selects a backend or falls back to CPU.
- Define what is allowed to be claimed in docs and packaging when a backend is
  unavailable.
- Keep accelerator support optional for CPU-only users.

## Non-Functional Requirements

- Preserve deterministic CPU fallback behavior.
- Keep the backend abstraction minimal and testable.
- Keep accelerator claims blocked until implementation and validation exist.

## Acceptance Criteria

- The shared H3 backend contract is documented and implemented.
- Backend selection and CPU fallback are covered by tests.
- The H3 claim-check rules and docs reflect the implemented contract.

## Out of Scope

- Vendor-specific backend kernels.
- Distributed execution.
- Multi-output regression.

