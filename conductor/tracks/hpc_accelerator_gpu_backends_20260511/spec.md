# H3 GPU Backends

## Overview

Implement optional GPU-family accelerator backends for replay workloads,
including CUDA, ROCm, Metal, and equivalent APIs where the H3 contract can be
honored.

## Functional Requirements

- Add at least one GPU-family backend implementation.
- Preserve CPU fallback for unsupported devices or kernels.
- Keep the GPU backend behind the shared H3 interface.
- Add GPU-specific parity and tolerance tests.

## Non-Functional Requirements

- The GPU backend must be optional.
- CPU-only users must not be forced to install GPU dependencies.
- Benchmarking must make CPU and GPU paths comparable.

## Acceptance Criteria

- A GPU-family backend can run supported replay workloads.
- Fallback behavior is tested and documented.
- The docs and claim-checks match the implementation.

## Out of Scope

- TPU, FPGA, and ASIC backends.
- Distributed execution.
- Backend-specific performance tuning beyond the first supported path.

