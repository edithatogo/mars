# Specification: SOTA HPC, ABI, and Parallelism Roadmap

## Overview

The Rust core already improves portability and correctness, but the project is
not yet positioned as a high-performance scientific computing platform. This
track defines the HPC roadmap, decides whether a narrow ABI is justified, and
sets the non-breaking path toward better performance, accelerator readiness,
and HPC ecosystem fit.

## Dependency Notes

- Depends on the current Rust core ownership boundary and the existing
  binding-backend contract.
- Must preserve the current Python API and the language-specific wrappers.
- Should build on the existing benchmark and observability work instead of
  replacing it.

## Functional Requirements

- Inventory the current HPC and parallelism gaps:
  - CPU profiling and benchmarking
  - memory profiling and allocation visibility
  - GPU readiness
  - TPU readiness
  - distributed / multi-node execution
  - performance-portability tooling
- Decide whether a stable ABI is relevant and, if so, what scope it should
  cover.
- Describe how an ABI could be introduced without breaking the existing API.
- Define the future-state architecture for runtime evaluation, profiling,
  bindings, and accelerator-ready execution.
- Produce a staged roadmap for near-term, mid-term, and longer-term HPC work.
- Document how the project would align with HPSF and E4S expectations.

## Non-Functional Requirements

- No breaking change to `import pymars as earth` or `earth.Earth(...)`.
- Preserve the current host-language binding model unless a narrow ABI layer is
  explicitly justified.
- Keep the roadmap additive and versioned rather than speculative.

## Acceptance Criteria

- The HPC gap inventory exists and is clearly tied to current repo state.
- The ABI recommendation is explicit and states how the public API remains
  stable.
- The roadmap distinguishes near-term profiling work from long-term
  accelerator/distributed work.
- HPSF and E4S readiness criteria are documented.
- The docs include current-state and future-state architecture diagrams.

## Out of Scope

- Implementing GPU kernels, TPU kernels, or distributed runtime support.
- Breaking the current Python API.
- Replacing the current Rust core with a different runtime.
- Making HPSF or E4S applications as part of this track.
