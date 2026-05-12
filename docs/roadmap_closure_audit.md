# Roadmap Closure Audit

This note records the current implementation closure state after the docs,
examples, accelerator-contract, and strictness work.

## Closed In-Repo

- Core Python runtime and scikit-learn compatibility
- Rust ABI and portable replay boundary
- Shared accelerator backend contract and fallback registry
- Optional GPU-family and specialized accelerator adapters
- Validation scaffolding for accelerator contract coverage
- Canonical Python notebook and binding example hub
- Strict quality-gate policy alignment

## Still Deferred or External

- Real GPU/TPU/FPGA/ASIC kernels
- Multi-node distributed execution
- Open registry review threads and external submissions
- H3/H4 claims beyond the explicit contract and adapter layers
- Explicitly unsupported APIs such as prediction intervals remain documented as
  non-goals rather than hidden gaps

## Audit Rule

Anything not backed by tests, docs, and validation evidence stays marked as
deferred or external until a future track closes it.
