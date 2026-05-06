# Spack Feasibility Notes

This directory contains a proof-of-concept Spack recipe sketch for `pymars`.
It is intentionally isolated from runtime code and shared registry release
docs.

## Assumptions

- Source installation is the target, not a binary accelerator runtime.
- The package is built from a stable source archive or release tag.
- Python, NumPy, SciPy, and scikit-learn are available as Spack dependencies.
- Rust tooling is available during build because the current project uses a
  Rust core and a Rust-backed runtime bridge.

## Current Recipe Shape

- `package.py` is a feasibility sketch, not an upstream Spack submission.
- The recipe keeps the dependency policy explicit and leaves actual source URL
  wiring to future release automation work.
- The recipe should remain free of accelerator runtimes and API changes.

## Local Checks

- `python3 -m py_compile packaging/spack/package.py`
- `spack spec pymars` once a local Spack environment is available
- `spack install --test=root pymars` once the sketch is wired to a real source
  archive
