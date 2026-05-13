# Spack Upstream-Ready Notes

This directory contains a Spack recipe for `mars-earth`, aligned to the H0 HPC
packaging lane.
It is intentionally isolated from runtime code and shared registry release
docs.

## Assumptions

- Source installation is the target (H0), not a binary accelerator runtime.
- The package is built from a stable source archive and pinned metadata.
- Python, NumPy, SciPy, scikit-learn, Rust, and Cargo are available as Spack
  dependencies.
- Rust tooling is available during build because the project uses a Rust-backed
  runtime bridge.

## Current Recipe Shape

- `package.py` tracks `mars-earth` package identity and pinned source metadata.
- The H0 recipe keeps the dependency policy explicit and is free of accelerator
  runtime assumptions.
- The recipe should remain free of accelerator runtimes and API changes (H0).

## Local Checks

- `python3 -m py_compile packaging/spack/package.py`
- `spack spec mars-earth@1.0.4` once a local Spack environment is available
- `spack install --test=root mars-earth@1.0.4` once the recipe is
  wire-checked with a real cache
