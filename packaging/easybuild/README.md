# EasyBuild Upstream Notes

This directory contains an EasyBuild easyconfig for `MarsEarth` (HPC identity) and
the `pymars` Python import name.

## Assumptions

- The lane targets source installation on Linux HPC-style systems.
- The package is built from a stable source archive and pinned package metadata.
- Python, NumPy, SciPy, and scikit-learn are available as build dependencies.
- Rust tooling is available during build because the project currently uses a
  Rust core and bridge.

## Current EasyConfig Shape

- `pymars-0.1.0.eb` is the current upstream-ready draft and uses pinned
  release metadata.
- The draft records dependency expectations and the module-install layout.
- No accelerator runtime, MPI layer, or public API change is implied (H0).

## Local Checks

- `eb --check-consistency packaging/easybuild/pymars-0.1.0.eb`
- `eb --dry-run packaging/easybuild/pymars-0.1.0.eb`
