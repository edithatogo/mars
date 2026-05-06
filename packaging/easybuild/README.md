# EasyBuild Feasibility Notes

This directory contains a proof-of-concept EasyBuild easyconfig sketch for
`pymars`.

## Assumptions

- The lane targets source installation on Linux HPC-style systems.
- The package is built from a stable source archive or release tag.
- Python, NumPy, SciPy, and scikit-learn are available as build dependencies.
- Rust tooling is available during build because the project currently uses a
  Rust core and bridge.

## Current EasyConfig Shape

- `pymars-0.1.0.eb` is a feasibility sketch, not a submission artifact.
- The sketch records dependency expectations and the module-install layout.
- No accelerator runtime, MPI layer, or public API change is implied.

## Local Checks

- `eb --check-consistency packaging/easybuild/pymars-0.1.0.eb`
- `eb --dry-run packaging/easybuild/pymars-0.1.0.eb`
