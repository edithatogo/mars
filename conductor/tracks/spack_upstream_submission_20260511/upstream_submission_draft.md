# Spack Submission Draft (H0-only)

Prepared for: Spack package repository upstream pull request.

- Contract level stated in draft: `H0` only.
- do not claim accelerator, MPI, or distributed execution in submission text.
- Evidence files:
  - `packaging/spack/package.py`
  - `packaging/spack/README.md`
  - `docs/hpc_track_checkpoint_notes.md`

Draft PR body:

```
Spack package: mars-earth 0.1.0

This PR proposes an H0-only Spack package recipe for the mars-earth source release.
It is built around a Rust-backed runtime with an explicit source checksum and
dependency policy suitable for HPC source builds.

do not claim accelerator backend, MPI, or distributed execution support for this package.
```

Outstanding blocker:
- External Spack upstream PR creation and final merge depends on Spack maintainers'
  submission workflow outside this workspace.
