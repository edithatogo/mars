# EasyBuild Submission Draft (H0-only)

Prepared for: external EasyBuild easyconfig review workflow.

- Contract level stated in draft: `H0` only.
- do not claim accelerator, MPI, or distributed execution in submission text.
- Evidence files:
  - `packaging/easybuild/pymars-0.1.0.eb`
  - `packaging/easybuild/README.md`
  - `docs/hpc_track_checkpoint_notes.md`

Draft PR body:

```
EasyBuild recipe: pymars-0.1.0 (MarsEarth runtime)

This draft provides an H0-only source-installable easyconfig for Spack-style/conda/HPC
environments. The package keeps Python runtime compatibility and Rust-backed core packaging
constraints explicit.

There is currently no mandatory accelerator dependency and no distributed execution
runtime claim. The submission targets source installability and packaging integrity.
```

Outstanding blocker:
- External upstream PR creation in EasyBuild infrastructure requires access to the
  review workflow outside this workspace.
