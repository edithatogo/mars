# Conda-forge Submission Draft (H0-only)

Prepared for: staged-recipes external submission or maintainer handoff.

- Contract level stated in draft: `H0` only.
- do not claim accelerator, MPI, or distributed execution in submission text.
- Evidence files:
  - `packaging/conda-forge/recipe/meta.yaml`
  - `packaging/conda-forge/README.md`
  - `docs/hpc_track_checkpoint_notes.md`
  - `docs/hpc_cpu_parallel_runtime_benchmarks.md` (for context; no compute contract
    claims in this packet)

Draft PR body:

```
### conda-forge feedstock for mars-earth

- Package: mars-earth
- Version: 0.1.0
- Source: https://github.com/edithatogo/mars
- Summary: Source-installable Python binding with Rust-backed runtime and
  explicit H0 packaging constraints.

HPC note: this feedstock only submits the H0 packaging lane. The project has CPU
replay runtime benchmark work in progress but does not claim H2/H3/H4 behavior.

The recipe intentionally avoids stub metadata and keeps dependency declarations
explicit to support HPC-style source builds where applicable.
```

Outstanding blocker:
- External staged-recipes PR requires repository access and submission workflow outside
  this workspace.
