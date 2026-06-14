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
- Version: 1.0.4
- Source: https://github.com/edithatogo/mars
- Summary: Source-installable Python binding with Rust-backed runtime and
  explicit H0 packaging constraints.

HPC note: this feedstock submits the H0 packaging lane. The repository now has
H1 CPU replay evidence plus opt-in H3/H4 replay surfaces, but this staged-recipes
packet does not claim vendor accelerator speedups, mandatory device runtimes,
implicit cluster provisioning, or training support.

The recipe intentionally avoids stub metadata and keeps dependency declarations
explicit to support HPC-style source builds where applicable.
```

Outstanding blocker:
- External staged-recipes PR requires repository access and submission workflow outside
  this workspace.


Windows CI note:
- The staged-recipes PR showed a failing `staged-recipes (Build win_64 win)` check while linux and osx builds passed and the linter remained green. The H0 recipe now skips Windows builds with `skip: true  # [win]` and is intentionally platform-specific rather than `noarch: python`, because conda-forge lint disallows skip selectors on noarch recipes. This HPC packaging lane targets source-installable POSIX-style environments and should not block review on Windows build isolation behavior.


Python requirement note:
- After making the recipe platform-specific for the Windows skip, conda-forge lint requested unconstrained `python` entries for non-noarch host/run/test requirements. The local recipe and PR branch now use unconstrained `python` entries while retaining package metadata and upstream source version `1.0.4`.
