# HPSF and E4S Packet Drafts (2026-05-11)

This document contains draft readiness packets for community submission lanes that are
ready for maintainer review. Both packets are framed as **H0-first** packets and
do not claim accelerator, MPI, multi-node, or distributed execution capabilities.

Submission state as of `2026-06-14`:
- the HPSF readiness inquiry has been submitted to the HPSF TAC issue tracker
  at `https://github.com/hpsfoundation/tac/issues/88`
- the E4S packet content remains in the repo as a draft while the same inquiry
  is used for packaging-readiness review
- the current status is `submitted for TAC review` rather than `draft`
- any full packet submission remains deferred until TAC feedback and explicit
  maintainer approval
- the current blocker record lives in `docs/release_inventory.md`

Maintainer contact details used in outbound packets should be rendered as:
`dylan dot mordaunt [at] vuw dot ac dot nz`.

## HPSF Readiness Packet (Draft)

- Packet scope: packaging, portability evidence, and baseline runtime readiness for
  HPC environments.
- Current contract level: `H0` implemented, `H1` measured for replay behavior,
  `H2` implemented in-repo but not claimed in this packet; `H3/H4` remain not yet
  implemented.
- Claim hygiene: do not include `accelerator`, `MPI`, or `distributed` language
  beyond explicit "not currently implemented" framing.
- Evidence included:
  - `docs/hpc_track_checkpoint_notes.md`
  - `docs/hpc_contracts.md`
  - `docs/hpc_parallel_execution_guide.md`
  - `docs/hpc_cpu_parallel_runtime_benchmarks.md`
  - `docs/release_inventory.md`
  - `packaging/spack/package.py`
  - `packaging/easybuild/pymars-0.1.0.eb`
  - `packaging/conda-forge/recipe/meta.yaml`

### Draft text

```
Subject: HPSF readiness inquiry: mars-earth (HPC contract H0/H1 current)

Project: mars-earth / pymars
Package identity: Python package mars-earth, runtime import as pymars / earth

Status:
- H0 packaging for Spack, EasyBuild, and conda-forge is prepared with concrete
  source/checksum/test metadata.
- H1 replay performance baseline exists for CPU multi-threaded runtime paths.
  - H2 boundary evidence exists in-repo but is not claimed here.
  - H3/H4 boundaries are currently not implemented.
- do not claim accelerator, MPI, or distributed execution support at this stage.

I am requesting HPSF guidance on whether this evidence and packaging shape are
sufficient for initial inclusion in the HPC ecosystem guidance set, or whether
additional packaging constraints should be addressed before full packaging publication.
```

## E4S Readiness Packet (Draft)

- Packet scope: installation and build readiness for E4S-like HPC stacks.
- Current contract level: `H0` implemented with explicit deferral of higher-level claims.
- Claim hygiene: do not claim accelerator, MPI, or distributed behavior yet.
- Evidence included:
  - `docs/release_inventory.md`
  - `docs/package_release_paths.md`
  - `packaging/spack/README.md`
  - `packaging/easybuild/README.md`
  - `packaging/conda-forge/README.md`
  - `docs/hpc_cpu_parallel_runtime_benchmarks.md`

### Draft text

```
Subject: E4S packaging readiness packet: marsearth/mars-earth (HPC compute: H0/H1)

Project name: mars-earth (Python runtime package identity; H0/H1 package/install readiness)

Prepared evidence:
- Spack recipe, EasyBuild easyconfig, and conda-forge staged-recipes draft all target
  source installability and H0 constraints.
- CPU replay benchmarking and thread controls are implemented and documented for
  runtime throughput work.
- do not claim accelerator backend, MPI, or distributed execution support at this
  stage.

This packet is submitted as readiness+feedback-oriented, scoped to H0/H1 artifacts.
Please advise any required packaging policy adjustments for a future
production-level E4S entry.
```

## Blocking statements to keep this packet compliant

- `H2` stable runtime boundary is present in the repository but intentionally
  excluded from this packet.
- `H3` accelerator-ready execution is not yet in place.
- `H4` distributed execution is not yet in place.
- The project remains explicitly non-goal for accelerator and distributed execution
  until dedicated tracks land.
