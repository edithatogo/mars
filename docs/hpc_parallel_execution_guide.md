# HPC Parallel Execution Guide

This guide converts the HPC contracts into a parallel implementation plan for
subagents. It is operational guidance; `docs/hpc_contracts.md` remains the
contract source of truth.

## Recommended Start Order

1. `hpc_contract_governance_20260511`
2. H0 packaging cleanup and validation tracks:
   - `spack_upstream_submission_20260511`
   - `easybuild_upstream_submission_20260511`
   - `conda_forge_feedstock_submission_20260511`
3. `hpc_cpu_parallel_runtime_20260511`
4. `hpc_abi_arrow_runtime_boundary_20260511`
5. `hpc_accelerator_portability_20260511`
6. `hpc_distributed_execution_20260511` (CPU cluster parallelism and partitioned replay)
7. `hpsf_e4s_readiness_submission_20260511`
8. `julia_general_registration_submission_20260511`

H0 packaging tracks can run in parallel with H1 benchmark-baseline work. H2,
H3, and H4 should not start implementation until H1 has produced stable
semantics and benchmark evidence.

## Parallel Subagent Slices

| Agent | Track | Write scope | Blocking dependencies |
| --- | --- | --- | --- |
| A | Governance + claims | `docs/hpc_contracts.md`, claim-check script/docs, release/community references | None |
| B | H0 packaging lane | `packaging/spack/**`, `packaging/easybuild/**`, `packaging/conda-forge/**`, submission notes | Contract governance baseline |
| C | H1 runtime lane | Rust runtime kernels, Rust/Python benchmark and parity tests | Contract governance baseline |
| D | H2 ABI lane | ABI/boundary files and tests, interoperability notes | H1 semantics and benchmark thresholds |
| E | H3 accelerator lane | Optional backend prototype and docs/tests | H1 benchmark data |
| F | H4 distributed lane | CPU cluster parallelism and distributed adapter files and tests | H1 partitioning semantics; H3 only if accelerator-distributed |

## Parallel Team Assignment (6 Subagents)

The request for six parallel workers maps cleanly to one-lane ownership:

- **Agent A**: Governance lane (`hpc_contract_governance_20260511`) and claim-check
  enforcement.
- **Agent B**: H0 packaging lanes (`spack`, `easybuild`, `conda-forge`) and
  any blockers from external tooling.
- **Agent C**: H1 runtime optimization, benchmark evidence, and serial/parity tests.
- **Agent D**: H2 boundary hardening, versioning contract, and boundary conformance.
- **Agent E**: H3 accelerator feasibility, optional dependency/feature-gate design, and safety policy.
- **Agent F**: H4 CPU cluster parallelism, distributed partitioning, and deterministic aggregation work.

Notes:

- HPSF/E4S and Julia registration tracks are still handled as governance-facing
  follow-ups after the primary lanes above have clear checkpoint evidence.
- The H4 lane currently refers to the replay-only local preview adapter; it is
  fail-fast on invalid inputs and does not imply retry or cluster orchestration
  semantics. CPU cluster parallelism remains the explicit cluster-oriented
  target for this lane.

## Coordination Rules

- Each agent must state its write scope before editing.
- Agents must not revert or rewrite files owned by another lane.
- Cross-lane API changes must be proposed in the owning track before edits.
- H0 agents must remove non-final URLs, versions, checksums, and stale
  package identities before upstream-bound files are considered complete.
- H1 agents must define benchmark thresholds before changing kernel behavior.
- H3 agents must not choose an accelerator backend until H1 benchmark data is
  available.
- HPSF/E4S agents must not submit full packets unless H0 is complete and H1/H2
  evidence is credible; otherwise they should draft pre-submission inquiries or
  deferral notes.
- Julia registry agents should keep their artifacts review-only and avoid touching
  runtime/kernel files while preparing Registrator materials.

## Validation Commands

Baseline repository checks:

```bash
jq . docs/release_metadata.json
rg -n "HPC-ready|HPSF|E4S" docs conductor packaging
./scripts/check_hpc_claims.sh
cat docs/hpc_claim_review_checklist.md
```

Track artifact checks:

```bash
for d in conductor/tracks/hpc_*_20260511 conductor/tracks/*submission_20260511; do
  test -f "$d/spec.md"
  test -f "$d/plan.md"
  test -f "$d/metadata.json"
  test -f "$d/index.md"
  jq . "$d/metadata.json" >/dev/null
done
```

Benchmark evidence checks:

```bash
cd rust-runtime
cargo bench --bench runtime_bench -- --noplot
python3 scripts/benchmark_runtime_threads.py --help
```
