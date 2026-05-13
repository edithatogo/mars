# HPC Claim Review Checklist

Use this checklist before opening any external HPC-related PR, package submission,
or public release note.

## Disallowed Terms by Contract Level

Before any submission or publish-ready wording is approved, confirm the following
term restrictions:

- `GPU`, `CUDA`, `ROCm`, `Metal`, or `TPU` claims require the project to be at least `H3`.
- `accelerator` claims require `H3` unless explicitly framed as `H3+`-future
  work.
- `MPI`, `multi-node`, or `multi worker` claims require `H4`.
- `distributed` claims require `H4` unless explicitly framed as a `H4` non-goal.

This policy is enforced by `scripts/check_hpc_claims.py` through regex rules and
can be reviewed with:

```bash
uv run python3 scripts/check_hpc_claims.py --strict
```

## Reviewer Checklist by Contract Level

### H0 - HPC-Packaging Ready

- [ ] Confirm submissions only claim source-installability, smoke-testability, and
  package-availability requirements.
- [ ] Confirm claims do not mention `accelerator`, `distributed`, or `GPU` execution in `H0`-scoped text.
- [ ] Confirm package identity and release versions are concrete (no placeholders).
- [ ] Confirm external links and track notes do not imply `H1`+ claims.

### H1 - CPU Throughput Runtime

- [ ] Confirm deterministic single-thread fallback exists.
- [ ] Confirm thread-count controls are resource bounded and documented.
- [ ] Confirm benchmark baselines include single-thread and multi-thread medians plus
  a clear threshold policy.
- [ ] Confirm parallel paths preserve serial behavior and error semantics.
- [ ] Confirm serial-vs-parallel fixture parity tests are present.

### H2 - Stable Runtime Boundary

- [ ] Confirm runtime-boundary documentation distinguishes API surfaces and
  stability expectations.
- [ ] Confirm memory ownership, error signaling, and versioning contract are
  explicit.
- [ ] Confirm Arrow/ABI interoperability claims are supported by docs and tests.
- [ ] Confirm at least one non-Python binding path has boundary-facing tests.

### H3 - Accelerator-Ready Runtime

- [ ] Confirm the `H3` accelerated backend is optional, not mandatory.
- [ ] Confirm CPU fallback remains functional and default-capable.
- [ ] Confirm `H3` capability checks and unsupported feature behavior are documented.
- [ ] Confirm numerical tolerances are explicit for backend parity in the `H3` scope.
- [ ] Confirm no `H3` accelerated package creates hard installation requirements.
- [ ] If H3 is deferred, confirm docs explicitly state:
  - CPU replay is the only supported path in this release set.
  - `H3` accelerator-ready execution is intentionally not yet implemented.
  - wording uses `not yet` / `non-goal` / `deferred` and links to a non-claim checkpoint.

### H4 - Distributed Execution

- [ ] Confirm partitioning, ordering, and deterministic aggregation are specified.
- [ ] Confirm failure and retries semantics are documented.
- [ ] Confirm `H4` cluster smoke/aggregation evidence is present.
- [ ] Confirm no implicit cluster start-up in default import/runtime behavior for `H4`.

## Reviewer Sign-off

For an external-ready submission, record:

- supported contract levels,
- exact evidence files referenced,
- any blocked claims explicitly preserved as non-goals or deferred work.
