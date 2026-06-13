# Implementation Plan

## Phase 0: Backend Selection and Deferral Hygiene

- [x] Task: Record and hold H3 deferral contract
    - [x] Add explicit H3 deferral wording to checkpoint evidence:
    - current supported runtime at Phase 0: CPU replay only
    - accelerator backend status at Phase 0: intentionally not yet implemented
    - gating rationale: H1 kernel shape and packaging tradeoff review pending
  - [x] Confirm checkpoint language appears in release-facing docs or packet drafts
    with explicit `not yet` / `non-goal` / `deferred` phrasing
  - [x] Update `docs/hpc_claim_review_checklist.md` H3 non-claim verification items if needed
    before any H3 submission claim is considered.

- [x] Task: Select the initial accelerator backend [adapter-only selection; no vendor kernel claimed]
    - [x] Review H1 benchmark data and kernel shape
    - [x] Compare candidate backends against portability and packaging constraints
    - [x] Define optional dependency and fallback policy
    - [x] Update tech-stack notes if the backend adds new tooling
        - No mandatory accelerator tooling is added. The selected initial path is the existing optional module-backed adapter layer (`cuda`, `rocm`, `metal`, and specialized marker-module adapters) with CPU fallback.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Backend Selection' (Protocol in workflow.md)
- [x] [Checkpoint] Conductor - Non-Claim Gate 'Phase 0: H3 Deferral State'
    - [x] Checkpoint evidence: `docs/hpc_track_checkpoint_notes.md`

## Phase 1: Accelerator Replay Prototype

- [x] Task: Implement optional accelerator replay
    - [x] Add device discovery and capability checks
    - [x] Add accelerator replay for supported basis terms
    - [x] Preserve CPU fallback behavior for unavailable backend cases
    - Evidence: `pymars.accelerator.predict_accelerated`,
      `pymars.accelerator.design_matrix_accelerated`, and
      `pymars.accelerator_backends.ArrayModuleAcceleratorBackend`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Accelerator Replay Prototype' (Protocol in workflow.md)

## Phase 2: Parity, Benchmarks, and Docs

- [x] Task: Validate H3 behavior
    - [x] Add CPU-vs-accelerator fixture parity tests
    - [x] Add no-device and fallback tests
    - [x] Run the HPC claim-check gate
    - [x] Add benchmark and documentation updates
    - Evidence: `tests/test_accelerator_runtime.py`,
      `tests/test_accelerator_validation.py`,
      `scripts/benchmark_accelerator_validation.py`, and
      `docs/hpc_track_checkpoint_notes.md`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Parity, Benchmarks, and Docs' (Protocol in workflow.md)
