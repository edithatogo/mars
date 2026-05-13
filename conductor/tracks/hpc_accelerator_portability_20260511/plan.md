# Implementation Plan

## Phase 0: Backend Selection and Deferral Hygiene

- [x] Task: Record and hold H3 deferral contract
  - [x] Add explicit H3 deferral wording to checkpoint evidence:
    - current supported runtime: CPU replay only
    - accelerator backend status: intentionally not yet implemented
    - gating rationale: H1 kernel shape and packaging tradeoff review pending
  - [x] Confirm checkpoint language appears in release-facing docs or packet drafts
    with explicit `not yet` / `non-goal` / `deferred` phrasing
  - [x] Update `docs/hpc_claim_review_checklist.md` H3 non-claim verification items if needed
    before any H3 submission claim is considered.

- [ ] Task: Select the initial accelerator backend
    - [ ] Review H1 benchmark data and kernel shape
    - [ ] Compare candidate backends against portability and packaging constraints
    - [ ] Define optional dependency and fallback policy
    - [ ] Update tech-stack notes if the backend adds new tooling
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Backend Selection' (Protocol in workflow.md)
- [ ] [Checkpoint] Conductor - Non-Claim Gate 'Phase 0: H3 Deferral State'
    - [ ] Checkpoint evidence: `docs/hpc_track_checkpoint_notes.md`

## Phase 1: Accelerator Replay Prototype

- [ ] Task: Implement optional accelerator replay
    - [ ] Add device discovery and capability checks
    - [ ] Add accelerator replay for supported basis terms
    - [ ] Preserve CPU fallback behavior for unsupported cases
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Accelerator Replay Prototype' (Protocol in workflow.md)

## Phase 2: Parity, Benchmarks, and Docs

- [ ] Task: Validate H3 behavior
    - [ ] Add CPU-vs-accelerator fixture parity tests
    - [ ] Add no-device and fallback tests
    - [ ] Run the HPC claim-check gate
    - [ ] Add benchmark and documentation updates
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Parity, Benchmarks, and Docs' (Protocol in workflow.md)
