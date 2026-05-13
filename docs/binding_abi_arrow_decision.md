# Binding ABI and Apache Arrow Decision

## Decision

Apache Arrow is **deferred**, not adopted as part of the current binding ABI.

## Context

The current binding boundary is already stable enough for the supported flows:

- JSON spec payloads are compact and versioned
- runtime evaluation uses small numeric row-major matrices
- training requests and replies already travel through explicit JSON payloads
- the repo already has an ABI-safe Rust core path and host wrappers

That makes Arrow a dependency and ABI expansion, not a correctness requirement.

## Why It Is Deferred

- It would expand the supported surface before the narrow ABI has finished
  settling.
- It would add a second serialization model alongside the current JSON
  contract.
- It would increase language-specific adapter work without a proven user need.
- It would complicate error handling and ownership rules for an optional fast
  path.

## What Is Kept Stable

- no public API changes
- no change to the `ModelSpec` JSON contract
- no change to the current Rust-owned runtime/training boundaries
- no change to the host wrapper expectations for Python, Rust, Go, C#, TypeScript,
  R, or Julia

## Future-Proofing

Arrow can still be prototyped later if a real bulk-data transfer case appears.
If that happens, the right shape is:

- optional and opt-in
- limited to a clearly separated data-transfer path
- behind a dedicated Conductor track
- validated against the existing JSON path before any default switch

## Record

- `status`: deferred
- `priority`: optional prototype only
- `next_action`: revisit only if bulk transfer or zero-copy exchange becomes a
  demonstrated bottleneck
