# Specification: Operator-Facing Diagnostics and Uncertainty Reporting

## Overview

The current Python surface already exposes model summaries, basis-trace
inspection, feature importances, and plotting helpers. This track makes that
operator-facing surface explicit, keeps it consistent after the Rust-backed
runtime migration, and decides whether uncertainty reporting should become a
first-class supported feature.

## Dependency Notes

- Depends on the stable Rust-backed runtime path, because diagnostics should be
  validated against the same portable `ModelSpec` artifacts that runtime users
  consume.
- Should not block core runtime parity or package publication.
- Should remain separate from new model-family work such as extra GLM families,
  ensemble wrappers, or robust-fitting variants.

## Functional Requirements

- Inventory the current diagnostics surface:
  - `summary`
  - `trace`
  - `feature_importances_`
  - basis-function and residual plots
  - partial dependence and ICE plots
- Decide which diagnostics are stable public API, which are experimental, and
  which remain internal helpers.
- Preserve diagnostics for Rust-backed fits, including any data needed by
  summary or feature-importance helpers.
- Decide the policy for uncertainty reporting, including prediction intervals
  or an explicit unsupported-feature response.
- Keep default logging quiet while ensuring diagnostic output can surface the
  relevant model context when requested.

## Non-Functional Requirements

- Diagnostics must remain deterministic for fixed fixtures.
- Any uncertainty output must be documented with clear model-family and data
  assumptions.
- Plotting and summary helpers should continue to work without requiring the
  Rust core to own visualization directly.

## Acceptance Criteria

- Public docs clearly describe the supported diagnostics surface.
- Rust-backed models do not lose the ability to produce supported diagnostics.
- Uncertainty-reporting behavior is either implemented or explicitly rejected
  with tested, user-facing errors.
- Tests cover the stable diagnostics and reporting contract.

## Out of Scope

- Core forward/pruning search changes.
- Cross-language training API expansion.
- Package publication and registry governance.
- New model families such as bagging wrappers or additional GLM variants.
