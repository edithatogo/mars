# Specification

## Objective

Harden the public estimators so the sklearn contract is explicit, well-tested, and easier to rely on for downstream users.

## Scope

- Re-run and improve `check_estimator` coverage for `EarthRegressor` and `EarthClassifier`.
- Reduce the current expected-failure list where the implementation can be brought into compliance.
- Decide and codify the policy for sparse inputs.
- Decide and codify the policy for multi-output regression and classification.
- Improve classifier ergonomics around calibrated probabilities, decision functions, and multiclass behavior.
- Update tests and user-facing documentation to reflect the supported contract.

## Out of Scope

- New non-sklearn model families.
- Non-Python runtime portability work.
- Cross-language model consumers.

## Acceptance Criteria

- `check_estimator` expectations are documented and only waived where there is a deliberate, tested boundary.
- Sparse-input and multi-output behavior is either implemented or rejected with clear, tested error messages.
- Classifier behavior for probabilities, decision scores, and multiclass workflows is documented and covered by tests.
- Public docs describe the supported estimator contract without ambiguity.
