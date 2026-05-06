# mars / earth repo gap matrix

This note compares the current repository against the upstream mars / earth
references for the comparison slice owned by the parity audit track.

Source base:

- [`py-earth` docs and README](https://github.com/scikit-learn-contrib/py-earth)
- [`py-earth` API/introduction docs](https://contrib.scikit-learn.org/py-earth/content.html)
- [R `earth` reference manual](https://cran.r-universe.dev/earth/doc/manual.html)
- Current repo docs, tests, and package manifests

## Summary

- Core validation, dense-input handling, and single-output sample weights are
  mostly compatible with the upstream references.
- Warning control is intentionally narrower in this repository: warnings are
  surfaced through logging and explicit exceptions rather than user-facing warn
  toggles.
- Deterministic fixture-backed outputs are present, but tie handling is not yet
  documented as an explicit public policy.
- Pickling / estimator serialization and `multioutput` support remain parity
  gaps.
- Package and release behavior intentionally diverges from the upstream
  single-package model because this repository targets a Rust-first,
  multi-registry package family.

## Comparison Matrix

| Subarea | Upstream baseline | Current repo counterpart | Classification | Notes |
| --- | --- | --- | --- | --- |
| Validation and error behavior | `py-earth` documents dense-input model fitting, explicit dimension checks, and fit-time validation; R `earth` also exposes validation-heavy APIs. | `pymars/earth.py`, `pymars/runtime.py`, and `pymars/_model_spec.py` use scikit-learn validation, explicit `ValueError` / `NotFittedError` paths, and portable-spec validation before replay. See also `tests/test_sklearn_compat.py` and `tests/test_coverage_helpers.py`. | compatible | The core validation story is aligned. The remaining question is whether any upstream message wording or error category should be preserved more literally. |
| Warning behavior | R `earth` exposes explicit `warn` controls and documents warning behavior in plotting / diagnostic flows. Upstream `py-earth` documents a more traditional exception-driven estimator surface. | The repo uses logging warnings in `pymars/earth.py` and repo docs specify a quiet-by-default policy in `docs/release_inventory.md`, `docs/release_checklist.md`, and `docs/ci_quality.md`. | upstream-only or intentionally out of scope | This is a deliberate simplification. The repo does not currently present a user-facing `warn` knob like R `earth`. |
| Deterministic outputs | Upstream docs describe canonical ordering and stable transformation behavior for documented examples, including repeated predictions from the same training data. | `tests/test_reference_regression.py` locks representative fitted outputs and metrics for several cases, with a small set of relaxed tolerances for known numeric sensitivity. | compatible | Deterministic fixture coverage exists, but the audit still needs to keep any upstream canonical ordering claims explicit in later notes. |
| Tie handling | Upstream public docs do not present a single explicit tie-breaking contract for all training situations, but they do document canonical ordering rules in some helper flows. | The repo does not yet document a user-facing tie policy. The current tests validate representative outputs, not a formal tie-breaking contract. | unknown | This is a good candidate for a later evidence note or a dedicated fixture if tie-breaking becomes user-visible in the Rust-first core. |
| Examples and documented claims | Upstream docs present canonical examples for the estimator surface and package-specific workflows. | The repo has estimator demos and narrative docs in `pymars/demos/`, `docs/`, and the README, but the current comparison set does not yet fully fixture-check example output parity. | unknown | This is the remaining direct comparison slice for the audit; keep it separate from the core behavioral rows so it can be closed with source-backed examples later. |
| Sample weights | `py-earth` documents `sample_weight` and `output_weight`; R `earth` documents `weights` and a `Force.weights` code path. Zero-weight rows should not contribute, and weights are part of the documented fit semantics. | `tests/test_sklearn_compat.py`, `tests/test_forward.py`, `tests/test_reference_regression.py`, and `rust-runtime/tests/training_tests.rs` cover weighted fits. `pymars/earth.py` validates non-negative weights and rejects zero-total weights. | compatible | Single-output sample weighting is in place. `output_weight` / multiresponse weighting remains out of scope for the current API. |
| missingness and invalid-data edge cases | Upstream `py-earth` documents `allow_missing`; R `earth` documents missing-data handling and related helpers. | The repo supports `allow_missing` and missing-value handling in `pymars/earth.py`, `tests/test_earth.py`, and the shared fixture corpus. Dense-only behavior is retained, and unsupported sparse / complex cases remain checked by estimator tests. | compatible | The current repo matches the dense-input + missingness story closely enough for this slice. |
| Formula / interface ergonomics | Upstream `py-earth` is estimator-centric, while R `earth` is formula-centric. | The repo remains estimator-centric in `pymars/earth.py`, `pymars/runtime.py`, and the docs; there is no public formula interface in the current Python surface. | upstream-only or intentionally out of scope | This is a deliberate cross-ecosystem difference. The audit should keep it explicit so a future formula layer is treated as an extension, not missing parity. |
| serialization and `multioutput` edge cases | Upstream `py-earth` docs say earth objects can be serialized with pickle. R `earth` also documents richer object reconstruction and output structures. | `tests/test_sklearn_compat.py` records an expected `check_estimators_pickle` failure, and `multioutput` regression remains an expected failure in the scikit-learn checks. | parity-critical | These are real parity gaps. They should stay visible in the audit so they do not get mistaken for intentional design choices. |
| Packaging / versioning / release behavior | Upstream `py-earth` is documented as a single Python package with Sphinx docs and source-install flow; R `earth` follows the R package / CRAN style release model. | `docs/release_inventory.md`, `docs/package_release_paths.md`, `docs/publication_handoff.md`, `pyproject.toml`, `rust-runtime/Cargo.toml`, `bindings/typescript/package.json`, `bindings/csharp/MarsRuntime.csproj`, and `bindings/julia/Project.toml` describe a Rust-first, multi-registry package family under the `mars-earth` brand. | upstream-only or intentionally out of scope | This is a deliberate departure from the upstream packaging model. The repo should keep it explicit, not try to fake a single upstream-style release path. |

## repo-side evidence commands

The current comparison slice is backed by the following local checks:

- `uv run pytest -q tests/test_sklearn_compat.py tests/test_reference_regression.py`
- `uv run pytest -q tests/test_earth.py tests/test_coverage_helpers.py`
- `uv run python scripts/check_release_alignment.py`
- `uv run mkdocs build --strict`

## Audit Interpretation

- Treat validation / error / warning behavior as mostly compatible, with a
  documented warning-policy boundary.
- Treat deterministic regression behavior as compatible, but keep tie
  handling explicit until the audit captures a stronger upstream contract.
- Treat single-output sample weights and missingness as compatible.
- Treat pickling, multioutput, and the packaging / release model as the main
  remaining parity or boundary items for this slice.
