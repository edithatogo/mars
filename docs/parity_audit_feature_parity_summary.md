# Feature Parity Summary

This note condenses the upstream `py-earth` feature matrix into a short
parity-audit summary. It should be read alongside:

- [py-earth Feature Matrix](parity_audit_feature_matrix.md)
- [Consolidated Parity Table](parity_audit_parity_table.md)
- [Parity Audit Gap Classification](parity_audit_gap_classification.md)

## Summary

- **Core model**: the upstream baseline is a scikit-learn-style mars
  estimator for regression, with `Earth` as the central model surface.
- **Basis terms**: the documented basis vocabulary is constant, linear, hinge,
  and their interaction products.
- **Training / pruning**: fitting is two-stage, with a forward pass that grows
  terms and a pruning pass that selects a subset by gcv-style criteria.
- **Categorical / missingness**: missing-data handling is explicitly
  documented via `allow_missing`; categorical support is acknowledged in the
  lineage, but the public API does not expose a dedicated categorical mode.
- **Diagnostics**: the documented surface includes summaries, traces, feature
  importance, `score`, `transform`, and derivative-related helpers.
- **R-specific extensions**: the upstream R `earth` lineage also exposes
  plotting, prediction intervals, GLM-style extensions, and update workflows;
  those are parity-critical upstream surfaces even though they are not part of
  the current repo gap matrix.
- **Formula / interface**: the upstream interface is estimator-centric rather
  than formula-centric, and the docs emphasize array-like, pandas, and
  scikit-learn compatibility.
- **Packaging**: the upstream package is documented as a conventional Python
  distribution with Sphinx docs and pickle support; it is not a Rust-first or
  multi-registry release model.

## Audit Read-Through

- The matrix treats the core model, basis vocabulary, training flow, and
  diagnostics as the primary parity surface.
- missingness is a documented user-facing behavior and should be compared as a
  contract, not a side effect.
- Categorical handling remains the least explicit area in the documented
  public API and should stay visible in later parity checks.
- The R-side plot / interval / GLM / update surface is still upstream
  parity-critical and should remain visible as a future comparison lane, not
  as already-closed work.
- Formula ergonomics and packaging semantics are useful comparison points, but
  they describe the upstream contract rather than a requirement to mirror it in
  the current repo.

## Use

Use this page as the compact entry point for the feature-matrix slice, then
follow the linked matrix and parity-table docs for line-level evidence and
classification detail.
