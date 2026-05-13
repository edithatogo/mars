# MARS / Earth Parity Audit Phase 0 Evidence

## Upstream References Reviewed

- `py-earth` README on GitHub
- R `earth` reference manual on r-universe

## Upstream Behavior Signals from py-earth

The `py-earth` README explicitly calls out the following future-work areas:

- improved speed
- exporting models to additional formats
- shared-memory multiprocessing during fitting
- cyclic predictors
- better categorical predictor support
- better support for large data sets
- iterative reweighting during fitting

The README also documents missing-data support and references the core
scikit-learn-style estimator interface and plotting/summary behavior.

## Upstream Behavior Signals from R earth

The `earth` manual exposes a broader operator-facing surface, including:

- summary output for fitted models
- plotting support for fitted models and variance models
- prediction intervals through variance models
- variable importance helpers
- GLM-style model extensions and related prediction modes
- model update workflows

## Audit Boundary

This track should classify which upstream behaviors are parity-critical for the
current repository, which are intentionally out of scope, and which are
reasonable later extensions once the Rust core and bindings are stable.
