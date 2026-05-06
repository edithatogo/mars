# marsruntime

Portable R runtime replay for mars `ModelSpec` artifacts.

This package evaluates validated model specifications and produces design
matrices and predictions through the shared runtime bridge.

## Install

Install from the package source tree or from the published registry once the
R release path is complete.

## Usage

```r
library(marsruntime)
spec <- list(
  spec_version = "1.0",
  basis_terms = list(),
  coefficients = list(),
  feature_schema = list(n_features = 0)
)
validate_model_spec(spec)
```

## Training

The package also exposes `fit_model(...)` for Rust-backed training. It returns a
portable `ModelSpec` that can be replayed through the same validation and
prediction helpers.

## Validation

```sh
Rscript tests/conformance.R
```

## Documentation

Run `R CMD check --no-manual --as-cran` from the repository root to validate
the package metadata, Rd help pages, and CRAN-safe test behavior.
