# marsruntime

Portable R runtime replay for mars `ModelSpec` artifacts.

This package evaluates validated model specifications and produces design
matrices and predictions through the shared runtime bridge.

## Training

The package also exposes `fit_model(...)` for Rust-backed training. It returns a
portable `ModelSpec` that can be replayed through the same validation and
prediction helpers.

## Validation

```sh
Rscript tests/conformance.R
```
