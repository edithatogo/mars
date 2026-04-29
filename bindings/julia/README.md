# MarsRuntime

Portable Julia runtime replay for mars `ModelSpec` artifacts.

This package evaluates validated model specifications and produces design
matrices and predictions through the shared runtime bridge.

## Training

The package also exposes `fit_model(...)` for Rust-backed training. It returns a
portable `ModelSpec` that can be replayed through the shared runtime helpers.

## Validation

```sh
julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```
