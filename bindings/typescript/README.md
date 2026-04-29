# @mars-earth/runtime

Portable JavaScript runtime replay for mars `ModelSpec` artifacts.

This package loads validated model specifications and evaluates design matrices
and predictions in JavaScript. It prefers the Rust CLI runtime when a built
binary is available and falls back to the local JavaScript evaluator for
compatibility. Training is intentionally unsupported here and is provided by
the training-capable bindings or the Rust CLI `fit` command.

## API

- `loadModelSpec(raw)`
- `validate(spec)`
- `designMatrix(spec, rows)`
- `predict(spec, rows)`
- `fitModel()` throws a clear unsupported-feature error

The Node package is runtime-only. Browser support is not provided by the
current Rust CLI mechanism.

## Validation

```sh
npm test
```
