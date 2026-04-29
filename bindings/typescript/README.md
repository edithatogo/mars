# @mars-earth/runtime

Portable JavaScript runtime replay for mars `ModelSpec` artifacts.

This package loads validated model specifications and evaluates design matrices
and predictions in JavaScript. It is runtime-only: training is intentionally
unsupported here and is provided by the training-capable bindings or the Rust
CLI `fit` command.

## API

- `loadModelSpec(raw)`
- `validate(spec)`
- `designMatrix(spec, rows)`
- `predict(spec, rows)`
- `fitModel()` throws a clear unsupported-feature error

## Validation

```sh
npm test
```
