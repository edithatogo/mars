# go runtime binding

Portable Go runtime replay for mars `ModelSpec` artifacts.

This module evaluates validated model specifications and produces design
matrices, predictions, and Rust-backed training outputs through the shared
runtime bridge.

## Training

The Go binding now exposes `FitModel(request)` for Rust-backed training. The
function returns a portable `ModelSpec` artifact that can be validated and
replayed through the shared runtime helpers.

## Validation

```sh
go test ./...
```
