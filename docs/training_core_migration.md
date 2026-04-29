# Training Core Migration

The portable runtime bindings now exercise the `ModelSpec` replay contract
across Python, Rust, Go, TypeScript, R, Julia, and C# package surfaces. The next
migration step is to replace duplicated runtime replay in each binding with
thin Rust-backed runtime bindings. Full training orchestration should then start
behind that validated Rust boundary.

## Binding Feedback

The binding MVPs exposed the core constraints that the Rust training layer must
respect:

- Inputs must stay row-major and shape-explicit at the shared boundary.
- Missing values need explicit NaN/null semantics in fixtures.
- Categorical replay currently supports numeric category encodings only.
- Errors need stable categories so bindings can translate them idiomatically.
- Package CI must run conformance before any training internals are exposed.

## Migration Order

1. Basis evaluation primitives.
2. Weighted least-squares, RSS, and GCV scoring.
3. Forward candidate scoring.
4. Pruning subset scoring.
5. Rust-backed runtime bindings across all supported languages.
6. Full forward/pruning orchestration.
7. Python estimator integration.
8. Training APIs for non-Python language bindings where supported.

The current Rust additions cover steps 1 through 4 as shared primitives:

- public basis-term evaluation
- weighted least-squares/RSS
- GCV scoring
- forward candidate scoring
- pruning subset scoring

Python continues to own the public estimator API until full orchestration has
parity fixtures and can be routed through Rust without changing sklearn-facing
behavior.
