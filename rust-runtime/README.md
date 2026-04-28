# pymars-runtime

`pymars-runtime` is the Rust replay prototype and future shared core boundary
for `mars`.

Today it validates portable `ModelSpec` artifacts and evaluates
`design_matrix`/`predict` against the checked-in fixture corpus without Python,
sklearn state, or pickle payloads.

It also contains the first training-core primitives:

- weighted least-squares fitting
- RSS calculation
- GCV calculation
- forward candidate scoring
- pruning subset scoring

The intended direction is for this crate to become the shared computational core
surfaced through:

- Python
- R
- Julia
- Rust
- C#
- Go
- TypeScript

## Current Public API

- `load_model_spec_str`
- `load_model_spec_path`
- `validate_model_spec`
- `design_matrix`
- `predict`
- `fit_least_squares`
- `score_candidate`
- `score_pruning_subset`

## Validation

Run the Rust parity tests from this directory:

```bash
cargo test
```

The fixture test discovers paired files in `../tests/fixtures`:

- `model_spec_<name>.json`
- `runtime_portability_fixture_<name>.json`

The Rust outputs must match the Python-produced fixture expectations.
