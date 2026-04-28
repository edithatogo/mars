# Shared Conformance Requirements

Every `mars` runtime or binding must pass the same fixture-backed checks before
it is treated as supported.

## Fixture Discovery

Conformance runners should discover paired files in `tests/fixtures`:

- `model_spec_<name>.json`
- `runtime_portability_fixture_<name>.json`

Each runtime fixture contains:

- `probe`: row-major input samples
- `design_matrix`: expected basis-matrix output
- `predict`: expected prediction output

Bindings may copy these fixtures into package-specific test directories, but the
repository fixtures remain the source of truth.

## Required Runtime Cases

Every binding must validate:

- loading a valid `1.x` `ModelSpec`
- rejecting malformed or unsupported specs
- feature-count mismatch failures
- continuous basis evaluation
- hinge basis evaluation
- interaction basis evaluation
- categorical basis evaluation
- missingness basis evaluation
- prediction from evaluated design matrices

## Numerical Tolerances

Runtime replay should match fixture outputs exactly where host representation
allows it. Floating-point comparisons must use:

- absolute tolerance: `1e-12`
- relative tolerance: `1e-12`
- NaN parity: NaN matches NaN only when the fixture expects NaN

Fitting conformance will use broader tolerances only after the Rust core owns
training behavior and the expected numerical paths are documented.

## Error Contract

Bindings should expose host-language errors that preserve the Rust core category
and message intent.

Required categories:

- malformed artifact
- unsupported artifact version
- missing required field
- unsupported basis term
- feature-count mismatch
- invalid categorical encoding
- numerical evaluation failure

## CI Expectations

Each supported binding should run:

- its host-language unit tests
- shared fixture parity tests
- artifact validation failure tests
- package import/load tests

The Rust core crate is the first required conformance target. Python follows
once runtime helpers are routed through Rust.
