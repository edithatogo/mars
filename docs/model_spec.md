# ModelSpec Contract

`pymars` exports fitted models as a JSON-serializable `ModelSpec` so runtime
evaluation can be validated independently of Python object state.

## Fixture coverage

The repository keeps the portability contract grounded with checked-in
fixtures:

- `tests/fixtures/model_spec_v1.json`
- `tests/fixtures/runtime_portability_fixture_v1.json`
- `tests/fixtures/model_spec_categorical.json`
- `tests/fixtures/runtime_portability_fixture_categorical.json`
- `tests/fixtures/model_spec_combined.json`
- `tests/fixtures/runtime_portability_fixture_combined.json`
- `tests/fixtures/model_spec_interaction.json`
- `tests/fixtures/runtime_portability_fixture_interaction.json`
- `tests/fixtures/model_spec_missingness.json`
- `tests/fixtures/runtime_portability_fixture_missingness.json`

Together they assert:

- the current runtime can still load a historical `1.0` artifact
- `design_matrix` output remains stable for a representative probe set
- `predict` output remains stable for the same probe set
- contract validation rejects malformed payloads explicitly

The runtime portability fixture is the handoff artifact for the Rust core and
language bindings. The Rust reference runtime consumes the checked-in probe sets
and matches the recorded `design_matrix` and `predict` outputs exactly. This
fixture corpus is the current validation authority for portable consumers and
future Python, R, Julia, Rust, C#, Go, and TypeScript bindings; `py-earth` and R
`earth` are not required runtime oracles for the project.
