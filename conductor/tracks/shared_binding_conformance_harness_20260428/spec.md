# Specification

## Objective

Create a shared conformance harness that every runtime binding can use to prove
portable `ModelSpec` replay compatibility before any training logic migrates to
Rust.

## Scope

- Define a binding-agnostic fixture manifest over the existing
  `tests/fixtures/model_spec_*.json` and
  `tests/fixtures/runtime_portability_fixture_*.json` pairs.
- Provide a reusable Python conformance runner that validates fixture discovery,
  expected output schema, and parity output files emitted by bindings.
- Document the expected binding output format.
- Ensure the harness is usable from language-specific CI jobs.
- Keep the harness independent of `py-earth` and R `earth`.

## Out of Scope

- Training/fitting conformance.
- Requiring every binding to be implemented in this track.
- Replacing the existing Rust fixture tests.

## Acceptance Criteria

- A shared conformance manifest exists.
- A runner can validate fixture schema and binding parity outputs.
- Documentation explains how bindings consume fixtures and emit results.
- CI documentation identifies how each binding invokes conformance checks.
- The harness passes against the current fixture corpus.
