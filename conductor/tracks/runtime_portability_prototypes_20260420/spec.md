# Specification

## Objective

Validate the portability contract with a Rust reference runtime and a shared
golden fixture corpus, establishing the first step toward a shared Rust core.

## Scope

- Generate a golden fixture corpus from Python `ModelSpec` artifacts and expected runtime outputs.
- Implement a Rust reference runtime that loads portable model artifacts without Python runtime dependencies.
- Validate Rust `validate`, `design_matrix`, and `predict` behavior against the Python-produced fixtures.
- Capture any contract gaps or ambiguities discovered while building the Rust runtime.
- Identify what remains before the Rust runtime can become the shared fitting and evaluation core.

## Out of Scope

- Production-ready bindings for every target language.
- Production binding work for R, Julia, Go, TypeScript, C#, Python extension packaging, or additional Rust packaging variants.
- ONNX or alternative interchange-layer evaluation.
- Public release management for portability packages.
- Long-term API support commitments beyond the contract-level findings.

## Acceptance Criteria

- The repo contains Python-generated golden fixtures covering representative valid `ModelSpec` exports and expected `design_matrix` and `predict` outputs.
- A Rust reference runtime loads a portable model artifact and executes `validate`, `design_matrix`, and `predict` without relying on Python objects, `Earth`, sklearn state, or pickle payloads.
- Rust runtime outputs match the Python fixture expectations for the representative fixture corpus.
- Any contract gaps discovered during implementation are recorded as explicit follow-up tasks rather than left implicit.
- The Rust prototype is documented as the seed of the future shared Rust core, not as a sidecar-only experiment.
