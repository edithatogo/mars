# Specification

## Objective

Clarify the public product surface so users can tell what the package is called, what import style is supported, which APIs are stable, and how the runtime portability story should be consumed.

## Scope

- Tighten docs so the package name, repo name, and import style are consistent.
- Add a clear stable-versus-experimental policy for `EarthCV` and `GLMEarth`.
- Publish compatibility and performance guidance for embedded and runtime consumers.

## Out of Scope

- New estimator implementation work.
- New portability-runtime implementation work.
- Cross-language prototype code.

## Acceptance Criteria

- User-facing docs consistently describe the naming and import conventions.
- `EarthCV` and `GLMEarth` stability expectations are explicit.
- Runtime and embedded-consumer guidance exists in a single discoverable place.
