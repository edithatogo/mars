# Julia General Registration Draft

## Track

- `julia_general_registration_submission_20260511`

## Submission state

- `MarsEarth` is not yet present in Julia General as of this run.
- `MarsRuntime` currently exists in Julia General with package name `MarsRuntime`
  and legacy UUID.
- This track requests a new `MarsEarth` registration with explicit legacy-note.

## Proposed Registry Text

- Package name: `MarsEarth`
- Package source: `https://github.com/earthanalyticsr/pymars`, subdir
  `bindings/julia`
- License: As declared in project metadata.
- Description: Portable Mars runtime replay package surface backed by the Rust runtime.
- Legacy note: `MarsRuntime` should be retained as a separate superseded legacy
  identity; this registration is for the new `MarsEarth` family identity.
- Scope: Runtime surface and replay features that align with the repository’s
  current contracts.

## Manual review blockers (current)

- Julia General registration requires external workflow access (Registrator/maintainer
  review).
- External submission URL and review status are pending at this time.

## Suggested registry PR body

```
Title: Register MarsEarth

This PR registers `MarsEarth` for the new runtime identity documented in this
repository. The existing `MarsRuntime` package remains published but is intentionally
treated as a legacy/pre-existing identity and should not be treated as the target
release package for this project.

- Distribution identity: `MarsEarth` (Julia General)
- Source: https://github.com/earthanalyticsr/pymars (subdir: bindings/julia)
- Contract level currently satisfied: H0 (HPC packaging), with H1 runtime parallelism
  implemented for replay in repository evidence.
```
