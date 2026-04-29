# Package Release Paths

This page documents the package-manager-specific release paths for the current
binding surface. It is the policy source for the publication track and the
release-rehearsal workflow.

## Release Criteria

- Stable runtime releases require Rust-backed conformance, artifact inspection,
  and approval in the protected release environment.
- Stable training releases additionally require Rust training orchestration and
  training-capable bindings to pass shared training conformance.
- Packages may be released as pre-release or experimental only when the release
  notes and registry metadata say so explicitly.
- Duplicate MVP replay implementations must not be published as stable unless
  they are removed or isolated as tested fallbacks.

## Versioning and Logging

- Use semantic versioning for the package line in each registry.
- Keep release notes and changelog entries aligned with the published version.
- Surface package version and, where practical, runtime/core version reporting
  from each binding.
- Keep logging quiet by default and preserve opt-in access to Rust-core error
  context.

## Package-Manager Paths

### Python

- Registry: PyPI.
- Package name: `mars-earth`.
- Release path: trusted publishing or the configured PyPI token workflow.
- Avoid publishing `pymars`, which already belongs to another project.
- Rehearse with `uv build`, artifact inspection, and install-from-wheel smoke
  tests.

### Rust

- Registry: crates.io.
- Package name: `pymars-runtime`.
- Release path: `cargo package` followed by `cargo publish` with
  `CARGO_REGISTRY_TOKEN` once ownership is confirmed.
- Rehearse with `cargo package --allow-dirty --list` and `cargo package`.

### TypeScript

- Registry: npm.
- Package name: `@mars-earth/runtime`.
- Release scope: runtime-only for now; the package uses Rust CLI acceleration
  where a built binary is available and keeps a JavaScript fallback for
  compatibility. Training is intentionally unsupported in this package line
  until a separate training surface is designed.
- Release path: `npm pack --dry-run`, artifact inspection, and `npm publish`
  with `NODE_AUTH_TOKEN`.
- Rehearse with `npm test`, `npm pack --dry-run`, and install-from-tarball
  smoke tests.

### R

- Registry: r-universe first, then CRAN when the package is ready for formal
  submission.
- Package name: `marsruntime`.
- Release path: build the source tarball, inspect package contents, and follow
  maintainer review before CRAN submission.
- Rehearse with `R CMD build`, local install, and conformance tests.

### Julia

- Registry: Julia General via Registrator.jl.
- Package name: `MarsRuntime`.
- Release path: register after UUID, metadata, and API stability are confirmed.
- Rehearse with `Pkg.instantiate()`, `Pkg.status()`, and `Pkg.test()`.

### C#

- Registry: NuGet.
- Package name: `MarsRuntime`.
- Release path: `dotnet pack` followed by `dotnet nuget push` with
  `NUGET_API_KEY` once the protected release environment is approved.
- Rehearse with `dotnet test`, `dotnet pack`, and install-from-nupkg smoke
  tests.

### Go

- Registry: Go modules.
- Release path: publish by signed repository tag rather than by registry push.
- Rehearse with `go test`, `go mod verify`, and module metadata inspection.

## Rollback, Yank, and Deprecation

- If a release is wrong, prefer registry-native yanking or deletion policies
  where the ecosystem allows it.
- Use a follow-up patch release to correct metadata, README links, or logging
  behavior.
- Deprecation notices should be called out in changelog entries and release
  notes.

## Approval Gates

- No stable publish without a successful rehearsal and explicit maintainer
  approval.
- No stable publish while external registry ownership or credential status is
  unresolved.
- Protected release environments must be used for real publication jobs.
