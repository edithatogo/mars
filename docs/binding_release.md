# Binding Release Management

Binding packages are released only after their conformance checks pass.

## Package Managers

| Binding | Package manager | Credential or gate |
| --- | --- | --- |
| Python | PyPI | Existing trusted publishing or PyPI token for `mars-earth` |
| Rust | crates.io | `CARGO_REGISTRY_TOKEN` |
| TypeScript | npm | `NPM_TOKEN` for `@mars-earth/runtime` |
| R | r-universe, then CRAN | Maintainer review and CRAN/r-universe setup |
| Julia | Julia General registry | Registrator.jl and maintainer approval |
| C# / .NET 11 preview | NuGet | `NUGET_API_KEY` |
| Go | Go modules | Signed repository tag for the Go module path |

## CI/CD

`bindings-ci.yml` validates the runtime replay surface for every binding family.
`bindings-publish.yml` provides manual publish scaffolding. The publish workflow
defaults to dry-run behavior and should not be used for real publication until
registry ownership, credentials, and versioning policies are confirmed.

## Release Rule

Do not publish stable runtime packages until Rust-backed bindings pass
conformance and duplicated MVP replay logic is removed, isolated as a tested
fallback, or explicitly labeled experimental/pre-release.

Do not publish stable training packages until Rust training orchestration and
training-capable language bindings pass shared training conformance.

Release readiness is separate from publication. Registry ownership, credentials,
protected environments, version policy, dry-run artifacts, and external blockers
must be resolved or recorded before a publish workflow runs.
