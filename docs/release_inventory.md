# Release and Parity Inventory

This page records the current package identities, version sources, logging
surface, and known release blockers across the supported language packages.
It is the working inventory for the release-readiness and registry-governance
tracks.

## Package Matrix

| Language | Package / module | Version source | Logging surface | Current release stance |
| --- | --- | --- | --- | --- |
| Python | `mars-earth` / `pymars` import name | `pyproject.toml` + `pymars/__init__.py` | `logging` module in package code and CLI | Public PyPI project exists; release-ready after registry and credential checks |
| Rust | `pymars-runtime` | `rust-runtime/Cargo.toml` | Rust logging is not yet centralized | Runtime crate and native extension source |
| R | `marsruntime` | `bindings/r/DESCRIPTION` | Host errors are surfaced from runtime helpers | Runtime replay package, release readiness pending |
| Julia | `MarsRuntime` | `bindings/julia/Project.toml` | Host errors are surfaced from runtime helpers | Runtime replay package, release readiness pending |
| C# | `MarsRuntime` | `bindings/csharp/MarsRuntime.csproj` | Host exceptions from runtime bridge/helpers | Runtime replay package, release readiness pending |
| Go | `github.com/edithatogo/mars/bindings/go` | `bindings/go/go.mod` | Host errors from runtime bridge/helpers | Runtime replay package, release readiness pending |
| TypeScript | `@mars-earth/runtime` | `bindings/typescript/package.json` | JavaScript exceptions from runtime helpers | Runtime replay package, release readiness pending |

## Known External Blockers

| Package | Owner / credential status | Action | Status | Date |
| --- | --- | --- | --- | --- |
| PyPI `mars-earth` | Trusted publishing or token not confirmed in this inventory | Confirm registry owner and publishing path | Open | 2026-04-28 |
| crates.io `pymars-runtime` | Owner/token not confirmed in this inventory | Confirm crate owner and registry token | Open | 2026-04-28 |
| npm `@mars-earth/runtime` | Token and organization ownership not confirmed in this inventory | Confirm npm org and publish token | Open | 2026-04-28 |
| NuGet `MarsRuntime` | API key and package ID ownership not confirmed in this inventory | Confirm package ID ownership and API key | Open | 2026-04-28 |
| Go module path | Signed tag strategy not yet confirmed in this inventory | Confirm module path and tagging policy | Open | 2026-04-28 |
| R `marsruntime` | r-universe / CRAN requirements not confirmed in this inventory | Confirm submission path and maintainer review steps | Open | 2026-04-28 |
| Julia `MarsRuntime` | General registry / Registrator path not confirmed in this inventory | Confirm registry path and UUID/review requirements | Open | 2026-04-28 |

## Registry Audit Results

The public registry-name audit has been run against the current package set.

| Target | Public status | Evidence | Action |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Present | [mars-earth on PyPI](https://pypi.org/project/mars-earth/) | Keep as the Python distribution name unless a rename is approved |
| PyPI `pymars` | Present, unrelated project | [pymars on PyPI](https://pypi.org/project/pymars/) | Avoid using `pymars` as the published package name |
| crates.io `pymars-runtime` | Not registered | [crates.io package page](https://crates.io/crates/pymars-runtime) | Reserve or publish when ownership is confirmed |
| npm `@mars-earth/runtime` | Not registered | [npm package page](https://www.npmjs.com/package/@mars-earth/runtime) | Reserve or publish when ownership is confirmed |
| NuGet `MarsRuntime` | Not registered | [NuGet package page](https://www.nuget.org/packages/MarsRuntime) | Reserve or publish when ownership is confirmed |
| Go module path | Controlled by repository tags | `bindings/go/go.mod` | Keep the module path aligned with signed tags |
| R `marsruntime` | Not published in a public registry yet | `bindings/r/DESCRIPTION` | Confirm r-universe/CRAN path during release prep |
| Julia `MarsRuntime` | Not published in a public registry yet | `bindings/julia/Project.toml` | Confirm General/Registrator path during release prep |

## Parity Notes

- Package versions are currently independent by ecosystem, but the release
  process should document any intentional skew.
- Logging should remain quiet by default; verbose diagnostics should be
  opt-in and should preserve the Rust core error context when available.
- Registry ownership and credentials are still the main external blockers for
  stable publication.

## Next Steps

- Confirm external registry ownership and publishing credentials for each
  package manager.
- Decide whether the next release line will keep package versions aligned or
  allow per-ecosystem version skew.
- Define the logging contract for each binding surface and document it in the
  release-readiness track.
- Run the `release-rehearsal.yml` workflow and record the resulting artifacts
  and smoke-test output for the supported package managers.

## Repository Wiring

- Python release automation already points at `mars-earth` in `pyproject.toml`.
- Rust, npm, and NuGet publish jobs already reference the expected secret names
  in GitHub Actions.
- Go release remains tag-driven.
- R and Julia release notes already point to their registry-specific release
  paths.
