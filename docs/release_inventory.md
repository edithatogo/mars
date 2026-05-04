# Release and Parity Inventory

This page records the current package identities, version sources, logging
surface, and known release blockers across the supported language packages.
It is the working inventory for the release-readiness and registry-governance
tracks.

The public project brand is `mars-earth`. Language-specific package and module
names remain ecosystem-native when a registry or toolchain requires them.

## Package Matrix

| Language | Package / module | Version source | Logging surface | Current release stance |
| --- | --- | --- | --- | --- |
| Python | `mars-earth` / `pymars` import name | `pyproject.toml` + `pymars/__init__.py` | `logging` module in package code and CLI | Published on PyPI as `1.0.4`; trusted publishing and workflow wiring are already configured |
| Rust | `mars-earth` | `rust-runtime/Cargo.toml` | Rust logging is not yet centralized | Published on crates.io as `0.1.0`; native extension source remains |
| R | `marsruntime` | `bindings/r/DESCRIPTION` | Host errors are surfaced from runtime helpers | Runtime replay package, release readiness pending |
| Julia | `MarsRuntime` | `bindings/julia/Project.toml` | Host errors are surfaced from runtime helpers | Runtime replay package, release readiness pending |
| C# | `mars-earth` | `bindings/csharp/MarsRuntime.csproj` | Host exceptions from runtime bridge/helpers | Published on NuGet as `0.0.0` |
| Go | `github.com/edithatogo/mars/bindings/go` | `bindings/go/go.mod` | Host errors from runtime bridge/helpers | Runtime replay package, release readiness pending |
| TypeScript | `mars-earth` | `bindings/typescript/package.json` | JavaScript exceptions from runtime helpers | Published on npm as `0.0.0` |

## Known External Blockers

| Package | Owner / credential status | Action | Status | Date |
| --- | --- | --- | --- | --- |
| crates.io `mars-earth` | Maintainer-owned namespace; publishing required the configured registry token and a verified crates.io email on the account | No blocker remains unless the crate is republished | Published | 2026-05-04 |
| R `marsruntime` | Maintainer-owned package name; r-universe / CRAN requirements not verified in this inventory | Confirm submission path and maintainer review steps | Owner confirmed | 2026-04-29 |
| Julia `MarsRuntime` | Maintainer-owned package name; registry path not verified in this inventory | Confirm registry path and UUID/review requirements | Owner confirmed | 2026-04-29 |

## Registry Audit Results

The public registry-name audit has been run against the current package set.

| Target | Public status | Evidence | Action |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Present | [mars-earth on PyPI](https://pypi.org/project/mars-earth/) | Keep as the Python distribution name unless a rename is approved |
| PyPI `pymars` | Present, unrelated project | [pymars on PyPI](https://pypi.org/project/pymars/) | Avoid using `pymars` as the published package name |
| crates.io `mars-earth` | Not published yet | [crates.io package page](https://crates.io/crates/mars-earth) | Publish once ownership and registry credentials are confirmed |
| npm `mars-earth` | Present | [npm package page](https://www.npmjs.com/package/mars-earth) | Package is live as `0.0.0` |
| NuGet `mars-earth` | Present | [NuGet package page](https://www.nuget.org/packages/mars-earth) | Package is live as `0.0.0` |
| Go module path | Controlled by repository tags | `bindings/go/go.mod` | Keep the module path aligned with signed tags |
| R `marsruntime` | Not published in a public registry yet | `bindings/r/DESCRIPTION` | Confirm r-universe/CRAN path during release prep |
| Julia `MarsRuntime` | Not published in a public registry yet | `bindings/julia/Project.toml` | Confirm General/Registrator path during release prep |

## Parity Notes

- Package versions are currently independent by ecosystem, but the release
  process should document any intentional skew.
- Logging should remain quiet by default; verbose diagnostics should be
  opt-in and should preserve the Rust core error context when available.
- Registry ownership and credentials are no longer blockers for PyPI, crates.io,
  npm, or NuGet; R and Julia remain the main external blockers and Go remains a
  repository-controlled release step.

## Next Steps

- Confirm the remaining manual-review registry paths for R and Julia.
- Use the [Release Checklist](release_checklist.md) to record the manual
  confirmations and to fill in any remaining blocker rows.
- Decide whether the next release line will keep package versions aligned or
  allow per-ecosystem version skew.
- Define the logging contract for each binding surface and document it in the
  release-readiness track.
- Run the `release-rehearsal.yml` workflow and record the resulting artifacts
  and smoke-test output for the supported package managers.

## Repository Wiring

- Python release automation already points at `mars-earth` in `pyproject.toml`
  and the GitHub release workflow publishes that distribution name.
- Rust, npm, and NuGet packages are live on their registries.
- Go release remains tag-driven.
- R and Julia release notes already point to their registry-specific release
  paths.
