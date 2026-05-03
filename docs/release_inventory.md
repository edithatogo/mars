# Release and Parity Inventory

This page records the current package identities, version sources, logging
surface, and known release blockers across the supported language packages.
It is the working inventory for the release-readiness and registry-governance
tracks.

All packages belong to the `mars-earth` family. The registry/package IDs below
are the ecosystem-specific identifiers used where a registry requires a
concrete name.

## Package Matrix

| Language | Package / module | Version source | Logging surface | Current release stance |
| --- | --- | --- | --- | --- |
| Python | `mars-earth` / `pymars` import name | `pyproject.toml` + `pymars/__init__.py` | `logging` module in package code and CLI | Public PyPI project exists; trusted publishing and workflow wiring are already configured |
| Rust | `mars-earth` | `rust-runtime/Cargo.toml` | Rust logging is not yet centralized | Runtime crate and native extension source |
| R | `marsruntime` | `bindings/r/DESCRIPTION` | Host errors are surfaced from runtime helpers | Runtime replay package, release readiness pending |
| Julia | `MarsRuntime` | `bindings/julia/Project.toml` | Host errors are surfaced from runtime helpers | Runtime replay package, release readiness pending |
| C# | `mars-earth` | `bindings/csharp/MarsRuntime.csproj` | Host exceptions from runtime bridge/helpers | Runtime replay package, release readiness pending |
| Go | `github.com/edithatogo/mars/bindings/go` | `bindings/go/go.mod` | Host errors from runtime bridge/helpers | Runtime replay package, release readiness pending |
| TypeScript | `mars-earth` | `bindings/typescript/package.json` | JavaScript exceptions from runtime helpers | Runtime replay package, release readiness pending |

## Known External Blockers

| Package | Owner / credential status | Action | Status | Date |
| --- | --- | --- | --- | --- |
| PyPI `mars-earth` | Confirmed by maintainer; trusted publishing is configured in GitHub Actions | Document the owner and publishing path as ready for release | Confirmed | 2026-04-29 |
| crates.io `mars-earth` | Maintainer-owned namespace; registry token not verified in this inventory | Confirm crate owner and registry token | Owner confirmed | 2026-04-29 |
| npm `mars-earth` | Maintainer-owned package name; publish token verified for registry auth, but publish still fails with `EOTP` unless an automation token or trusted publishing is used | Confirm npm org, automation-token/trusted-publishing path, and GitHub Actions wiring | Partial | 2026-05-03 |
| NuGet `mars-earth` | Maintainer-owned package ID; trusted publishing policy configured in GitHub Actions | Confirm package ID ownership and GitHub Actions login path | Ready | 2026-04-30 |
| Go module path | Repository-controlled, tag-based release path documented in this inventory | Confirm module path and tagging policy | Documented | 2026-04-29 |
| R `marsruntime` | Maintainer-owned package name; r-universe / CRAN requirements not verified in this inventory | Confirm submission path and maintainer review steps | Owner confirmed | 2026-04-29 |
| Julia `MarsRuntime` | Maintainer-owned package name; registry path not verified in this inventory | Confirm registry path and UUID/review requirements | Owner confirmed | 2026-04-29 |

## Registry Audit Results

The public registry-name audit has been run against the current package set.

| Target | Public status | Evidence | Action |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Present | [mars-earth on PyPI](https://pypi.org/project/mars-earth/) | Keep as the Python distribution name unless a rename is approved |
| PyPI `pymars` | Present, unrelated project | [pymars on PyPI](https://pypi.org/project/pymars/) | Avoid using `pymars` as the published package name |
| crates.io `mars-earth` | Not registered | [crates.io package page](https://crates.io/crates/mars-earth) | Reserve or publish when ownership is confirmed |
| npm `mars-earth` | Not registered | [npm package page](https://www.npmjs.com/package/mars-earth) | Publish still needs an automation token or trusted publishing; current granular token hits `EOTP` |
| NuGet `mars-earth` | Not registered | [NuGet package page](https://www.nuget.org/packages/mars-earth) | Reserve or publish when ownership is confirmed |
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
  package manager, excluding PyPI `mars-earth`, which is already owned and
  wired for trusted publishing.
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
- Rust and NuGet publish jobs already reference the expected secret names or
  trusted-publishing paths in GitHub Actions. npm still needs an
  automation-token or trusted-publishing path; the current granular token hits
  `EOTP` during publish.
- Go release remains tag-driven.
- R and Julia release notes already point to their registry-specific release
  paths.
