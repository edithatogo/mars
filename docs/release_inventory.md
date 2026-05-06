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
| R | `marsruntime` | `bindings/r/DESCRIPTION` | Host errors are surfaced from runtime helpers | Runtime replay package with package-level help, vignette, manual build path, and CRAN-safe checks complete; ready for external submission |
| Julia | `MarsRuntime` | `bindings/julia/Project.toml` | Host errors are surfaced from runtime helpers | Registration submitted to Julia General; awaiting review |
| C# | `mars-earth` | `bindings/csharp/MarsRuntime.csproj` | Host exceptions from runtime bridge/helpers | Published on NuGet as `0.0.0` |
| Go | `github.com/edithatogo/mars/bindings/go` | `bindings/go/go.mod` | Host errors from runtime bridge/helpers | Published via signed annotated `bindings/go/v0.1.0` tag; release is tag-driven |
| TypeScript | `mars-earth` | `bindings/typescript/package.json` | JavaScript exceptions from runtime helpers | Published on npm as `0.0.0` |

## Known External Blockers

| Package | Owner / credential status | Action | Status | Date |
| --- | --- | --- | --- | --- |
| R `marsruntime` | Maintainer-owned package name; r-universe / CRAN requirements verified locally | Confirm external submission path and maintainer review steps | Ready for submission | 2026-05-06 |
| Julia `MarsRuntime` | Maintainer-owned package name; registration submitted to Julia General | Await General review and merge | Submitted | 2026-05-04 |

## Registry Audit Results

The public registry-name audit has been run against the current package set.

| Target | Public status | Evidence | Action |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Present | [mars-earth on PyPI](https://pypi.org/project/mars-earth/) | Keep as the Python distribution name unless a rename is approved |
| PyPI `pymars` | Present, unrelated project | [pymars on PyPI](https://pypi.org/project/pymars/) | Avoid using `pymars` as the published package name |
| crates.io `mars-earth` | Present | [crates.io package page](https://crates.io/crates/mars-earth) | Package is live as `0.1.0` |
| npm `mars-earth` | Present | [npm package page](https://www.npmjs.com/package/mars-earth) | Package is live as `0.0.0` |
| NuGet `mars-earth` | Present | [NuGet package page](https://www.nuget.org/packages/mars-earth) | Package is live as `0.0.0` |
| Go module path | Controlled by repository tags | `bindings/go/go.mod` | Keep the module path aligned with signed `bindings/go/v<version>` tags; `bindings/go/v0.1.0` has been published |
| R `marsruntime` | Not published in a public registry yet | `bindings/r/DESCRIPTION` | Locally complete; ready for r-universe submission and later CRAN review |
| Julia `MarsRuntime` | Not published in a public registry yet; registration PR open | `bindings/julia/Project.toml` | Await General review and merge |

## Parity Notes

- Package versions are currently independent by ecosystem, but the release
  process should document any intentional skew.
- Logging should remain quiet by default; verbose diagnostics should be
  opt-in and should preserve the Rust core error context when available.
- Registry ownership and credentials are no longer blockers for PyPI, crates.io,
  npm, or NuGet. R is ready for submission, Go is a documented tag-driven
  release policy, and Julia is submitted and awaiting registry review.

## Next Steps

- Confirm the remaining manual-review registry path for R.
- Use the [Release Checklist](release_checklist.md) to record the manual
  confirmations and to fill in any remaining blocker rows.
- Decide whether the next release line will keep package versions aligned or
  allow per-ecosystem version skew.
- Define the logging contract for each binding surface and document it in the
  release-readiness track.

## Repository Wiring

- Python release automation already points at `mars-earth` in `pyproject.toml`
  and the GitHub release workflow publishes that distribution name.
- Rust, crates.io, npm, and NuGet packages are live on their registries.
- Go release remains tag-driven via signed annotated `bindings/go/v<version>`
  tags; `bindings/go/v0.1.0` is published.
- R release notes already point to the registry-specific release path, the
  package-level help topic, vignette, and manual build path are in place, and the Julia
  registry submission is already open.
