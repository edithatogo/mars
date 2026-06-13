# Release and Parity Inventory

This page records the current package identities, version sources, logging
surface, and known release blockers across the supported language packages.
It is the working inventory for the release-readiness and registry-governance
tracks. The canonical machine-readable source of truth is
[release_metadata.json](release_metadata.json).

The public project brand is `mars-earth`. Language-specific package and module
names remain ecosystem-native when a registry or toolchain requires them. HPC
readiness and external HPC packaging claims are governed by
[HPC Contracts](hpc_contracts.md).

## Package Matrix

| Language | Package / module | Version source | Logging surface | Current release stance |
| --- | --- | --- | --- | --- |
| Python | `mars-earth` / `pymars` import name | `pyproject.toml` + `pymars/__init__.py` | `logging` module in package code and CLI | Published on PyPI as `1.0.4`; trusted publishing and workflow wiring are already configured |
| Rust | `mars-earth` | `rust-runtime/Cargo.toml` | Rust logging is not yet centralized | Published on crates.io as `0.1.0`; native extension source remains |
| R | `marsearth` | `bindings/r/DESCRIPTION` | Host errors are surfaced from runtime helpers | Published on CRAN as `0.0.0` on 2026-05-19 after superseding earlier same-day uploads under invalid or legacy package names |
| Julia | `MarsEarth` | `bindings/julia/Project.toml` | Host errors are surfaced from runtime helpers | Renamed from the already-published `MarsRuntime`; `MarsEarth` needs a new Julia General registration and `MarsRuntime` should be treated as superseded |
| C# | `mars-earth` | `bindings/csharp/MarsRuntime.csproj` | Host exceptions from runtime bridge/helpers | Published on NuGet as `0.0.0` |
| Go | `github.com/edithatogo/mars/bindings/go` | `bindings/go/go.mod` | Host errors from runtime bridge/helpers | Published via signed annotated `bindings/go/v0.1.0` tag; release is tag-driven |
| TypeScript | `mars-earth` | `bindings/typescript/package.json` | JavaScript exceptions from runtime helpers | Published on npm as `0.0.0` |

## Known External Blockers

| Package | Owner / credential status | Action | Status | Date |
| --- | --- | --- | --- | --- |
| Spack `mars-earth` | Upstream Spack review workflow not accessible from this workspace | Submit the prepared H0-only recipe PR and record the review URL | PR open: https://github.com/spack/spack-packages/pull/4781; no human review feedback yet | 2026-05-11 |
| EasyBuild `MarsEarth` / `pymars-0.1.0` | Upstream EasyBuild review workflow not accessible from this workspace | Submit the prepared H0-only easyconfig PR and record the review URL | PR open: https://github.com/easybuilders/easybuild-easyconfigs/pull/25951; no human review feedback yet | 2026-05-11 |
| conda-forge `mars-earth` | staged-recipes workflow not accessible from this workspace | Submit the prepared H0-only recipe PR and record the review URL | PR open: https://github.com/conda-forge/staged-recipes/pull/33290; automated conda-forge lint feedback addressed; no human review yet | 2026-05-11 |
| HPSF/E4S packet drafts | TAC review pending; forum selection remains a follow-up | Review the H0/H1 readiness inquiry and decide whether a fuller packet should advance | TAC readiness inquiry submitted: https://github.com/hpsfoundation/tac/issues/88; no TAC comments yet | 2026-05-11 |
| R `marsearth` | Maintainer-owned package name; r-universe registry configured for source repository | Keep CRAN package metadata and check-result links current; treat earlier `marsruntime` / `mars.earth` uploads as superseded | Published on CRAN as `0.0.0`: https://cran.r-project.org/web/packages/marsearth/index.html | 2026-05-19 |
| Julia `MarsEarth` | New package identity; `MarsRuntime` is already published | Register `MarsEarth` in Julia General; keep `MarsRuntime` as superseded legacy package (raw query `https://raw.githubusercontent.com/JuliaRegistries/General/master/M/MarsRuntime/Package.toml` exists; `.../MarsEarth/Package.toml` currently returns 404) | Blocked pending maintainer access to Registrator.jl; no registration PR yet | 2026-06-14 |

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
| R `marsearth` | r-universe registry repo configured; earlier `marsruntime` / `mars.earth` uploads superseded | `bindings/r/DESCRIPTION`, `bindings/r/cran-comments.md`, CRAN package page, CRAN check results | Published on CRAN as `0.0.0`; package source, binaries, reference manual, and vignette are available from CRAN |
| Julia `MarsEarth` | Not yet registered in Julia General; `MarsRuntime` published as `0.1.0` | `bindings/julia/Project.toml`; Julia General query: `MarsRuntime` exists, `MarsEarth` currently not found | New registration required because Julia registry identity is package-name plus UUID |

## Parity Notes

- Package versions are currently independent by ecosystem, but the release
  process should document any intentional skew.
- Logging should remain quiet by default; verbose diagnostics should be
  opt-in and should preserve the Rust core error context when available.
- Registry ownership and credentials are no longer blockers for PyPI, crates.io,
  npm, NuGet, or the R `marsearth` CRAN release. R is published on CRAN after earlier
  `marsruntime` / `mars.earth` CRAN uploads were superseded, Go is
  a documented tag-driven release policy, and Julia needs a new `MarsEarth`
  registration because `MarsRuntime` is a separate published package identity.

## Next Steps

- Monitor the open H0-only Spack, EasyBuild, and conda-forge upstream PRs and
  respond to maintainer review.
- Monitor the HPSF TAC readiness inquiry and decide whether it should advance
  to a fuller HPSF/E4S packet after TAC feedback.
- Register the Julia package as `MarsEarth`; keep `MarsRuntime` documented as
  a superseded legacy package.
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
  package-level help topic, packaged fixtures, vignette, and manual build path
  are in place, and `marsearth` is published on CRAN as version `0.0.0`.
- R CRAN post-publish smoke was rechecked on 2026-06-14 by installing
  `marsearth` from CRAN into a temporary library, loading it, and running
  `predict_model` plus `design_matrix` against `tests/fixtures/model_spec_v1.json`.
