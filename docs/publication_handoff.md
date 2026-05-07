# Publication Handoff

This page captures the remaining information needed to move from release
readiness to actual publication. The canonical machine-readable release
metadata is [release_metadata.json](release_metadata.json).

Versioning follows the current release policy in `release_metadata.json`:
ecosystem-native package names, intentional version skew across ecosystems,
release notes required for publishable changes, and quiet-by-default logging.

## Required Inputs

| Package | What I need from you | Where it should live | Current state |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Nothing further unless you change the publish path | GitHub Actions trusted publishing | Published |
| crates.io `mars-earth` | No further action unless you republish or change ownership | `CARGO_REGISTRY_TOKEN` in GitHub Actions | Published |
| npm `mars-earth` | Nothing further unless you change the publish path | `NPM_TOKEN` (automation token) or npm trusted publishing | Published |
| NuGet `mars-earth` | Nothing further unless you change the publish path | `NuGet/login@v1` short-lived API key exchange | Published |
| Go module | Signed annotated tag policy and who signs/reviews release tags | Repository policy / release checklist | Published via `bindings/go/v0.1.0` tag |
| R `marsruntime` | r-universe / CRAN submission path and maintainer account details | Release checklist and maintainer notes | Package help, vignette, and manual build path are complete; awaiting external submission |
| Julia `MarsRuntime` | Registration submitted through Registrator.jl; package-local license files added; release notes retriggered; auto-merge checks passed and the PR is waiting out the mandatory General review period | Release checklist and maintainer notes | Submitted; waiting out mandatory General review period |

## How to Provide It

- Put secret values into GitHub Actions secret settings, not in the repository.
- Reply with the R submission-path details when you are ready to move that
  package through r-universe or CRAN. Julia remains on the General registry
  path after the package-local license fix and release-note retrigger; it is
  now waiting out the mandatory General review period.

## Once Set

- I can update the release inventory and publication track to mark the
  relevant blockers complete.
- I can then move on to the publication execution track without re-discovering
  registry state.
