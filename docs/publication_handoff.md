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
| Spack H0 recipe | External Spack PR access and maintainer review | H0 submission track and upstream recipe draft | PR open: https://github.com/spack/spack-packages/pull/4781; no human review feedback yet |
| EasyBuild H0 easyconfig | External EasyBuild PR access and maintainer review | H0 submission track and upstream easyconfig draft | PR open: https://github.com/easybuilders/easybuild-easyconfigs/pull/25951; no human review feedback yet |
| conda-forge H0 feedstock | staged-recipes submission access and maintainer review | H0 submission track and recipe draft | PR open: https://github.com/conda-forge/staged-recipes/pull/33290; automated conda-forge lint feedback addressed; no human review yet |
| HPSF/E4S packet drafts | TAC review and follow-up forum selection | HPSF/E4S packet draft and tracker notes | TAC readiness inquiry submitted: https://github.com/hpsfoundation/tac/issues/88; no TAC comments yet |
| R `marsearth` | Keep CRAN metadata and check-result links current; supersede the earlier `marsruntime` / `mars.earth` uploads | Release checklist, `bindings/r/cran-comments.md`, and maintainer notes | Published on CRAN as `0.0.0`; CRAN install/load/predict/design-matrix smoke passed on 2026-06-14 |
| Julia `MarsEarth` | Register new Julia package; keep `MarsRuntime` as superseded legacy package | Release checklist and maintainer notes | Blocked pending maintainer access to Registrator.jl; no registration PR yet |

## How to Provide It

- Put secret values into GitHub Actions secret settings, not in the repository.
- For R, use the published CRAN `marsearth` package page and check-result page
  as the release evidence, and treat the earlier `marsruntime` / `mars.earth`
  CRAN uploads as superseded. For Julia, register `MarsEarth` as a new package
  and keep `MarsRuntime` as superseded legacy.

## Once Set

- I can update the release inventory and publication track to mark the
  relevant blockers complete.
- I can then move on to the publication execution track without re-discovering
  registry state.
