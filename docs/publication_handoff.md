# Publication Handoff

This page captures the remaining information needed to move from release
readiness to actual publication.

## Required Inputs

| Package | What I need from you | Where it should live | Current state |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Nothing further unless you change the publish path | GitHub Actions trusted publishing | Published |
| crates.io `mars-earth` | No further action unless you republish or change ownership | `CARGO_REGISTRY_TOKEN` in GitHub Actions | Published |
| npm `mars-earth` | Nothing further unless you change the publish path | `NPM_TOKEN` (automation token) or npm trusted publishing | Published |
| NuGet `mars-earth` | Nothing further unless you change the publish path | `NuGet/login@v1` short-lived API key exchange | Published |
| Go module | Tag-signing rule and who signs/reviews release tags | Repository policy / release checklist | Waiting on policy confirmation |
| R `marsruntime` | r-universe / CRAN submission path and maintainer account details | Release checklist and maintainer notes | Waiting on maintainer confirmation |
| Julia `MarsRuntime` | Registration submitted through Registrator.jl; awaiting General review | Release checklist and maintainer notes | Submitted |

## How to Provide It

- Put secret values into GitHub Actions secret settings, not in the repository.
- Reply with the Go tag policy if you want the release flow to enforce a
  specific signing or review rule.
- Reply with the R and Julia submission-path details when you are ready to
  move those packages through r-universe, CRAN, or the General registry.

## Once Set

- I can update the release inventory and publication track to mark the
  relevant blockers complete.
- I can then move on to the publication execution track without re-discovering
  registry state.
