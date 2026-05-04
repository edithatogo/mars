# Publication Handoff

This page captures the remaining information needed to move from release
readiness to actual publication.

## Required Inputs

| Package | What I need from you | Where it should live | Current state |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Nothing further unless you change the publish path | GitHub Actions trusted publishing | Ready |
| crates.io `mars-earth` | Registry token, confirmation of the owner/team, and a verified crates.io email on the publishing account | `CARGO_REGISTRY_TOKEN` in GitHub Actions | Waiting on secret setup |
| npm `mars-earth` | Automation token for the first publish, or trusted publishing once the package exists and can be registered | `NPM_TOKEN` (automation token) or npm trusted publishing | Waiting on secret setup |
| NuGet `mars-earth` | Trusted publishing owner confirmation and GitHub Actions login path | `NuGet/login@v1` short-lived API key exchange | Ready |
| Go module | Tag-signing rule and who signs/reviews release tags | Repository policy / release checklist | Waiting on policy confirmation |
| R `marsruntime` | r-universe / CRAN submission path and maintainer account details | Release checklist and maintainer notes | Waiting on maintainer confirmation |
| Julia `MarsRuntime` | Registrator.jl / General submission path and package UUID/review details | Release checklist and maintainer notes | Waiting on maintainer confirmation |

## How to Provide It

- Put secret values into GitHub Actions secret settings, not in the repository.
- Reply with the maintainer-approved owner/team names for crates.io, npm,
  NuGet, R, and Julia.
- Reply with the Go tag policy if you want the release flow to enforce a
  specific signing or review rule.

## Once Set

- I can update the release inventory and publication track to mark the
  relevant blockers complete.
- I can then move on to the publication execution track without re-discovering
  registry state.
