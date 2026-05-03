# Publication Handoff

This page captures the remaining information needed to move from release
readiness to actual publication for the `mars-earth` package family.

The canonical package matrix lives in [Release Inventory](release_inventory.md).

## Required Inputs

| `mars-earth` family target | What I need from you | Where it should live | Current state |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Nothing further unless you change the publish path | GitHub Actions trusted publishing | Ready |
| crates.io `mars-earth` | Registry token and confirmation of the owner/team | `CARGO_REGISTRY_TOKEN` in GitHub Actions | Waiting on secret setup |
| npm `mars-earth` | Automation token or trusted-publishing equivalent and scope owner confirmation | `NPM_TOKEN` or npm trusted publishing | Waiting on automation path |
| NuGet `mars-earth` | Trusted publishing owner confirmation and GitHub Actions login path | `NuGet/login@v1` short-lived API key exchange | Ready |
| Go module | Tag-signing rule and who signs/reviews release tags | Repository policy / release checklist | Waiting on policy confirmation |
| R `marsruntime` | r-universe / CRAN submission path and maintainer account details | Release checklist and maintainer notes | Waiting on maintainer confirmation |
| Julia `MarsRuntime` | Registrator.jl / General submission path and package UUID/review details | Release checklist and maintainer notes | Waiting on maintainer confirmation |

## Open The Pages

- [crates.io: `mars-earth`](https://crates.io/crates/mars-earth)
- [npm: `mars-earth`](https://www.npmjs.com/package/mars-earth)
- [NuGet: `mars-earth`](https://www.nuget.org/packages/mars-earth)
- [Registrator.jl Web UI](https://juliaregistries.github.io/Registrator.jl/stable/webui/)
- [CRAN](https://cran.r-project.org/)
- [r-universe](https://r-universe.dev/search)
