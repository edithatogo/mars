# Publication Handoff

This page captures the remaining information needed to move from release
readiness to actual publication for the `mars-earth` package family.

## Required Inputs

| `mars-earth` family target | What I need from you | Where it should live | Current state |
| --- | --- | --- | --- |
| PyPI `mars-earth` | Nothing further unless you change the publish path | GitHub Actions trusted publishing | Ready |
| crates.io `mars-earth-runtime` | Registry token and confirmation of the owner/team | `CARGO_REGISTRY_TOKEN` in GitHub Actions | Waiting on secret setup |
| npm `@mars-earth/runtime` | Publish token or trusted-publishing equivalent and scope owner confirmation | `NPM_TOKEN` or npm trusted publishing | Waiting on secret setup |
| NuGet `MarsEarth.Runtime` | API key and package owner confirmation | `NUGET_API_KEY` in GitHub Actions | Waiting on secret setup |
| Go module | Tag-signing rule and who signs/reviews release tags | Repository policy / release checklist | Waiting on policy confirmation |
| R `mars-earth` family package | r-universe / CRAN submission path and maintainer account details | Release checklist and maintainer notes | Waiting on maintainer confirmation |
| Julia `MarsEarth` | Registrator.jl / General submission path and package UUID/review details | Release checklist and maintainer notes | Waiting on maintainer confirmation |

## How to Provide It

- Put secret values into GitHub Actions secret settings, not in the repository.
- Reply with the maintainer-approved owner/team names for crates.io, npm,
  NuGet, R, and Julia.
- Reply with the Go tag policy if you want the release flow to enforce a
  specific signing or review rule.

## Open The Pages

- [crates.io: `mars-earth-runtime`](https://crates.io/crates/pymars-runtime)
- [npm: `@mars-earth/runtime`](https://www.npmjs.com/package/@mars-earth/runtime)
- [NuGet: `MarsEarth.Runtime`](https://www.nuget.org/packages/MarsRuntime)
- [Registrator.jl Web UI](https://juliaregistries.github.io/Registrator.jl/stable/webui/)
- [CRAN](https://cran.r-project.org/)
- [r-universe](https://r-universe.dev/search)

## Once Set

- I can update the release inventory and publication track to mark the
  relevant blockers complete.
- I can then move on to the publication execution track without re-discovering
  registry state.
