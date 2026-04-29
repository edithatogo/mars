# Release Checklist

This checklist is for maintainers who are preparing an actual publication run.
It captures the manual confirmation steps that cannot be inferred safely from
the repository alone.

## Before a Release Rehearsal

- Confirm the package family being rehearsed and whether it is runtime-only or
  includes training APIs.
- Confirm the protected release environment exists and is required for the
  publish job.
- Confirm the latest release candidate artifacts were produced by the current
  commit.
- Confirm the rehearsal workflow uploaded reviewable artifacts for the target
  package family.

## Registry Ownership and Credentials

- Confirm the PyPI project owner and whether trusted publishing or a token is
  configured for `mars-earth`.
- Confirm the crates.io owner and publishing token for `pymars-runtime`.
- Confirm the npm organization/package owner and publish token for
  `@mars-earth/runtime`.
- Confirm the NuGet package ID owner and API key for `MarsRuntime`.
- Confirm the Go module tag-signing and release-tag policy.
- Confirm the R submission path through r-universe and CRAN.
- Confirm the Julia submission path through Registrator.jl and the General
  registry.

## Before Publishing

- Confirm the blocker table in `release_inventory.md` has an owner, action,
  date, and status for every unresolved item.
- Confirm the release notes describe any intentional version skew between
  package ecosystems.
- Confirm the stable-versus-pre-release label matches the package contents.
- Confirm the package-specific smoke test passed from the built artifact.
- Confirm provenance or attestation artifacts were generated where the
  ecosystem supports them.

## After Publishing

- Confirm the published artifact matches the rehearsal artifact.
- Confirm the registry page and README links point to the canonical project.
- Confirm any release notes or changelog entries were updated to match the
  published version.
- Record any post-publish issue in the blocker table or the release notes.
