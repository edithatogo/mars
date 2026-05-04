# Release Checklist

This checklist is for maintainers who are preparing an actual publication run.
It captures the manual confirmation steps that cannot be inferred safely from
the repository alone.

The [Publication Handoff](publication_handoff.md) page is the quickest way to
capture the remaining registry inputs from the maintainer.

## Maintainer Inputs Needed

To move the remaining publication gates forward, the maintainer needs to
confirm or configure:

- the R release path through r-universe and CRAN
- the Julia release path through Registrator.jl and the General registry
- the Go tag-signing and release-tag policy

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

- Confirm the PyPI project owner and trusted publishing configuration for
  `mars-earth` if any publishing change is planned.
- Confirm the crates.io registry token and verified publishing email for
  `mars-earth`.
- Confirm the npm automation token or trusted publishing path for `mars-earth`
  only if a future republish or ownership change is planned.
- Confirm the NuGet trusted publishing policy for `mars-earth` only if a
  future republish or ownership change is planned.
- Confirm the Go module tag-signing and release-tag policy.
- Confirm the R submission path through r-universe and CRAN is available.
- Confirm the Julia submission path through Registrator.jl and the General
  registry is available.

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
