# Release Checklist

This checklist is for maintainers preparing a publication run.

Use [Publication Handoff](publication_handoff.md) to capture the remaining
registry inputs.

## Registry Ownership and Credentials

- Confirm the PyPI project owner and trusted publishing configuration for
  `mars-earth` if any publishing change is planned.
- Confirm the crates.io registry token for `mars-earth` exists.
- Confirm the npm automation token or trusted publishing path for `mars-earth`
  exists and is wired as `NPM_TOKEN` or the equivalent GitHub Actions/OIDC
  path.
- Confirm the NuGet trusted publishing policy for `mars-earth` exists and the
  GitHub Actions login path is configured.
- Confirm the Go module tag-signing and release-tag policy.
- Confirm the R submission path through r-universe and CRAN is available.
- Confirm the Julia submission path through Registrator.jl and the General
  registry is available.
