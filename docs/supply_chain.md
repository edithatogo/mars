# Supply Chain Security

This page records the repo-level policy for dependency automation, secret
scanning, and artifact provenance.

## Dependency Automation

- Renovate is the dependency automation source of truth for GitHub Actions and
  supported package ecosystems.
- lockfile updates should stay reproducible and come from the scheduled
  dependency automation window unless a maintainer is handling a targeted fix.
- Minor and patch updates may automerge only when the required status checks
  pass.
- Major updates should remain explicit review items and keep a changelog or
  release-note trail.
- Dependency review runs on pull requests and is treated as a merge gate for
  moderate-or-higher changes.

## Secret Scanning

- Repository secret scanning and push protection should be enabled in GitHub
  settings for the repository or organization.
- Secret-related alerts should be treated as release blockers until the leaked
  material is removed and the history is cleaned up.
- Secrets should never be committed to workflow logs, release artifacts, or
  package metadata.

## Artifact Provenance

- Release rehearsals and real release jobs should generate build provenance
  attestations for built binaries and package artifacts where practical.
- Attestations should cover the artifact that consumers actually install or
  execute.
- Consumers should verify provenance with GitHub attestation tooling where the
  ecosystem supports it.
- The repo uses `actions/attest@v4` in release and rehearsal workflows for the
  built artifacts that can be attested directly.
