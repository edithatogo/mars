# Specification: Supply Chain Scorecard and SBOM

## Overview

Add supply-chain evidence that strengthens PyPA, foundation, and HPC-readiness
narratives without expanding core functionality. The work should improve
visibility into dependency, build, and release provenance.

## Requirements

- Add or document OpenSSF Scorecard execution for the repository.
- Add SBOM generation for release artifacts where the current build system can
  support it without secret material.
- Add release provenance or attestation guidance for GitHub Actions releases.
- Update supply-chain documentation to explain generated artifacts, validation
  commands, and remaining external setup.
- Keep workflows non-interactive and safe for forked pull requests.

## Dependencies

- Depends on existing CI and release manifests.
- Supports PyPA, NumFOCUS, HPSF, E4S, and foundation submission evidence.
- Can run in parallel with citation, packaging, ABI, and governance packet work.

## Acceptance Criteria

- Workflow or documented command exists for Scorecard evidence.
- Workflow or documented command exists for SBOM generation.
- Supply-chain docs explain how to review generated artifacts.
- CI changes do not require tokens for ordinary pull-request validation.

## Out of Scope

- Storing secrets or registry tokens in the repository.
- Changing package publication credentials.
- Rewriting the existing release process.
