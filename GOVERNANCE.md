# Governance

This project is maintainer-led and CODEOWNERS-backed. Governance is intentionally
small and explicit so technical decisions, publication decisions, and external
submission packets stay aligned with the actual repository state.

## Decision Model

- Day-to-day code, docs, and workflow changes are reviewed through pull
  requests and the existing CI gates.
- Release and publication decisions must stay aligned with the release
  inventory, release checklist, and publication handoff docs.
- Community submission packets are prepared separately from code changes and
  should only be treated as ready when the relevant readiness doc says so.
- Changes that affect public API, package identity, or release flow should be
  documented in the appropriate Conductor track before they are treated as
  stable.

## Maintainer Responsibilities

- Keep the repository in a releasable state.
- Keep the release inventory and community-readiness docs truthful.
- Triage bugs, documentation issues, and community questions.
- Review or route external submission packets for scikit-learn-contrib,
  pyOpenSci, rOpenSci, NumFOCUS, JOSS, PyPA, .NET Foundation, Julia, R, HPSF,
  and E4S.
- Preserve the scikit-learn-compatible Python surface while the Rust core
  continues to mature.

## Review and Change Control

- PRs should use the issue templates and follow the commit and CI conventions in
  `CONTRIBUTING.md`.
- `CODEOWNERS` defines the maintainer routing used by GitHub for review.
- Release-bearing changes should be validated against `docs/release_inventory.md`
  and `docs/release_checklist.md` before they are treated as finished.
- Citation metadata is owned by the citation lane. This governance document does
  not edit citation files itself; it only describes how they fit into review and
  submission packets.

## External Submission Policy

- scikit-learn-contrib: treat as the Python-facing compatibility home.
- pyOpenSci: prepare a pre-submission inquiry and scientific workflow narrative.
- rOpenSci: prepare the R submission path and package-review packet.
- NumFOCUS: prepare the stewardship, sustainability, and maintainer model.
- JOSS: prepare the paper packet only after citation metadata and a stable
  release are in place.
- PyPA: treat as packaging-alignment evidence, not an external submission.
- .NET Foundation: treat as ecosystem stewardship and project-health alignment.
- Julia: treat as a registry and community review workflow through General.
- R: treat as a package-review and publication workflow through r-universe and
  CRAN.
- HPSF and E4S: treat as HPC packaging and portability readiness packets.

## Related Docs

- [Support](SUPPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Community Submission Readiness](docs/community_submission_readiness.md)
- [Ecosystem and Foundation Alignment](docs/ecosystem_foundation_alignment.md)
