# Specification: R Package Publication Submission

## Overview

Prepare the `marsearth` R package for external publication and carry out the
browser-based submission workflow required to publish it through the intended
registry path.

This track focuses on the R package only. It assumes the package is already
locally complete and that the remaining work is submission, verification, and
release-state synchronization.

## Dependency Notes

- Depends on the R package readiness work already completed in prior tracks.
- Depends on release inventory and publication-handoff state being accurate.
- Depends on maintainer access for the external submission path.
- Browser interaction is required for the submission step when registry forms
  or web UI actions are involved.

## Functional Requirements

- Confirm the current R package metadata, version, and release notes are
  ready for publication.
- Confirm the external publication path to r-universe and, when appropriate,
  CRAN.
- Use a browser-based workflow to submit the package through the external
  publication process.
- Record any blocker encountered during submission, including owner, action,
  and date.
- Verify the package is visible from the target registry after submission or
  record the external state if the registry still requires review.
- Update release documentation and inventory files to reflect the publication
  status.

## Non-Functional Requirements

- Registry credentials and maintainer secrets must not be committed.
- Browser-based submission steps must be reproducible and auditable.
- The process must fail closed if the package metadata, build artifacts, or
  submission prerequisites are not ready.

## Acceptance Criteria

- The R publication path is documented and actionable.
- A browser-based submission attempt is completed or a concrete blocker is
  recorded.
- The release inventory and handoff docs match the current registry state.
- Any external review dependency is explicit rather than implied.

## Out of Scope

- Implementing new R package functionality.
- Changing the package’s public API or runtime behavior.
- Publishing unrelated language packages.
