# Specification: Ecosystem and Foundation Alignment for Polyglot Scientific Libraries

## Overview

This track defines how the project should position itself with the broader
scientific and packaging ecosystems that matter to a polyglot scientific
library. The goal is not external submission yet; the goal is to make the
project ready for ecosystem-facing stewardship conversations with Apache Arrow,
PyPA, the .NET Foundation, and the Julia and R communities.

## Dependency Notes

- Depends on the existing release metadata, roadmap, and stewardship tracks.
- Must not change the public API or package names.
- Should complement, not replace, the scientific submission-readiness track.

## Functional Requirements

- Inventory the project against ecosystem expectations for:
  - Apache Arrow
  - PyPA
  - .NET Foundation
  - Julia communities
  - R communities
- Identify what those ecosystems value in a polyglot scientific library:
  - packaging and distribution maturity
  - interoperability and data interchange
  - community governance and stewardship
  - documentation and maintenance posture
- Produce a gap matrix describing:
  - current strengths
  - missing repo/doc artifacts
  - ecosystem-specific positioning notes
  - shared changes that improve multiple ecosystem stories at once
- Define the repo/doc changes that would make those ecosystems a credible fit
  for future stewardship, adoption, or listing conversations.

## Non-Functional Requirements

- No breaking API changes.
- Keep the ecosystem narrative aligned with the current `mars-earth` brand and
  the `pymars` import path.
- Keep the output actionable for future work, not overly speculative.

## Acceptance Criteria

- The ecosystem/foundation readiness matrix exists in the docs.
- Apache Arrow, PyPA, .NET Foundation, Julia communities, and R communities
  are explicitly represented.
- The repo/doc changes required for each ecosystem are clear.
- The track points back to the roadmap and the stewardship submission-readiness
  work.

## Out of Scope

- External submissions or applications.
- Changing the public API.
- Adding new runtime features purely for ecosystem optics.
