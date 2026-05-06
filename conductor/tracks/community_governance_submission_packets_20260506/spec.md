# Specification: Community Governance and Submission Packets

## Overview

Prepare reusable governance and submission packets for scientific, language,
and foundation communities. The packets should make the project easier to
submit for review while keeping external submissions gated on maintainer
approval and account access.

## Requirements

- Add or update governance, code of conduct, support, and contributor materials
  needed by community reviewers.
- Prepare submission packets for scikit-learn-contrib, pyOpenSci, rOpenSci,
  NumFOCUS, JOSS, PyPA, .NET Foundation, Julia, R, HPSF, and E4S.
- Link packets back to citation, supply-chain, packaging, ABI, and roadmap
  evidence.
- Mark every external submission step that requires maintainer account action.
- Keep package naming and import namespace language consistent.

## Dependencies

- Depends on citation metadata for final packet text.
- Depends on supply-chain and packaging evidence for PyPA, HPSF, and E4S.
- Can run in parallel with supply-chain, ABI, and workspace work while citation
  artifacts are being drafted.

## Acceptance Criteria

- Community-readiness docs include packet checklists and evidence links.
- Governance docs explain maintainership, contributions, support, and conduct.
- External account actions are explicitly separated from source-editable work.
- `uv run mkdocs build --strict` passes.

## Out of Scope

- Submitting applications to external communities.
- Creating legal entities or changing project ownership.
- Changing package names or public APIs.
