# Specification: Citation Metadata and JOSS Packet

## Overview

Create the root citation and software-paper artifacts needed for scientific
review and external stewardship submissions. This track is documentation and
metadata only; it must not change the runtime API.

## Requirements

- Add `CITATION.cff` with project title, authorship, license, repository URL,
  package URL, and preferred citation.
- Add `codemeta.json` with machine-readable software metadata aligned with the
  release inventory and package naming policy.
- Add a JOSS-style `paper.md` and `paper.bib` draft with a statement of need,
  package summary, references, and reproducibility notes.
- Document how citation metadata, DOI setup, and software-paper artifacts relate
  to pyOpenSci, rOpenSci, NumFOCUS, and JOSS readiness.
- Preserve the invariant that external package names use `mars-earth` and the
  Python import namespace remains `pymars`.

## Dependencies

- Depends on the current release inventory and package naming policy.
- Blocks final community submission packets that require citation metadata.
- Can run in parallel with supply-chain, HPC packaging, ABI, and workspace work.

## Acceptance Criteria

- Root citation files validate as structured data where applicable.
- The JOSS packet is present and clearly marked as draft until a DOI exists.
- Docs link the new citation artifacts from the community-readiness pages.
- `uv run mkdocs build --strict` passes.

## Out of Scope

- Submitting to JOSS or creating an external DOI account integration.
- Changing source code, runtime behavior, or package names.
