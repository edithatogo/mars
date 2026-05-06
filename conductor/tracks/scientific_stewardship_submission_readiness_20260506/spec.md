# Specification: Scientific Stewardship and Submission Readiness for Polyglot Scientific Libraries

## Overview

The project is now technically mature across multiple language surfaces, but
its external scientific identity is still fragmented. This track creates the
community-readiness roadmap for the whole project and turns that roadmap into a
concrete submission-prep package for scikit-learn-contrib, pyOpenSci, rOpenSci,
NumFOCUS, JOSS, speck, and EasyBuild.

The track also defines how a polyglot scientific library should organize its
repository and documentation so the shared Rust core, language bindings,
release governance, and scientific-community guidance are easy to find and
maintain.

## Dependency Notes

- Depends on the current published package state and canonical release
  metadata.
- Should not change the public Python API or the existing package names.
- Should stay independent from accelerator/HPC implementation work, but it may
  reference the HPC roadmap and community positioning.

## Functional Requirements

- Inventory the current repository against community-readiness expectations for
  scikit-learn-contrib, pyOpenSci, rOpenSci, NumFOCUS, JOSS, speck, and
  EasyBuild.
- Produce a gap matrix that identifies:
  - current strengths
  - required repo/documentation changes
  - submission artifacts still missing
  - which changes are shared across communities versus community-specific
- Include venue-specific notes for JOSS, speck, and EasyBuild, even where the
  submission target is a paper, build workflow, or ecosystem listing rather than
  a traditional package-review body.
- Define a recommended repository and documentation organization for a
  polyglot scientific library.
- Produce a roadmap for community-facing submission readiness, including what
  should be updated before any external submission packet is sent.
- Document the package identity and stewardship story across Python, Rust, R,
  Julia, Go, C#, and TypeScript.
- Include mermaid diagrams showing the current and target stewardship states.

## Non-Functional Requirements

- No breaking public API changes.
- Preserve the current `pymars` import path and `mars-earth` brand.
- Keep the output useful for future external review, but do not make external
  submissions part of this track.
- Keep documentation concise enough to serve as a living roadmap, not a static
  essay.

## Acceptance Criteria

- A stewardship gap matrix exists in the docs.
- The repo/docs organization recommendation is explicit and actionable.
- Submission-readiness criteria are documented for each target community.
- The roadmap page includes current-state and future-state mermaid diagrams.
- The Conductor roadmap clearly references the community-readiness work.

## Out of Scope

- Making the actual external submissions.
- Changing the public API or packaging names.
- Implementing HPC acceleration or ABI changes.
- Altering the already-published registry state.
