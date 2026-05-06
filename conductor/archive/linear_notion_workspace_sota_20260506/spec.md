# Specification: Linear and Notion Workspace SOTA for Polyglot Stewardship

## Overview

This track establishes a SOTA operating model for the project management and
knowledge spaces used around `pymars`. The goal is not to change the library
itself, but to make the surrounding Linear and Notion spaces first-class,
maintainable, and clearly aligned with the project's polyglot, scientific, and
release-governance work.

The track is intentionally split into four reviewable phases:

- Linear development
- Linear review
- Notion development
- Notion review

The workspace model should also provide a place to track scientific venues and
ecosystem targets such as JOSS, speck, and EasyBuild.

## Dependency Notes

- Depends on the existing Conductor setup and the current docs/track registry.
- Should not change the public library API.
- Should align with the scientific stewardship roadmap and the HPC/ABI roadmap.

## Functional Requirements

- Define the intended Linear structure for this project:
  - project or issue hierarchy
  - review loops and ownership
  - a place to track community and release work
- Define the intended Notion structure for this project:
  - knowledge base pages
  - stewardship and decision logs
  - docs that support submission readiness and roadmap governance
- Define how the workspace surfaces will capture and track:
  - scientific stewardship work
  - release and publication work
  - JOSS, speck, and EasyBuild readiness
  - Apache Arrow, PyPA, .NET Foundation, Julia communities, and R communities
- Define a review loop for each workspace area so the development and review
  passes can be carried out separately and iteratively.
- Keep the workspace model aligned with the repo roadmap rather than creating
  a disconnected planning silo.

## Non-Functional Requirements

- No library API changes.
- Workspace taxonomy should be stable and readable for maintainers.
- The setup should favor low-maintenance, high-signal pages/issues over
  duplicate copies of repo documentation.
- The final structure should be easy to update as tracks complete.

## Acceptance Criteria

- Linear and Notion have a documented workspace operating model.
- The four-phase development/review cadence is explicit.
- JOSS, speck, and EasyBuild are represented in the workspace plan.
- The workspace plan links back to the repo roadmap and Conductor tracks.
- Apache Arrow, PyPA, .NET Foundation, Julia communities, and R communities
  are represented in the workspace plan.
- The plan preserves the distinction between development notes and review
  notes.

## Out of Scope

- Building new library features.
- Changing the public API or package layout.
- Creating external submissions.
- Replacing the current repo documentation system.
