# Specification: Workspace Automation Export

## Overview

Prepare a source-controlled workspace export plan for Linear and Notion so the
SOTA work lanes can be mirrored into external project-management spaces when
authenticated access is available.

## Requirements

- Define a Linear project, milestone, label, and issue taxonomy for the six
  SOTA implementation lanes.
- Define a Notion workspace structure for roadmap, submission packets,
  dependencies, decisions, and evidence.
- Provide CLI-oriented setup and export commands for `linear-cli` and
  `notionctl` without storing tokens.
- Include a synchronization rule from Conductor track status to external
  workspaces.
- Keep source-controlled templates safe to share publicly.

## Dependencies

- Depends on the lane taxonomy in the SOTA dependency plan.
- Requires authenticated local CLI sessions before external workspace writes can
  be executed.
- Can run in parallel with citation, packaging, and governance work.

## Acceptance Criteria

- Workspace automation docs include source-controlled templates or command
  snippets for Linear and Notion.
- Secret handling is explicit and no tokens are committed.
- The workspace layout mirrors the six Conductor lanes and their dependencies.
- `uv run mkdocs build --strict` passes.

## Out of Scope

- Creating external pages or issues without authenticated access.
- Storing Notion or Linear credentials in the repository.
- Replacing Conductor as the source of truth.
