# Implementation Plan

## Phase 0: Workspace Taxonomy

- [ ] Task: Define external workspace model
    - [ ] Map the six Conductor lanes to Linear projects, milestones, labels, and issues
    - [ ] Map the six Conductor lanes to Notion databases, pages, and evidence views
    - [ ] Record required CLI authentication checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Workspace Taxonomy' (Protocol in workflow.md)

## Phase 1: Templates and Commands

- [ ] Task: Add workspace export artifacts
    - [ ] Add safe Linear setup and export command templates
    - [ ] Add safe Notion setup and export command templates
    - [ ] Document how status flows from Conductor to external workspaces
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Templates and Commands' (Protocol in workflow.md)

## Phase 2: Documentation Integration

- [ ] Task: Link workspace automation to the SOTA lanes
    - [ ] Update workspace automation docs
    - [ ] Link workspace setup from the dependency and remaining roadmap pages
    - [ ] Mark all account-level actions as external gates
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Documentation Integration' (Protocol in workflow.md)

## Phase 3: Validation and Handoff

- [ ] Task: Validate workspace docs
    - [ ] Run `uv run mkdocs build --strict`
    - [ ] Verify no credentials or workspace IDs are committed
    - [ ] Record any remaining auth-dependent steps
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and Handoff' (Protocol in workflow.md)
