# conda-forge Feedstock Notes

This lane prepares a staged-recipes-ready draft for `mars-earth` under H0 rules.
It does not imply accelerator, distributed, or training claims (H0).

## Assumptions

- The H0 package should remain source-installable without accelerator runtimes.
- The lane remains isolated from runtime and release registries while keeping the
  feedstock draft and readiness evidence explicit.
- The current release story remains anchored in the existing Python, Rust, and
  binding build commands.

## What This Lane Does

- Tracks `recipe/meta.yaml` for a `mars-earth` feedstock draft.
- Keeps feasibility and submission details separate from release registries.
- Keeps API and runtime unchanged.

## What This Lane Does Not Do

- Prepare a staged-recipes PR submission in place of this draft lane.
- Change the public package API.
