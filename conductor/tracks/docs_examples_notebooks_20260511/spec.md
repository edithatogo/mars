# Docs, Examples, and Notebooks

## Overview

Create comprehensive documentation and runnable examples for the Python API
and each language binding, including notebooks that cover the major workflows.

## Functional Requirements

- Add a canonical notebook or notebook-equivalent for each binding where
  notebook execution is practical.
- Add runnable examples for fit, predict, export, validate, explain, and
  package/binding workflows.
- Keep docs synchronized with code and release claims.

## Non-Functional Requirements

- Examples must be reproducible.
- Notebook execution should be testable in CI where practical.
- Documentation must remain accurate when code changes.

## Acceptance Criteria

- Each supported language has at least one runnable example or notebook
  covering the binding surface.
- Core workflows have canonical examples.
- Docs build and example execution are covered by tests.

## Out of Scope

- Accelerator kernel implementation.
- Distributed execution.
- Packaging submission work.

