# Quality Gate Strictness

## Overview

Make the repository's formatting, linting, typing, docstring, validation, and
security policy uniformly strict across CI and local automation.

## Functional Requirements

- Ensure CI and pre-commit agree on strict checks.
- Remove leftover advisory-only behavior where it is not intentional.
- Keep security and validation gates blocking for real issues.

## Non-Functional Requirements

- Strictness changes must not hide real failures.
- Advisory jobs should remain advisory only when explicitly intended.
- The policy should be readable and auditable.

## Acceptance Criteria

- Formatting, linting, typing, and docstring policy are strict and aligned.
- Validation and security jobs fail when they should.
- Advisory jobs are limited to profiling or exploratory workflows.

## Out of Scope

- New product features.
- Accelerator kernels.
- Documentation content beyond quality-policy alignment.

