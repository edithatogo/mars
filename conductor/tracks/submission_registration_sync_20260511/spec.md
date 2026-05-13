# Specification: Submission and Registration Synchronization

## Overview

Keep the remaining submission and registration state synchronized in the repo
docs and Conductor trackers.

This track is for status synchronization only. It does not replace the actual
external submission or review tracks for Spack, EasyBuild, conda-forge,
HPSF/TAC, or Julia General.

## Functional Requirements

- Record the current status and URLs for the open Spack, EasyBuild, and
  conda-forge review lanes.
- Record the HPSF/TAC readiness inquiry URL and current state.
- Record the Julia General registration state and any follow-up notes.
- Capture any maintainer decision points that still need user input.
- Mirror the current review and registration state in the release-facing docs.
- Keep the status-sync work separate from action tracks that open or submit
  external requests.

## Non-Functional Requirements

- Keep the documentation factual and claim-safe.
- Do not hide external decision points inside passive status notes.

## Acceptance Criteria

- The open packaging and governance lanes have current URLs, statuses, and
  follow-up notes in the trackers and release docs.
- Any remaining external decision points are explicitly called out rather than
  hidden.
- `./scripts/check_hpc_claims.sh --strict` still passes after the updates.

## Out of Scope

- Opening or submitting the actual external PRs or registration requests.
- Tutorial content expansion.
- Docs-stack migration work.
