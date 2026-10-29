# voiage alignment follow-ups (mars)

Bounded follow-up to the existing scientific ecosystem track; it reuses that owner and adds no parallel programme.

## Scope
1. Confirm the public tabular/array boundary uses Polars (>=2.1) and PyArrow (>=26); record any retained exceptions.
2. Consume `voiage.contracts.normalized_input` bundles; verify `content_digest` round trips (multi-chunk columns, Parquet-normalised types).
3. Align toolchain: Python 3.15.1, uv, ruff, ty (Rust 1.92 where a Rust core exists); record retained-version exceptions.
4. Resolve or scope open Dependabot alerts; keep signed-commit rules satisfied for bot updates.
5. Installed-artifact consumer evidence outside the checkout (command, exit status, imported paths, revisions, hashes).

## Acceptance
Existing gates stay required. Unexecuted checks are not claimed. No release or external submission is implied; approval is pending the owner.
