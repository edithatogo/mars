# Canonical Assurance Harness Contract

## Entry Point

The repository exposes one non-interactive interface:

```text
uv run python tools/assurance.py run <profile> [options]
```

`list --json` provides machine-readable discovery. The harness executes static
argument vectors without a shell, defaults to fail-fast, supports an explicit
`--keep-going` diagnostic mode, and returns `0` only when every selected command
passes. Usage errors return `2`; executed profiles with any failure return `1`.

## Profiles

| Profile | Required | Purpose | Maximum local budget |
| --- | --- | --- | --- |
| `fast` | PR gate | Policy, formatting/linting, typing, focused unit/integration/property tests, and coverage | 15 minutes |
| `full` | PR/release gate | Fast plus complete Python, Rust, binding, documentation, smoke, and end-to-end assurance | 60 minutes |
| `security` | PR/release gate | Static analysis, dependency and licence audits, secrets, workflow policy, and SARIF production | 30 minutes |
| `release-rehearsal` | Release gate | Clean builds, package inspection/install tests, SBOM/provenance rehearsal, and cross-runtime conformance without publication | 90 minutes |
| `preview` | Non-blocking canary | Pre-release runtimes, nightly toolchains, emerging runners, and sandboxed agentic diagnostics | 90 minutes |

Required profiles fail closed. Preview results remain visible but cannot merge,
publish, mutate protected refs, or become authoritative without promotion through
the assurance control model.

## Determinism and Environment

- Default seed: `1729`; `--seed` is recorded and propagated through
  `PYTHONHASHSEED`, `HYPOTHESIS_SEED`, and suite-specific seed adapters.
- `CI=true`, UTF-8, non-interactive operation, repository-root working directory,
  and locked dependency/toolchain inputs are mandatory.
- Wall-clock timestamps describe execution only; comparisons use normalized
  paths, stable ordering, explicit time zones, fixed clocks where modeled, and
  deterministic state digests.
- Network access is denied unless a command declares a bounded registry or hosted
  evidence dependency. Missing credentials never cause an interactive prompt.
- Linux is the canonical command/evidence environment. Windows and macOS use the
  same profile contract; unavailable platform-specific commands are explicit
  `blocked` results, not silent skips or successes.

## Outputs and Receipts

Every invocation writes beneath `build/assurance/<run-id>/`:

- `receipt.json`: schema version, profile, source commit, dirty-state digest,
  platform/toolchain identities, seed, start/finish times, command argument
  vectors, durations, exit codes, outcomes, and artifact digests;
- `junit.xml`, `coverage.xml`, SARIF, benchmark, package, SBOM, provenance, and
  conformance outputs when produced by the selected profile;
- bounded stdout/stderr logs with secrets redacted.

Receipts are written atomically, preserve successful commands when later commands
fail, distinguish `passed`, `failed`, `blocked`, `not-run`, and `dry-run`, and
record hosted workflow/check parity. A receipt is evidence of one execution, not
proof that a hosted control remains enabled.

## Resource and Failure Policy

- Each command and profile has an enforced timeout; child processes are terminated
  on expiry and the receipt records `timed-out`.
- Output and retained failure evidence are bounded; secrets, credentials, and
  unrestricted environment dumps are prohibited.
- Fail-fast is the admission default. `--keep-going` collects independent failure
  evidence but retains a failing process exit.
- Interrupted runs atomically retain a partial receipt. Reruns create new run IDs
  and never overwrite prior admitted evidence.
- Hosted required checks invoke the same profile catalog and command identities;
  drift between local and hosted mappings is a policy failure.

## Platform and Promotion Rules

The harness must work from PowerShell, POSIX shells, and CI without shell-specific
command composition. Profiles declare supported operating systems and required
tools. A preview command may be promoted only after deterministic reproduction,
documented reliability and budget evidence, and an update to
`.github/assurance-controls.json`.
