# Baseline Measurement Snapshot

Date: 2026-04-15

This snapshot records the current quality baseline for Phase 0 of the
`Strict Ruff, Strict Typing, >90% Coverage` track.

## Commands

- `uv run ruff check pymars tests --output-format=concise`
- `uv run mypy pymars/`
- `uv run pytest --cov=pymars --cov-report=term-missing`

## Ruff

- Total violations: 52
- Fixable violations reported by Ruff: 12
- Hidden fixes available: 40
- Top files by violations:
  - `pymars/_basis.py`: 22
  - `pymars/_forward.py`: 14
  - `pymars/_record.py`: 6
  - `pymars/explain.py`: 6
  - `pymars/_missing.py`: 1
  - `pymars/_sklearn_compat.py`: 1
  - `pymars/cli.py`: 1
  - `pymars/earth.py`: 1

### Current Ruff Configuration

Enabled rule groups in `pyproject.toml`:

- `E`
- `W`
- `F`
- `I`
- `C`
- `B`
- `UP`
- `YTT`
- `BLE`
- `S`
- `C9`
- `SIM`
- `ARG`
- `PERF`
- `RUF`
- `PTH`
- `FLY`
- `RET`

Currently ignored rule groups and codes:

- `ANN`
- `S101`
- `E501`
- `C901`
- `RUF013`
- `RUF012`
- `RUF059`
- `ARG`
- `PTH`
- `BLE001`
- `RET`
- `UP006`
- `UP035`
- `PERF`
- `SIM102`
- `SIM108`
- `SIM101`
- `RUF043`
- `RUF015`
- `RUF046`
- `E701`
- `F841`
- `F821`
- `E402`
- `B904`
- `S301`
- `W293`
- `B007`
- `E702`

## Mypy

- Total errors: 130
- Files with the most errors:
  - `pymars/earth.py`: 35
  - `pymars/_forward.py`: 20
  - `pymars/_pruning.py`: 19
  - `pymars/_basis.py`: 13
  - `pymars/_sklearn_compat.py`: 10
  - `pymars/explain.py`: 7
  - `pymars/_categorical.py`: 5
  - `pymars/_missing.py`: 4
  - `pymars/_record.py`: 4
  - `pymars/cli.py`: 4

## Coverage

- Overall coverage: `66.81%`
- Test outcome: `157 passed`, `3 skipped`
- Coverage gate result: failed against `fail_under = 80`

## Notes

- `mypy` emitted a configuration warning because `mypy.ini` sets
  `python_version = 3.8`, which is below the version supported by the installed
  toolchain in this environment.
- `pytest` emitted unknown-mark warnings for `unit`, `integration`, `e2e`,
  and `golden` during this run because the active config was `pytest.ini`.
