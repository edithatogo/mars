# Native/Python boundary validation — 29 October 2026

Base: d68b54a1c9a5eee40e1b7a724e002d5083ff6d8f. Scope: optional PyO3
feature, maturin opt-in and dependency guard. No numerical algorithm changed.
Toolchains: CPython 3.15.1, rustc/cargo 1.99.1, maturin 1.15.0.

- Red: default dependency tree contained PyO3 before the feature split.
- Green: locked/offline normal + build dependency guard reports no PyO3.
- Regression: a real temporary Cargo graph with a build-only local pyo3 crate
  is rejected; one unittest passed.
- Native and `--features python`: fixture_tests (1), foreign_tests (6),
  training_fixture_tests (6), all passed in both modes. Two build jobs;
  dev/test debug info disabled, target under `.conductor/local/scientific-alignment`.
- `maturin build --locked --offline --interpreter .venv/bin/python` succeeded.
- Wheel installed with no dependencies into a separate Python 3.15.1 environment
  outside the checkout. Isolated import verified `_IS_COMPILED` and site-packages
  origin. v1 fixture predictions matched at relative/absolute tolerance 1e-9.
  First probe used the incorrect fixture key `predictions`; corrected to the
  actual `predict` key and rerun successfully. No product change was needed.
- Agent correctness/security review: optional split sound; fixed missing build
  dependency coverage and documentation that confused tested Python with ABI floor.

The abi3-py39 floor is preserved for this bounded split. Whole-project Python
3.15 migration, full installed package suite, cross-platform CI and measured
resource receipts remain open. These targeted tests do not close the track.

## Python migration candidate follow-up

PR235's initial hosted audit found90 advisories in10 packages. Updated the active
Python package/CI/tool defaults to3.15.1 and refreshed the lock. Locked dev install
passed; six binding/boundary tests passed; pip-audit of that actual environment
reported no known vulnerabilities. Fixed new-script Ruff findings and retained
justified shell-free metadata subprocess calls. Hosted rerun remains required.
The initial native/installed fixture receipts above remain scoped to their stated
inputs; package migration is not declared complete by these focused checks.
