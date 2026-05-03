# CI and Quality Policy

This repo enforces a conservative quality gate:

- Python tests and coverage thresholds
- `ruff`, `ty`, and `codespell`
- Vale prose lint
- `mkdocs --strict`
- release rehearsal and artifact inspection
- package alignment checks so docs and manifests stay in sync

See [Supply Chain Security](supply_chain.md) for provenance and release
automation policy.
