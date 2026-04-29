# CI, Quality, and Automation Policy

This page records the checks that gate merges and releases while the project
finishes the Rust-core binding rollout.

## Required Checks

These checks are intended to block merges on pull requests:

- Python tests: `uv run pytest tests/ -q --tb=short -x`
- Python coverage floor: `uv run pytest tests/ -q --tb=short -x --cov-fail-under=80`
- Python binding conformance: `uv run pytest -q tests/test_binding_conformance.py tests/test_model_spec.py`
- Python lint and type checks: `uv run ruff check pymars tests`, `uv run ruff format --check pymars tests`, `uv run ty check pymars/`
- Python docs build: `uv run mkdocs build --strict`
- Rust format and tests: `cargo fmt --check`, `cargo test --test fixture_tests --test foreign_tests --test training_fixture_tests`
- Rust package verification: `cargo package --allow-dirty`
- Go tests: `go test ./...`
- Go training conformance: `go test -run TestFitModelViaRust ./...`
- Go temp-consumer smoke test: create a temporary module that imports `github.com/edithatogo/mars/bindings/go` via a local replace and run `go test`
- TypeScript tests and package dry-run: `npm test`, `npm pack --dry-run`
- TypeScript unsupported-training check: `npm test`
- TypeScript packed-artifact smoke test: `npm pack`, `npm install --prefix ...`
- R package build and conformance: `R CMD build bindings/r`, `Rscript tests/conformance.R`
- R training conformance: `Rscript tests/conformance.R`
- R packed-artifact smoke test: `R CMD INSTALL --library=... marsruntime_*.tar.gz`
- Julia conformance: `julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`
- Julia training conformance: `julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`
- C# tests and package build: `dotnet test bindings/csharp/MarsRuntime.Tests/MarsRuntime.Tests.csproj`, `dotnet pack bindings/csharp/MarsRuntime.csproj -c Release`
- C# training conformance: `dotnet test bindings/csharp/MarsRuntime.Tests/MarsRuntime.Tests.csproj`
- C# packed-artifact smoke test: `dotnet restore ... --source bindings/csharp/bin/Release`, `dotnet build ...`

## Advisory Checks

These checks are useful but may remain scheduled or release-only where the
tooling is heavier:

- Rust clippy
- Dependency audit tools for Cargo, npm, and .NET ecosystems
- Built-artifact install smoke tests for each package manager
- Release provenance and attestation checks
- Mutation testing with mutmut
- Profiling with Scalene and py-spy
- Rust dependency audit with `cargo audit`
These run on scheduled or manual workflows rather than blocking every pull
request.
The code-quality workflow also uploads diagnostic logs for the Python tests,
ruff, ty, and mkdocs checks so maintainers can compare local and CI output
quickly.

The repo-level supply-chain policy lives in [Supply Chain Security](supply_chain.md).
It records Renovate as the dependency automation source of truth, sets the
secret-scanning expectation, and describes the provenance attestation policy
for release and rehearsal artifacts.

## Workflow Reliability

- CI uses concurrency cancellation for stale pushes and pull requests.
- The release rehearsal workflow also cancels stale runs on the same ref.
- Security workflows run dependency review on pull requests.
- Security workflows also run Cargo, Python, and release-path supply-chain
  checks.
- Release workflows run under a protected environment.
- The Rust runtime CLI is built before host-language runtime conformance jobs.
- Rust cache is used for the shared Rust build job.
- The Rust runtime CLI also runs in a Linux, macOS, and Windows validation
  matrix.
- Retention days are set on uploaded artifacts so diagnostics stay available
  long enough to inspect release and security failures.

## Local Parity

The workflow commands above are the source of truth for local validation. When
adding a new check, update this page and the workflow in the same change.
