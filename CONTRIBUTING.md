# Contributing to pymars

Thank you for your interest in improving **pymars**! The project is a pure Python adaptation of the `py-earth` library and must remain fully compatible with scikit-learn. This document summarizes key guidelines from `AGENTS.md`, how to run tests, and the recommended workflow for feature development or bug fixes.

## Coding Standards

- **Pure Python only.** No compiled extensions are accepted.
- Focus on correctness and clarity first; optimize Python code only when needed.
- **scikit-learn compatibility** is mandatory. Estimators should:
  - inherit from `sklearn.base.BaseEstimator` (and `RegressorMixin` or `ClassifierMixin` as appropriate),
  - implement `fit`, `predict`, `score`, `get_params`, and `set_params`,
  - validate inputs with `sklearn.utils.validation.check_X_y` and `check_array`,
  - return `self` from `fit`,
  - store learned attributes with trailing underscores (e.g., `coef_`).
- Keep the class structure close to `py-earth`. The main model class is `Earth` in `pymars/earth.py` and should be usable as:
  ```python
  import pymars as earth
  model = earth.Earth()
  ```
- Follow **PEP 8** and use type hints. Document modules, classes, functions, and methods with clear documentation explaining *why* things are done.
- Run a linter such as `ruff` to catch style issues.
- Write comments that focus on reasoning and motivation rather than line-by-line explanations.
- Aim for minimal dependencies (typically NumPy and SciPy). List them in `requirements.txt` and `pyproject.toml`.

## Running Tests

- Tests reside in the `tests/` directory and use `pytest`.
- **Test-Driven Development** is encouraged. Provide tests for new features, edge cases, and error handling. Include scikit-learn compatibility checks (for example with `sklearn.utils.estimator_checks.check_estimator`).
- Install dependencies and run the test suite with:
  ```bash
  pip install -r requirements.txt && pytest
  ```

## CI Parity

Run the same checks locally when practical before opening a pull request:

```bash
uv run pytest tests/ -q --tb=short -x --cov-fail-under=80
uv run pytest -q tests/test_binding_conformance.py tests/test_model_spec.py
uv run pytest -q tests/test_python_routing.py tests/test_rust_compatibility.py tests/test_model_spec.py
uv run ruff check pymars tests
uv run ruff format --check pymars tests
uv run ty check pymars/
uv run mkdocs build --strict
vale --config .vale.ini README.md docs/
cargo fmt --check --manifest-path rust-runtime/Cargo.toml
cargo test --manifest-path rust-runtime/Cargo.toml --test fixture_tests --test foreign_tests --test training_fixture_tests
cargo clippy --manifest-path rust-runtime/Cargo.toml -- -D warnings
go test ./...
npm test --prefix bindings/typescript
Rscript tests/conformance.R
julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
dotnet test bindings/csharp/MarsRuntime.Tests/MarsRuntime.Tests.csproj
```

## Failure Triage

- If a workflow fails, rerun the matching local command and compare it with the
  diagnostic artifacts uploaded by CI.
- For code-quality failures, inspect the `code-quality-diagnostics` artifact.
- For release rehearsal failures, inspect the uploaded package lists and smoke
  test output in the release artifacts.
- For dependency or security failures, review dependency-review, Bandit, Safety,
  and `cargo audit` output before changing code.
- If the failure is isolated to one binding, rerun the binding-specific smoke
  test first and only touch the shared Rust runtime if the issue reproduces
  there.

## Workflow for New Features or Bug Fixes

1. Review `ROADMAP.md` and `TODO.md` before starting a task. Keep them updated and log significant decisions in `SESSION_LOGS.md`.
2. Create a feature branch from `main`.
3. Make small, atomic commits with clear messages (e.g., `feat: add hinge basis function`, `fix: resolve pruning error`).
4. Keep code and tests in the same branch and ensure `pytest` passes locally.
5. Update documentation (`ROADMAP.md`, `TODO.md`, `SESSION_LOGS.md`, or user docs) as needed.
6. Open a pull request for review when the feature or fix is complete and tested.

Following these guidelines helps keep the project consistent and maintainable. Happy coding!
