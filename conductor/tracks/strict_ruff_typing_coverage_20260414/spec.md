# Track Specification: Strict Ruff, Strict Typing, >90% Coverage

## Objective
Bring the `mars-earth` codebase to production-grade quality by enforcing strict linting (ruff), strict type checking (ty), and achieving >90% test coverage.

## Scope
- All Python source files in `pymars/`
- All test files in `tests/`
- CI/CD workflows for quality gates
- pyproject.toml configuration

## Out of Scope
- Refactoring algorithm logic (no functional changes unless fixing bugs)
- Breaking API changes (public API must remain compatible)
- Performance optimizations (separate concern)

## Success Criteria
1. **Ruff**: `ruff check pymars/` passes with zero violations (strict rule set)
2. **Ty**: `ty check pymars/` passes with zero errors (strict mode)
3. **Coverage**: `pytest --cov=pymars` reports >90% line coverage
4. **CI**: All quality gate workflows pass on every push

## Acceptance Criteria
- Every public function/method has type annotations for parameters and return type
- No `# type: ignore` without a documented reason comment
- No broad per-file ignores in ruff config (specific rules only)
- All modules individually meet >90% coverage
- Hypothesis property-based tests added for core algorithm invariants

## Risks
- **Type annotation debt**: Large codebase may have significant untyped code
- **False positives**: Some type-checking diagnostics may be upstream type stub issues
- **Coverage gaps**: Some error paths or edge cases may be hard to test

## Dependencies
- Ruff (already installed)
- Ty (already installed)
- Pytest-cov (already installed)
- Hypothesis (already installed)
