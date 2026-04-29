# Binding ABI and API Contract

The Rust-backed binding track uses a single compatibility contract across all
host languages.

## Ownership Rules

- The Rust core owns the shared computational semantics.
- Host-language wrappers own local parsing, native data conversion, and package
  lifecycle management.
- Exposed buffers, handles, and strings must have a documented release path.
- Wrappers must not require callers to manage internal Rust allocation details
  directly unless the binding API documents that ownership explicitly.

## Null, NaN, and Missingness Rules

- Numeric missing values are represented as `NaN` at the Rust boundary.
- Host-language `null`/`None` values should be converted to `NaN` where the
  binding supports numeric replay semantics.
- Categorical values must follow the documented encoding for the relevant
  contract version.
- Unsupported or ambiguous missingness cases must fail with a category-stable
  error instead of guessing.

## Error Rules

- Shared errors are categorized using the existing `MarsError` categories.
- Host languages may map those categories to idiomatic exceptions, but the
  category name must remain visible in the message or error metadata.
- New error categories are breaking changes unless introduced through a versioned
  contract update.

## SemVer Rules

- Breaking ABI/API changes require a major version bump.
- Non-breaking additions should preserve existing function names, error
  categories, and fixture behavior.
- Host-language wrappers may version independently, but they must not claim
  compatibility beyond the shared Rust contract they actually implement.

## Binding Validation Rules

- All bindings must validate `ModelSpec` before evaluation.
- Runtime-only packages must reject training operations with a stable
  unsupported-feature error until training support is implemented.
- Fixture conformance must run at the source-package level and, where feasible,
  on built artifacts.
