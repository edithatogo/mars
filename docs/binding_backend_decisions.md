# Binding Backend Decisions

The binding backends intentionally mix host-native ergonomics with a shared
Rust runtime boundary.

Selected mechanisms:

- Python: PyO3 and `maturin`
- Rust: shared runtime crate and native extension boundary
- TypeScript: runtime wrapper with Rust CLI acceleration
- C#: .NET wrapper with trusted publishing to NuGet
- Go: runtime wrapper around the shared CLI surface
- R and Julia: host-language wrappers around the shared runtime contract
