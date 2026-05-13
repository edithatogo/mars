# Bindings

This repo keeps the release family under the `mars-earth` brand while
preserving ecosystem-native import and namespace behavior where required.

Current targets:

- Python
- Rust
- TypeScript
- Go
- R
- Julia
- C#

## Documentation Entry Points

| Binding | Primary docs surface | Typical first stop |
| --- | --- | --- |
| Python | Tutorials, examples, usage, API reference | [Examples](examples/index.md) |
| Rust | Runtime and ABI docs | [Rust Core](rust_core.md) |
| TypeScript | README and conformance tests | `bindings/typescript/README.md` |
| Go | README and conformance tests | `bindings/go/README.md` |
| R | README, Rd pages, vignette | `bindings/r/README.md` and `bindings/r/vignettes/marsearth.Rmd` |
| Julia | README and package tests | `bindings/julia/README.md` |
| C# | README and smoke tests | `bindings/csharp/README.md` |

The shared binding conformance harness lives in `bindings/conformance/README.md`.
The example hub collects the notebook and binding quickstarts in one place.

See [Package Release Paths](package_release_paths.md) for the registry-specific
release policy.
