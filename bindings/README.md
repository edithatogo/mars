# Runtime Bindings

The binding-first roadmap validates portable `ModelSpec` replay across language
surfaces before migrating training internals into Rust.

Current binding status:

| Binding | Status | Local validation |
| --- | --- | --- |
| Python | Existing compatibility baseline | `uv run pytest -q tests/test_model_spec.py` |
| Rust | Runtime replay prototype | `cd rust-runtime && cargo test` |
| Go | MVP runtime replay binding | `cd bindings/go && go test ./...` |
| TypeScript | MVP runtime replay binding | `cd bindings/typescript && npm test` |
| R | Source package surface | `cd bindings/r && Rscript tests/conformance.R` with `jsonlite` |
| Julia | Source package surface | `julia --project=bindings/julia bindings/julia/test/runtests.jl` with `JSON` |
| C# | Source package surface targeting .NET 11 preview | `cd bindings/csharp && dotnet test MarsRuntime.Tests/MarsRuntime.Tests.csproj` |

All supported bindings should consume `bindings/conformance/manifest.json` or
the same fixture pairs from `tests/fixtures`.

## Package Publication Targets

| Binding | Package manager | Release path |
| --- | --- | --- |
| Python | PyPI | Existing `mars-earth` distribution |
| Rust | crates.io | Publish `pymars-runtime` once crate ownership and API stability are confirmed |
| TypeScript | npm | Publish `@mars-earth/runtime` |
| R | CRAN or r-universe | Start with r-universe, graduate to CRAN when API and checks are mature |
| Julia | Julia General registry | Register `MarsRuntime.jl` after package UUID and API are stable |
| C# | NuGet | Publish `MarsRuntime` package |
| Go | Go modules | Publish from repository tags under the Go module path |

Publishing must be gated by CI, registry credentials, version checks, and an
explicit release approval. The initial publish workflow is intentionally manual
or tag-gated; it should not publish on ordinary pull requests.
