# MarsRuntime

Portable .NET runtime replay for mars `ModelSpec` artifacts.

This package loads validated model specifications and evaluates design matrices
and predictions on .NET 11. It is part of the binding-first roadmap: language
surfaces share the same fixture contract before the training core migrates to
Rust.

## Validation

```sh
dotnet test MarsRuntime.Tests/MarsRuntime.Tests.csproj
```
