# Parity Audit Packaging and Docs Summary

This note records the packaging and release conventions that matter for the
`py-earth` / `earth` parity audit slice, with an emphasis on what this repository
does intentionally differently.

## Upstream Baseline

- `py-earth` is documented as a scikit-learn-style Python package with a
  source-install flow, Sphinx docs, and pickling support. It is framed as a
  single Python distribution rather than a multi-registry package family.
- R `earth` is published as an R package on CRAN / r-universe, with package
  metadata, a PDF manual, package docs, and vignette-oriented documentation.
- Source references: [`py-earth` README](https://github.com/scikit-learn-contrib/py-earth),
  [`py-earth` docs](https://contrib.scikit-learn.org/py-earth/content.html),
  [R `earth` manual](https://cran.r-universe.dev/earth/doc/manual.html), and
  [R `earth` package page](https://www.r-pkg.org/pkg/earth).

## Repository Conventions

- The repo publishes the Python distribution as `mars-earth` while keeping the
  `pymars` import name for compatibility.
- Release inventory and release-path docs treat the project as a
  Rust-first, multi-registry package family under the shared `mars-earth`
  brand.
- Registry-specific release paths are documented separately for Python, Rust,
  TypeScript, R, Julia, C#, and Go.

## Generated Manual Expectations

For the R `marsearth` package, the release docs expect:

- per-function Rd pages
- a package-level help topic
- a build-ready vignette source
- `R CMD check --no-manual --as-cran`
- `R CMD Rd2pdf` as the manual build path when the local toolchain supports it
- source-tree and installed-package validation of the same runtime helpers

The checked-in R docs already reflect that structure: `man/marsearth-package.Rd`
describes the package-level help topic, and `bindings/r/vignettes/marsearth.Rmd`
documents the vignette and manual build commands.

## Intentional Release-Boundary Differences

- Upstream `py-earth` is a single Python project; this repo is intentionally a
  multi-language package family with separate registry paths.
- Upstream R `earth` is a CRAN / r-universe package named `earth`; this repo’s
  R surface is `marsearth`, with its own release checklist and publication
  handoff.
- The repository keeps the `mars-earth` brand visible across ecosystems, but the
  concrete package names remain ecosystem-native where required.
- Release readiness is explicit and manual for registry-sensitive surfaces:
  the repo documents rehearsal, artifact inspection, and maintainer approval
  instead of assuming a one-step publish flow.

## Sources

- [docs/package_release_paths.md](package_release_paths.md)
- [docs/release_inventory.md](release_inventory.md)
- [docs/release_checklist.md](release_checklist.md)
- [docs/publication_handoff.md](publication_handoff.md)
- [docs/parity_audit_evidence.md](parity_audit_evidence.md)
- [docs/parity_audit_feature_matrix.md](parity_audit_feature_matrix.md)
- [docs/parity_audit_r_earth_matrix.md](parity_audit_r_earth_matrix.md)
- [docs/parity_audit_repo_gap_matrix.md](parity_audit_repo_gap_matrix.md)
- `bindings/r/README.md`
- `bindings/r/man/marsearth-package.Rd`
- `bindings/r/vignettes/marsearth.Rmd`
- `pyproject.toml`
- `README.md`
