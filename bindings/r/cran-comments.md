## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local macOS aarch64-apple-darwin25.4.0, R 4.6.0
* `R CMD check --no-manual --as-cran marsearth_0.0.0.tar.gz`

## Notes

This submission supersedes earlier same-day uploads under the package names
`marsruntime` and `mars.earth`. The intended R package name is `marsearth`,
because `mars-earth` is not a valid R package name and `marsearth` aligns with
the Julia `MarsEarth` package name.

The package includes portable replay tests using packaged JSON fixtures. The
training helper is optional and is exercised when the companion runtime binary
is available through `MARS_RUNTIME_BIN`; otherwise that external-binary path is
skipped during checks.

## Downstream dependencies

There are currently no downstream dependencies for this package.
