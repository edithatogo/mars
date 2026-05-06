\name{marsruntime-package}
\alias{marsruntime}
\alias{marsruntime-package}
\docType{package}
\title{marsruntime: Portable R Runtime Replay for Mars}
\description{
The `marsruntime` package exposes a portable R runtime surface for validated
Mars `ModelSpec` artifacts. It can load, validate, replay, and, when the Rust
runtime helper is available, fit portable model specifications.
}
\details{
The package is designed to work both from the source tree and from an installed
package. Runtime replay helpers fall back to pure R logic when the Rust runtime
binary is unavailable. Training via `fit_model()` requires the Rust runtime
helper and returns a portable `ModelSpec` that can be replayed through the same
helpers.
}
\section{Quick Start}{
Load a validated `ModelSpec`, inspect its design matrix, and generate
predictions with the same portable runtime surface:
\preformatted{
library(marsruntime)
spec <- load_model_spec("path/to/model_spec.json")
design_matrix(spec, matrix(c(0, 1, 2), ncol = 1))
predict_model(spec, matrix(c(0, 1, 2), ncol = 1))
}
}
\section{Training}{
If the Rust runtime helper is present, `fit_model()` can fit a portable
specification from feature and response data:
\preformatted{
if (nzchar(Sys.getenv("MARS_RUNTIME_BIN", unset = ""))) {
  fitted <- fit_model(matrix(c(0, 1, 2), ncol = 1), c(1, 3, 5))
  predict_model(fitted, matrix(c(0, 1, 2), ncol = 1))
}
}
}
\section{Validation}{
The package includes conformance tests that replay shared fixtures and verify
the runtime-backed training path when the Rust binary is available.
}
\section{Documentation}{
Per-function help is available for `load_model_spec()`, `validate_model_spec()`,
`design_matrix()`, `predict_model()`, and `fit_model()`. The release docs
describe the R publication path, package-check expectations, and the manual
build path.
}
\examples{
spec <- list(
  spec_version = "1.0",
  basis_terms = list(),
  coefficients = list(),
  feature_schema = list(n_features = 0)
)
validate_model_spec(spec)
}
\keyword{package}
