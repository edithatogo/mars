\name{fit_model}
\alias{fit_model}
\title{Fit a Portable Mars Model}
\usage{
fit_model(
  x,
  y,
  max_terms = 21,
  max_degree = 1,
  penalty = 3.0,
  minspan = 0.0,
  endspan = 0.0,
  threshold = 0.001,
  allow_linear = TRUE,
  allow_missing = FALSE,
  categorical_features = integer(),
  feature_names = NULL,
  sample_weight = NULL
)
}
\arguments{
  \item{x}{Training feature matrix.}
  \item{y}{Training response vector.}
  \item{max_terms}{Maximum number of basis terms to retain.}
  \item{max_degree}{Maximum interaction degree.}
  \item{penalty}{Complexity penalty used during pruning.}
  \item{minspan}{Minimum span between knots.}
  \item{endspan}{End-span constraint for knot placement.}
  \item{threshold}{Stopping threshold for forward selection.}
  \item{allow_linear}{Whether to include linear terms.}
  \item{allow_missing}{Whether to include missing-value terms.}
  \item{categorical_features}{One-based indexes of categorical columns.}
  \item{feature_names}{Optional feature names for the exported spec.}
  \item{sample_weight}{Optional sample weights for the training rows.}
}
\value{A validated portable model specification.}
\description{Fit a mars model using the shared runtime bridge.}
