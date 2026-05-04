\name{design_matrix}
\alias{design_matrix}
\title{Compute a Design Matrix from a Model Specification}
\usage{design_matrix(spec, rows)}
\arguments{
  \item{spec}{A validated model specification.}
  \item{rows}{A numeric matrix or data frame of input rows.}
}
\value{A numeric matrix of basis-function values.}
\description{Compute the design matrix for a portable mars model specification.}
