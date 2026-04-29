source("R/runtime.R")

fitted_spec <- fit_model(
  x = matrix(c(0, 1, 2), ncol = 1),
  y = c(1, 3, 5),
  max_terms = 5,
  max_degree = 1,
  penalty = 3.0
)
stopifnot(is.list(fitted_spec))
stopifnot(length(fitted_spec$basis_terms) > 0)
replay <- predict_model(fitted_spec, matrix(c(0, 1, 2), ncol = 1))
stopifnot(length(replay) == 3)
stopifnot(all(abs(replay - c(1, 3, 5)) <= 1e-12))
