source("R/runtime.R")

fixtures_dir <- "../../tests/fixtures"
model_specs <- sort(list.files(fixtures_dir, pattern = "^model_spec_.*\\.json$"))
stopifnot(length(model_specs) > 0)

assert_vector_close <- function(actual, expected) {
  stopifnot(length(actual) == length(expected))
  for (idx in seq_along(actual)) {
    if (is.na(expected[[idx]])) {
      stopifnot(is.nan(actual[[idx]]) || is.na(actual[[idx]]))
    } else {
      stopifnot(abs(actual[[idx]] - expected[[idx]]) <= 1e-12)
    }
  }
}

assert_matrix_close <- function(actual, expected) {
  stopifnot(nrow(actual) == length(expected))
  for (row_idx in seq_len(nrow(actual))) {
    assert_vector_close(actual[row_idx, ], as.numeric(unlist(lapply(expected[[row_idx]], function(value) {
      if (is.null(value)) NaN else value
    }))))
  }
}

for (file in model_specs) {
  suffix <- sub("\\.json$", "", sub("^model_spec_", "", file))
  spec <- load_model_spec(file.path(fixtures_dir, file))
  fixture <- jsonlite::fromJSON(
    file.path(fixtures_dir, paste0("runtime_portability_fixture_", suffix, ".json")),
    simplifyVector = FALSE
  )
  probe <- do.call(rbind, lapply(fixture$probe, function(row) {
    as.numeric(unlist(lapply(row, function(value) if (is.null(value)) NaN else value)))
  }))
  actual_design_matrix <- design_matrix(spec, probe)
  assert_matrix_close(actual_design_matrix, fixture$design_matrix)
  actual_predictions <- predict_model(spec, probe)
  expected_predictions <- as.numeric(unlist(lapply(fixture$predict, function(value) {
    if (is.null(value)) NaN else value
  })))
  assert_vector_close(actual_predictions, expected_predictions)
}

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
assert_vector_close(replay, c(1, 3, 5))
