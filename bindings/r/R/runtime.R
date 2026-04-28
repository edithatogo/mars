load_model_spec <- function(path_or_json) {
  if (file.exists(path_or_json)) {
    spec <- jsonlite::fromJSON(path_or_json, simplifyVector = FALSE)
  } else {
    spec <- jsonlite::fromJSON(path_or_json, simplifyVector = FALSE)
  }
  validate_model_spec(spec)
  spec
}

validate_model_spec <- function(spec) {
  if (is.null(spec$spec_version) || !grepl("^[0-9]+\\.[0-9]+$", spec$spec_version)) {
    stop("malformed artifact: spec_version must be '<major>.<minor>'", call. = FALSE)
  }
  if (!startsWith(spec$spec_version, "1.")) {
    stop(sprintf("unsupported artifact version: %s", spec$spec_version), call. = FALSE)
  }
  if (length(spec$basis_terms) != length(spec$coefficients)) {
    stop("malformed artifact: coefficients length must match basis_terms", call. = FALSE)
  }
  invisible(TRUE)
}

design_matrix <- function(spec, rows) {
  validate_model_spec(spec)
  rows <- as.matrix(rows)
  expected <- spec$feature_schema$n_features
  if (!is.null(expected) && ncol(rows) != expected) {
    stop("feature-count mismatch", call. = FALSE)
  }
  result <- matrix(0, nrow = nrow(rows), ncol = length(spec$basis_terms))
  for (i in seq_len(nrow(rows))) {
    for (j in seq_along(spec$basis_terms)) {
      result[i, j] <- evaluate_basis(spec$basis_terms[[j]], rows[i, ])
    }
  }
  result
}

predict_model <- function(spec, rows) {
  as.vector(design_matrix(spec, rows) %*% unlist(spec$coefficients))
}

evaluate_basis <- function(basis, row) {
  kind <- basis$kind
  if (kind == "constant") {
    return(1.0)
  }
  if (kind == "linear") {
    return(row[[basis$variable_idx + 1]])
  }
  if (kind == "hinge") {
    value <- row[[basis$variable_idx + 1]]
    if (isTRUE(basis$is_right_hinge)) {
      return(max(value - basis$knot_val, 0))
    }
    return(max(basis$knot_val - value, 0))
  }
  if (kind == "categorical") {
    value <- row[[basis$variable_idx + 1]]
    if (is.nan(value)) {
      return(NaN)
    }
    return(ifelse(value == basis$category, 1.0, 0.0))
  }
  if (kind == "interaction") {
    left <- evaluate_basis(basis$parent1, row)
    right <- evaluate_basis(basis$parent2, row)
    if (is.nan(left) || is.nan(right)) {
      return(NaN)
    }
    return(left * right)
  }
  if (kind == "missingness") {
    return(ifelse(is.nan(row[[basis$variable_idx + 1]]), 1.0, 0.0))
  }
  stop(sprintf("unsupported basis term: %s", kind), call. = FALSE)
}
