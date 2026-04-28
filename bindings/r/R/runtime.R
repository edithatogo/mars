load_model_spec <- function(path_or_json) {
  raw <- if (file.exists(path_or_json)) {
    jsonlite::fromJSON(path_or_json, simplifyVector = FALSE)
  } else {
    jsonlite::fromJSON(path_or_json, simplifyVector = FALSE)
  }
  validate_model_spec(raw)
  raw
}

validate_model_spec <- function(spec) {
  if (tryCatch(.validate_model_spec_rust(spec), error = function(error) FALSE)) {
    return(invisible(TRUE))
  }
  .validate_model_spec_pure(spec)
}

design_matrix <- function(spec, rows) {
  if (.rust_runtime_available()) {
    result <- tryCatch(
      .design_matrix_rust(spec, rows),
      error = function(error) NULL
    )
    if (!is.null(result)) {
      return(result)
    }
  }
  .design_matrix_pure(spec, rows)
}

predict_model <- function(spec, rows) {
  if (.rust_runtime_available()) {
    result <- tryCatch(
      .predict_rust(spec, rows),
      error = function(error) NULL
    )
    if (!is.null(result)) {
      return(result)
    }
  }
  .predict_model_pure(spec, rows)
}

.validate_model_spec_rust <- function(spec) {
  if (!.rust_runtime_available()) {
    return(FALSE)
  }
  .invoke_rust_runtime("validate", spec)
  TRUE
}

.design_matrix_rust <- function(spec, rows) {
  .invoke_rust_runtime("design-matrix", spec, rows)
}

.predict_rust <- function(spec, rows) {
  .invoke_rust_runtime("predict", spec, rows)
}

.invoke_rust_runtime <- function(command, spec, rows = NULL) {
  binary <- .rust_runtime_binary()
  if (binary == "") {
    stop("Rust runtime binary is not available", call. = FALSE)
  }

  spec_file <- tempfile(fileext = ".json")
  jsonlite::write_json(
    spec,
    spec_file,
    auto_unbox = TRUE,
    pretty = FALSE,
    digits = NA,
    null = "null"
  )

  args <- c(command, "--spec-file", spec_file)
  if (!is.null(rows)) {
    rows_file <- tempfile(fileext = ".json")
    rows_payload <- lapply(seq_len(nrow(rows)), function(i) as.list(rows[i, ]))
    jsonlite::write_json(
      rows_payload,
      rows_file,
      auto_unbox = TRUE,
      pretty = FALSE,
      digits = NA,
      na = "null",
      null = "null"
    )
    args <- c(args, "--rows-file", rows_file)
  }

  output <- system2(binary, args, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) {
    stop(paste(output, collapse = "\n"), call. = FALSE)
  }
  if (length(output) == 0) {
    return(invisible(TRUE))
  }
  parsed <- jsonlite::fromJSON(paste(output, collapse = "\n"))
  if (command == "design-matrix") {
    return(as.matrix(parsed))
  }
  if (command == "predict") {
    return(as.numeric(parsed))
  }
  parsed
}

.rust_runtime_available <- function() {
  .rust_runtime_binary() != ""
}

.rust_runtime_binary <- function() {
  env_binary <- Sys.getenv("MARS_RUNTIME_BIN", unset = "")
  candidates <- c(
    env_binary,
    file.path("..", "..", "rust-runtime", "target", "debug", "mars-runtime-cli"),
    file.path("..", "..", "rust-runtime", "target", "release", "mars-runtime-cli")
  )
  for (candidate in candidates) {
    if (nzchar(candidate) && file.exists(candidate)) {
      return(normalizePath(candidate))
    }
  }
  ""
}

.validate_model_spec_pure <- function(spec) {
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

.design_matrix_pure <- function(spec, rows) {
  .validate_model_spec_pure(spec)
  rows <- as.matrix(rows)
  expected <- spec$feature_schema$n_features
  if (!is.null(expected) && ncol(rows) != expected) {
    stop("feature-count mismatch", call. = FALSE)
  }
  result <- matrix(0, nrow = nrow(rows), ncol = length(spec$basis_terms))
  for (i in seq_len(nrow(rows))) {
    for (j in seq_along(spec$basis_terms)) {
      result[i, j] <- .evaluate_basis_pure(spec$basis_terms[[j]], rows[i, ])
    }
  }
  result
}

.predict_model_pure <- function(spec, rows) {
  matrix <- .design_matrix_pure(spec, rows)
  coefficients <- unlist(spec$coefficients)
  as.vector(matrix %*% coefficients)
}

.evaluate_basis_pure <- function(basis, row) {
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
    left <- .evaluate_basis_pure(basis$parent1, row)
    right <- .evaluate_basis_pure(basis$parent2, row)
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
