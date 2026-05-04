args <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(script_file) > 0) {
  dirname(normalizePath(script_file[1]))
} else {
  getwd()
}

runtime_source <- file.path(script_dir, "..", "R", "runtime.R")
if (file.exists(runtime_source)) {
  source(runtime_source)
} else {
  library(marsruntime)
}

if (Sys.getenv("MARS_RUNTIME_BIN", unset = "") == "") {
  local_binary <- file.path(script_dir, "..", "..", "..", "rust-runtime", "target", "debug", "mars-runtime-cli")
  if (file.exists(local_binary)) {
    Sys.setenv(MARS_RUNTIME_BIN = normalizePath(local_binary))
  }
}

rust_runtime_available <- function() {
  if (exists(".rust_runtime_available", mode = "function")) {
    return(.rust_runtime_available())
  }
  getFromNamespace(".rust_runtime_available", "marsruntime")()
}

if (!rust_runtime_available()) {
  message("Skipping training conformance: Rust runtime binary is not available")
  quit(status = 0)
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
stopifnot(length(replay) == 3)
stopifnot(all(abs(replay - c(1, 3, 5)) <= 1e-12))
