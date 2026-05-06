//! Rust runtime and future core boundary for `mars`.
//!
//! This crate currently owns portable `ModelSpec` validation, basis-matrix
//! evaluation, and prediction replay. It is intentionally small, fixture-backed,
//! and independent of Python object reconstruction. The crate should grow into
//! the shared Rust computational core used by Python, R, Julia, Rust, C#, Go,
//! and TypeScript bindings.
//!
//! The public replay API is:
//!
//! - [`load_model_spec_str`]
//! - [`load_model_spec_path`]
//! - [`validate_model_spec`]
//! - [`design_matrix`]
//! - [`predict`]
//!
//! `ModelSpec` artifact semantics are defined by the repository documentation,
//! not by host-language wrapper behavior.

pub mod errors;
pub mod foreign;
pub mod model_spec;
pub mod python;
pub mod runtime;
pub mod training;

pub(crate) mod observability {
    use std::env;
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();

    pub(crate) fn span(name: &'static str) -> SpanGuard {
        let active = enabled();
        if active {
            eprintln!("trace: enter {name}");
        }
        SpanGuard { name, active }
    }

    fn enabled() -> bool {
        *ENABLED.get_or_init(|| {
            env::var_os("RUST_LOG").is_some() || env::var_os("MARS_RUNTIME_TRACE").is_some()
        })
    }

    pub(crate) struct SpanGuard {
        name: &'static str,
        active: bool,
    }

    impl Drop for SpanGuard {
        fn drop(&mut self) {
            if self.active {
                eprintln!("trace: exit {}", self.name);
            }
        }
    }
}

pub use errors::{MarsError, MarsResult};
pub use model_spec::{BasisTermSpec, FeatureSchema, ModelSpec};
pub use runtime::{
    design_matrix, evaluate_basis, load_model_spec_path, load_model_spec_str, predict,
    validate_model_spec,
};
pub use training::{
    calculate_gcv, effective_parameters, fit_least_squares, fit_model, forward_pass,
    model_spec_from_terms, prune_model, score_candidate, score_pruning_subset, CandidateScore,
    ForwardPassResult, LeastSquaresFit, PruningResult, TrainingParams, TrainingRequest,
};
