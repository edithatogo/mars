use std::path::Path;
use std::{env, fs, thread};

use crate::errors::{MarsError, MarsResult};
use crate::model_spec::{BasisTermSpec, ModelSpec};
use crate::observability;
use rayon::prelude::*;

pub fn load_model_spec_str(raw: &str) -> MarsResult<ModelSpec> {
    let _span = observability::span("runtime::load_model_spec_str");
    let spec: ModelSpec = serde_json::from_str(raw).map_err(|error| {
        MarsError::MalformedArtifact(format!("failed to deserialize model spec JSON: {error}"))
    })?;
    validate_model_spec(&spec)?;
    Ok(spec)
}

pub fn load_model_spec_path(path: impl AsRef<Path>) -> MarsResult<ModelSpec> {
    let _span = observability::span("runtime::load_model_spec_path");
    let path_ref = path.as_ref();
    let raw = fs::read_to_string(path_ref).map_err(|error| {
        MarsError::MalformedArtifact(format!(
            "failed to read model spec from {}: {error}",
            path_ref.display()
        ))
    })?;
    load_model_spec_str(&raw)
}

pub fn load_model_spec_json_or_path(path_or_json: impl AsRef<str>) -> MarsResult<ModelSpec> {
    let _span = observability::span("runtime::load_model_spec_json_or_path");
    let raw = path_or_json.as_ref();
    let trimmed = raw.trim_start();

    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return load_model_spec_str(raw);
    }

    load_model_spec_path(Path::new(raw))
}

pub fn validate_model_spec(spec: &ModelSpec) -> MarsResult<()> {
    let _span = observability::span("runtime::validate_model_spec");
    let mut parts = spec.spec_version.split('.');
    let major = parts.next().ok_or_else(|| {
        MarsError::MissingRequiredField("spec_version must contain a major version".to_string())
    })?;
    let minor = parts.next().ok_or_else(|| {
        MarsError::MissingRequiredField("spec_version must contain a minor version".to_string())
    })?;
    if parts.next().is_some() || major.is_empty() || minor.is_empty() {
        return Err(MarsError::MalformedArtifact(
            "spec_version must be in '<major>.<minor>' format".to_string(),
        ));
    }
    if major != "1" {
        return Err(MarsError::UnsupportedArtifactVersion(format!(
            "unsupported model spec major version: {}",
            spec.spec_version
        )));
    }
    if !spec.params.is_object() {
        return Err(MarsError::MalformedArtifact(
            "params must be a JSON object".to_string(),
        ));
    }
    if spec.basis_terms.len() != spec.coefficients.len() {
        return Err(MarsError::MalformedArtifact(
            "coefficients length must match basis_terms length".to_string(),
        ));
    }
    if let Some(n_features) = spec.feature_schema.n_features {
        if !spec.feature_schema.feature_names.is_empty()
            && spec.feature_schema.feature_names.len() != n_features
        {
            return Err(MarsError::MalformedArtifact(
                "feature_names length must match n_features when both are present".to_string(),
            ));
        }
    }

    for (idx, basis) in spec.basis_terms.iter().enumerate() {
        if basis.kind.trim().is_empty() {
            return Err(MarsError::MissingRequiredField(format!(
                "basis term {idx} has an empty kind"
            )));
        }
        match basis.kind.as_str() {
            "constant" => {}
            "linear" => {
                let variable_idx = basis.variable_idx.ok_or_else(|| {
                    MarsError::MissingRequiredField(
                        "linear basis terms require variable_idx".to_string(),
                    )
                })?;
                validate_variable_idx(spec, variable_idx, idx)?;
            }
            "hinge" => {
                let variable_idx = basis.variable_idx.ok_or_else(|| {
                    MarsError::MissingRequiredField(
                        "hinge basis terms require variable_idx".to_string(),
                    )
                })?;
                validate_variable_idx(spec, variable_idx, idx)?;
                if basis.knot_val.is_none() {
                    return Err(MarsError::MissingRequiredField(
                        "hinge basis terms require knot_val".to_string(),
                    ));
                }
                if basis.is_right_hinge.is_none() {
                    return Err(MarsError::MissingRequiredField(
                        "hinge basis terms require is_right_hinge".to_string(),
                    ));
                }
            }
            "categorical" => {
                let variable_idx = basis.variable_idx.ok_or_else(|| {
                    MarsError::MissingRequiredField(
                        "categorical basis terms require variable_idx".to_string(),
                    )
                })?;
                validate_variable_idx(spec, variable_idx, idx)?;
                numeric_category_value(basis)?;
            }
            "interaction" => {
                if basis.parent1.is_none() {
                    return Err(MarsError::MissingRequiredField(
                        "interaction basis terms require parent1".to_string(),
                    ));
                }
                if basis.parent2.is_none() {
                    return Err(MarsError::MissingRequiredField(
                        "interaction basis terms require parent2".to_string(),
                    ));
                }
            }
            "missingness" => {
                let variable_idx = basis.variable_idx.ok_or_else(|| {
                    MarsError::MissingRequiredField(
                        "missingness basis terms require variable_idx".to_string(),
                    )
                })?;
                validate_variable_idx(spec, variable_idx, idx)?;
            }
            unsupported => {
                return Err(MarsError::UnsupportedBasisTerm(format!(
                    "unsupported basis kind '{unsupported}'"
                )));
            }
        }
    }

    Ok(())
}

pub fn design_matrix(spec: &ModelSpec, rows: &[Vec<f64>]) -> MarsResult<Vec<Vec<f64>>> {
    let _span = observability::span("runtime::design_matrix");
    validate_model_spec(spec)?;
    validate_rows(spec, rows)?;
    let thread_count = runtime_thread_count();

    if should_use_parallel(thread_count, rows.len()) {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|error| {
                MarsError::MalformedArtifact(format!("failed to configure thread pool: {error}"))
            })?;

        thread_pool.install(|| {
            rows.par_iter()
                .map(|row| evaluate_design_row(spec, row))
                .collect()
        })
    } else {
        rows.iter()
            .map(|row| evaluate_design_row(spec, row))
            .collect()
    }
}

pub fn predict(spec: &ModelSpec, rows: &[Vec<f64>]) -> MarsResult<Vec<f64>> {
    let _span = observability::span("runtime::predict");
    let thread_count = runtime_thread_count();
    validate_model_spec(spec)?;
    validate_rows(spec, rows)?;

    if should_use_parallel(thread_count, rows.len()) {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|error| {
                MarsError::MalformedArtifact(format!("failed to configure thread pool: {error}"))
            })?;

        thread_pool.install(|| rows.par_iter().map(|row| predict_row(spec, row)).collect())
    } else {
        rows.iter().map(|row| predict_row(spec, row)).collect()
    }
}

fn evaluate_design_row(spec: &ModelSpec, row: &[f64]) -> MarsResult<Vec<f64>> {
    spec.basis_terms
        .iter()
        .map(|basis| evaluate_basis(basis, row))
        .collect()
}

fn predict_row(spec: &ModelSpec, row: &[f64]) -> MarsResult<f64> {
    spec.basis_terms
        .iter()
        .zip(spec.coefficients.iter())
        .try_fold(0.0, |total, (basis, coefficient)| {
            evaluate_basis(basis, row).map(|value| total + value * coefficient)
        })
}

fn validate_rows(spec: &ModelSpec, rows: &[Vec<f64>]) -> MarsResult<()> {
    if let Some(n_features) = spec.feature_schema.n_features {
        for (row_idx, row) in rows.iter().enumerate() {
            if row.len() != n_features {
                return Err(MarsError::FeatureCountMismatch {
                    row_index: row_idx,
                    actual: row.len(),
                    expected: n_features,
                });
            }
        }
    }
    Ok(())
}

fn validate_variable_idx(
    spec: &ModelSpec,
    variable_idx: usize,
    basis_idx: usize,
) -> MarsResult<()> {
    if let Some(n_features) = spec.feature_schema.n_features {
        if variable_idx >= n_features {
            return Err(MarsError::MalformedArtifact(format!(
                "basis term {} references variable_idx {} outside n_features {}",
                basis_idx, variable_idx, n_features
            )));
        }
    }
    Ok(())
}

pub fn evaluate_basis(basis: &BasisTermSpec, row: &[f64]) -> MarsResult<f64> {
    match basis.kind.as_str() {
        "constant" => Ok(1.0),
        "linear" => {
            let variable_idx = basis.variable_idx.ok_or_else(|| {
                MarsError::MissingRequiredField(
                    "linear basis terms require variable_idx".to_string(),
                )
            })?;
            let value = row[variable_idx];
            let value = if let Some(parent) = basis.parent1.as_deref() {
                let parent_value = evaluate_basis(parent, row)?;
                if value.is_nan() || parent_value.is_nan() {
                    f64::NAN
                } else {
                    value * parent_value
                }
            } else {
                value
            };
            Ok(value)
        }
        "hinge" => {
            let variable_idx = basis.variable_idx.ok_or_else(|| {
                MarsError::MissingRequiredField(
                    "hinge basis terms require variable_idx".to_string(),
                )
            })?;
            let knot_val = basis.knot_val.ok_or_else(|| {
                MarsError::MissingRequiredField("hinge basis terms require knot_val".to_string())
            })?;
            let value = row[variable_idx];
            let value = if basis.is_right_hinge.unwrap_or(true) {
                (value - knot_val).max(0.0)
            } else {
                (knot_val - value).max(0.0)
            };
            let value = if let Some(parent) = basis.parent1.as_deref() {
                let parent_value = evaluate_basis(parent, row)?;
                if value.is_nan() || parent_value.is_nan() {
                    f64::NAN
                } else {
                    value * parent_value
                }
            } else {
                value
            };
            Ok(value)
        }
        "categorical" => {
            let variable_idx = basis.variable_idx.ok_or_else(|| {
                MarsError::MissingRequiredField(
                    "categorical basis terms require variable_idx".to_string(),
                )
            })?;
            let category = numeric_category_value(basis)?;
            let value = row[variable_idx];
            let value = if value.is_nan() {
                f64::NAN
            } else if value == category {
                1.0
            } else {
                0.0
            };
            let value = if let Some(parent) = basis.parent1.as_deref() {
                let parent_value = evaluate_basis(parent, row)?;
                if value.is_nan() || parent_value.is_nan() {
                    f64::NAN
                } else {
                    value * parent_value
                }
            } else {
                value
            };
            Ok(value)
        }
        "interaction" => {
            let left = evaluate_basis(
                basis.parent1.as_deref().ok_or_else(|| {
                    MarsError::MissingRequiredField(
                        "interaction basis terms require parent1".to_string(),
                    )
                })?,
                row,
            )?;
            let right = evaluate_basis(
                basis.parent2.as_deref().ok_or_else(|| {
                    MarsError::MissingRequiredField(
                        "interaction basis terms require parent2".to_string(),
                    )
                })?,
                row,
            )?;
            if left.is_nan() || right.is_nan() {
                Ok(f64::NAN)
            } else {
                Ok(left * right)
            }
        }
        "missingness" => {
            let variable_idx = basis.variable_idx.ok_or_else(|| {
                MarsError::MissingRequiredField(
                    "missingness basis terms require variable_idx".to_string(),
                )
            })?;
            let value: f64 = if row[variable_idx].is_nan() { 1.0 } else { 0.0 };
            let value = if let Some(parent) = basis.parent1.as_deref() {
                let parent_value = evaluate_basis(parent, row)?;
                if value.is_nan() || parent_value.is_nan() {
                    f64::NAN
                } else {
                    value * parent_value
                }
            } else {
                value
            };
            Ok(value)
        }
        unsupported => Err(MarsError::UnsupportedBasisTerm(format!(
            "unsupported basis kind '{unsupported}'"
        ))),
    }
}

fn numeric_category_value(basis: &BasisTermSpec) -> MarsResult<f64> {
    let category = basis.category.as_ref().ok_or_else(|| {
        MarsError::MissingRequiredField("categorical basis terms require category".to_string())
    })?;
    category.as_f64().ok_or_else(|| {
        MarsError::InvalidCategoricalEncoding(
            "categorical basis terms currently require a numeric category".to_string(),
        )
    })
}

const DEFAULT_THREAD_HINT: usize = 1;
const THREAD_ENV_VAR: &str = "MARS_EARTH_RUNTIME_THREADS";

fn runtime_thread_count() -> usize {
    let configured = env::var(THREAD_ENV_VAR)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_THREAD_HINT);

    configured.min(available_threads())
}

fn should_use_parallel(thread_count: usize, rows: usize) -> bool {
    thread_count > 1 && rows > 1
}

fn available_threads() -> usize {
    thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::{design_matrix, load_model_spec_path, predict, runtime_thread_count};
    use std::env;
    use std::path::PathBuf;

    #[test]
    fn runtime_thread_count_defaults_to_single_when_unset() {
        let _guard = ThreadOverrideGuard::new(None);
        assert_eq!(runtime_thread_count(), 1);
    }

    #[test]
    fn runtime_thread_count_clamps_to_env_request() {
        let _guard = ThreadOverrideGuard::new(Some("1"));
        assert_eq!(runtime_thread_count(), 1);
    }

    #[test]
    fn design_matrix_uses_default_single_thread_mode() {
        let spec = load_model_spec_fixture();
        let rows = vec![
            vec![0.0, 0.0, 0.1],
            vec![0.5, 0.25, 0.2],
            vec![1.0, 0.75, 0.3],
        ];

        let _guard = ThreadOverrideGuard::new(Some("1"));
        let matrix = design_matrix(&spec, &rows).expect("design matrix should compute");
        let single = predict(&spec, &rows).expect("prediction should compute");
        assert_eq!(matrix.len(), 3);
        assert_eq!(single.len(), 3);
    }

    #[test]
    fn design_matrix_prediction_matches_across_thread_modes() {
        let spec = load_model_spec_fixture();
        let rows = vec![
            vec![0.0, 0.0, 0.1],
            vec![0.5, 0.25, 0.2],
            vec![1.0, 0.75, 0.3],
            vec![1.5, 1.0, 0.4],
        ];

        let _single = ThreadOverrideGuard::new(Some("1"));
        let single = predict(&spec, &rows).expect("single-thread prediction should compute");

        let _parallel = ThreadOverrideGuard::new(Some("2"));
        let parallel = predict(&spec, &rows).expect("parallel prediction should compute");

        assert_eq!(single.len(), parallel.len());
        assert!(single
            .iter()
            .zip(parallel.iter())
            .all(|(left, right)| (left - right).abs() < 1e-12));
    }

    #[test]
    fn set_thread_override_defaults_to_single_when_invalid() {
        let _clear_guard = ThreadOverrideGuard::new(None);
        let _invalid_guard = ThreadOverrideGuard::new(Some("not-a-number"));
        let threads = runtime_thread_count();
        assert_eq!(threads, 1);
    }

    fn load_model_spec_fixture() -> crate::model_spec::ModelSpec {
        load_model_spec_path(repo_root().join("tests/fixtures/model_spec_v1.json"))
            .expect("fixture should load")
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("rust-runtime should sit under the repository root")
            .to_path_buf()
    }

    struct ThreadOverrideGuard {
        previous: Option<String>,
    }

    impl ThreadOverrideGuard {
        fn new(value: Option<&'static str>) -> Self {
            let previous = env::var(super::THREAD_ENV_VAR).ok();
            if let Some(raw) = value {
                env::set_var(super::THREAD_ENV_VAR, raw);
            } else {
                env::remove_var(super::THREAD_ENV_VAR);
            }
            ThreadOverrideGuard { previous }
        }
    }

    impl Drop for ThreadOverrideGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                env::set_var(super::THREAD_ENV_VAR, previous);
            } else {
                env::remove_var(super::THREAD_ENV_VAR);
            }
        }
    }
}
