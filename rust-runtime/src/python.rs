use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::{
    design_matrix, load_model_spec_json_or_path, load_model_spec_str, predict, validate_model_spec,
};
use crate::training::{fit_model, TrainingRequest};

#[pyfunction]
fn load_model_spec_canonical_json(path_or_json: &str) -> PyResult<String> {
    let spec = load_model_spec_json_or_path(path_or_json).map_err(map_error)?;
    serde_json::to_string(&spec).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn load_model_spec_path_json(path: &str) -> PyResult<String> {
    let spec = crate::runtime::load_model_spec_path(Path::new(path)).map_err(map_error)?;
    serde_json::to_string(&spec).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn validate_model_spec_json(spec_json: &str) -> PyResult<()> {
    let spec = load_model_spec_str(spec_json).map_err(map_error)?;
    validate_model_spec(&spec).map_err(map_error)?;
    Ok(())
}

#[pyfunction]
fn design_matrix_json(spec_json: &str, rows: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let spec = load_model_spec_str(spec_json).map_err(map_error)?;
    design_matrix(&spec, &rows).map_err(map_error)
}

#[pyfunction]
fn predict_json(spec_json: &str, rows: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let spec = load_model_spec_str(spec_json).map_err(map_error)?;
    predict(&spec, &rows).map_err(map_error)
}

#[pyfunction]
fn inspect_model_spec_json(spec_json: &str) -> PyResult<String> {
    let spec: serde_json::Value = serde_json::from_str(spec_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let summary = serde_json::json!({
        "spec_version": spec.get("spec_version").cloned().unwrap_or(serde_json::Value::Null),
        "model_type": spec.get("model_type").cloned().unwrap_or(serde_json::Value::Null),
        "n_features": spec
            .get("feature_schema")
            .and_then(|feature_schema| feature_schema.get("n_features"))
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        "n_basis_terms": spec
            .get("basis_terms")
            .and_then(|basis_terms| basis_terms.as_array())
            .map_or(0, |basis_terms| basis_terms.len()),
        "metrics": spec.get("metrics").cloned().unwrap_or_else(|| serde_json::json!({})),
    });
    serde_json::to_string(&summary).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn export_model_json(spec_json: &str) -> PyResult<String> {
    let spec = load_model_spec_str(spec_json).map_err(map_error)?;
    serde_json::to_string_pretty(&spec).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn fit_model_json(request_json: &str) -> PyResult<String> {
    let request: TrainingRequest = serde_json::from_str(request_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let spec = fit_model(&request).map_err(map_error)?;
    serde_json::to_string(&spec).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn pymars_runtime(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(load_model_spec_canonical_json, module)?)?;
    module.add_function(wrap_pyfunction!(load_model_spec_path_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_model_spec_json, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_json, module)?)?;
    module.add_function(wrap_pyfunction!(predict_json, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_model_spec_json, module)?)?;
    module.add_function(wrap_pyfunction!(export_model_json, module)?)?;
    module.add_function(wrap_pyfunction!(fit_model_json, module)?)?;
    module.add("_IS_COMPILED", true)?;
    Ok(())
}

fn map_error(error: crate::errors::MarsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::export_model_json;
    use super::inspect_model_spec_json;
    use super::{load_model_spec_canonical_json, load_model_spec_path_json};
    use crate::model_spec::ModelSpec;
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn inspect_model_spec_json_returns_expected_summary() {
        let spec_json = read_fixture("model_spec_v1.json");
        let summary_json = inspect_model_spec_json(&spec_json).expect("inspection should succeed");
        let summary: Value =
            serde_json::from_str(&summary_json).expect("summary should parse as JSON");

        assert_eq!(summary["spec_version"], "1.0");
        assert_eq!(summary["model_type"], "Earth");
        assert_eq!(summary["n_features"], 3);
        assert_eq!(summary["n_basis_terms"], 5);
        assert!(summary["metrics"].is_object());
    }

    #[test]
    fn load_model_spec_canonical_json_returns_canonical_json() {
        let spec_json = read_fixture("model_spec_v1.json");
        let expected =
            crate::runtime::load_model_spec_str(&spec_json).expect("loading should succeed");
        let canonical_json =
            load_model_spec_canonical_json(&spec_json).expect("loading should succeed");
        let loaded: ModelSpec =
            serde_json::from_str(&canonical_json).expect("canonical spec should parse");

        assert_eq!(loaded, expected);
    }

    #[test]
    fn load_model_spec_path_json_returns_canonical_json() {
        let path = repo_root().join("tests/fixtures/model_spec_v1.json");
        let expected = crate::runtime::load_model_spec_path(&path).expect("loading should succeed");
        let canonical_json =
            load_model_spec_path_json(path.to_str().expect("fixture path should be valid UTF-8"))
                .expect("loading should succeed");
        let loaded: ModelSpec =
            serde_json::from_str(&canonical_json).expect("canonical spec should parse");

        assert_eq!(loaded, expected);
    }

    #[test]
    fn export_model_json_returns_pretty_json() {
        let spec_json = read_fixture("model_spec_v1.json");
        let expected =
            crate::runtime::load_model_spec_str(&spec_json).expect("loading should succeed");
        let exported_json = export_model_json(&spec_json).expect("export should succeed");
        let exported: ModelSpec =
            serde_json::from_str(&exported_json).expect("exported spec should parse");

        assert_eq!(exported, expected);
    }

    fn read_fixture(name: &str) -> String {
        let path = repo_root().join("tests/fixtures").join(name);
        fs::read_to_string(&path).expect("fixture should exist")
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("rust-runtime should sit under the repository root")
            .to_path_buf()
    }
}
