use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::{design_matrix, load_model_spec_str, predict, validate_model_spec};
use crate::training::{fit_model, TrainingRequest};

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
fn fit_model_json(request_json: &str) -> PyResult<String> {
    let request: TrainingRequest = serde_json::from_str(request_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let spec = fit_model(&request).map_err(map_error)?;
    serde_json::to_string(&spec).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn pymars_runtime(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_model_spec_json, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_json, module)?)?;
    module.add_function(wrap_pyfunction!(predict_json, module)?)?;
    module.add_function(wrap_pyfunction!(fit_model_json, module)?)?;
    module.add("_IS_COMPILED", true)?;
    Ok(())
}

fn map_error(error: crate::errors::MarsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
