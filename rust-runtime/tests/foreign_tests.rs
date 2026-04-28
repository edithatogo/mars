use std::ffi::CString;
use std::fs;
use std::path::PathBuf;
use std::ptr;

use pymars_runtime::foreign::{
    mars_foreign_error_free, mars_foreign_matrix_free, mars_foreign_vector_free,
    mars_model_spec_design_matrix, mars_model_spec_free, mars_model_spec_from_json,
    mars_model_spec_predict, mars_model_spec_validate, MarsForeignError, MarsForeignMatrix,
    MarsForeignStatus, MarsForeignVector, MarsModelSpecHandle,
};
use serde::Deserialize;

#[test]
fn ffi_rejects_malformed_and_unsupported_specs_with_stable_status_codes() {
    let mut handle: *mut MarsModelSpecHandle = ptr::null_mut();
    let mut error = MarsForeignError::default();

    let malformed = CString::new(r#"{"spec_version":"1.0","params":{}"#).expect("valid cstring");
    let status = unsafe { mars_model_spec_from_json(malformed.as_ptr(), &mut handle, &mut error) };
    assert_eq!(status, MarsForeignStatus::MalformedArtifact);
    assert!(handle.is_null());
    assert!(!error.message.is_null());
    unsafe { mars_foreign_error_free(&mut error) };

    let unsupported = CString::new(
        r#"{"spec_version":"2.0","params":{},"feature_schema":{"n_features":1,"feature_names":["x"]},"basis_terms":[{"kind":"constant"}],"coefficients":[1.0]}"#,
    )
    .expect("valid cstring");
    let status =
        unsafe { mars_model_spec_from_json(unsupported.as_ptr(), &mut handle, &mut error) };
    assert_eq!(status, MarsForeignStatus::UnsupportedArtifactVersion);
    assert!(handle.is_null());
    assert!(!error.message.is_null());
    unsafe { mars_foreign_error_free(&mut error) };
}

#[test]
fn ffi_rejects_invalid_utf8_input() {
    let mut handle: *mut MarsModelSpecHandle = ptr::null_mut();
    let mut error = MarsForeignError::default();
    let invalid_utf8 = b"{\"spec_version\":\"1.0\",\"params\":{},\"feature_schema\":{\"n_features\":1,\"feature_names\":[]},\"basis_terms\":[{\"kind\":\"constant\"}],\"coefficients\":[1.0]}\xff\0";

    let status = unsafe {
        mars_model_spec_from_json(invalid_utf8.as_ptr() as *const i8, &mut handle, &mut error)
    };

    assert_eq!(status, MarsForeignStatus::InvalidUtf8);
    assert!(handle.is_null());
    assert!(!error.message.is_null());
    unsafe { mars_foreign_error_free(&mut error) };
}

#[test]
fn ffi_design_matrix_replays_nan_and_reports_feature_mismatch() {
    let spec_json = read_fixture("model_spec_missingness.json");
    let runtime_fixture = read_runtime_fixture("runtime_portability_fixture_missingness.json");

    let mut handle: *mut MarsModelSpecHandle = ptr::null_mut();
    let mut error = MarsForeignError::default();
    let spec = CString::new(spec_json).expect("valid cstring");
    let status = unsafe { mars_model_spec_from_json(spec.as_ptr(), &mut handle, &mut error) };
    assert_eq!(status, MarsForeignStatus::Ok);
    assert!(!handle.is_null());

    let rows = flatten_rows(&runtime_fixture.probe);
    let mut matrix = MarsForeignMatrix::default();
    let status = unsafe {
        mars_model_spec_design_matrix(
            handle,
            rows.as_ptr(),
            runtime_fixture.probe.len(),
            runtime_fixture.probe.first().map_or(0, Vec::len),
            &mut matrix,
            &mut error,
        )
    };
    assert_eq!(status, MarsForeignStatus::Ok);
    assert_eq!(matrix.rows, runtime_fixture.design_matrix.len());
    assert_eq!(
        matrix.cols,
        runtime_fixture.design_matrix.first().map_or(0, Vec::len)
    );
    let actual = unsafe { std::slice::from_raw_parts(matrix.data, matrix.len) };
    let expected = flatten_rows(&runtime_fixture.design_matrix);
    assert_close_matrix(actual, &expected);
    unsafe { mars_foreign_matrix_free(&mut matrix) };

    let mut vector = MarsForeignVector::default();
    let status = unsafe {
        mars_model_spec_predict(
            handle,
            rows.as_ptr(),
            runtime_fixture.probe.len(),
            runtime_fixture.probe.first().map_or(0, Vec::len),
            &mut vector,
            &mut error,
        )
    };
    assert_eq!(status, MarsForeignStatus::Ok);
    let actual = unsafe { std::slice::from_raw_parts(vector.data, vector.len) };
    assert_close_vector(actual, &runtime_fixture.predict);
    unsafe { mars_foreign_vector_free(&mut vector) };

    let bad_rows = vec![1.0_f64, 2.0_f64];
    let status = unsafe {
        mars_model_spec_design_matrix(handle, bad_rows.as_ptr(), 1, 2, &mut matrix, &mut error)
    };
    assert_eq!(status, MarsForeignStatus::FeatureCountMismatch);
    assert!(!error.message.is_null());
    unsafe { mars_foreign_error_free(&mut error) };

    unsafe { mars_model_spec_free(handle) };
}

#[test]
fn ffi_rejects_null_handles() {
    let mut error = MarsForeignError::default();
    let status = unsafe { mars_model_spec_validate(ptr::null(), &mut error) };
    assert_eq!(status, MarsForeignStatus::NullPointer);
    assert!(!error.message.is_null());
    unsafe { mars_foreign_error_free(&mut error) };
}

#[derive(Debug, Deserialize)]
struct RuntimeFixture {
    probe: Vec<Vec<Option<f64>>>,
    design_matrix: Vec<Vec<Option<f64>>>,
    predict: Vec<Option<f64>>,
}

fn read_fixture(name: &str) -> String {
    let path = repo_root().join("tests/fixtures").join(name);
    fs::read_to_string(&path).expect("fixture should exist")
}

fn read_runtime_fixture(name: &str) -> RuntimeFixture {
    let path = repo_root().join("tests/fixtures").join(name);
    serde_json::from_str(&fs::read_to_string(&path).expect("runtime fixture should exist"))
        .expect("runtime fixture should deserialize")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust-runtime should sit under the repository root")
        .to_path_buf()
}

fn flatten_rows(rows: &[Vec<Option<f64>>]) -> Vec<f64> {
    rows.iter()
        .flat_map(|row| row.iter().map(|value| value.unwrap_or(f64::NAN)))
        .collect()
}

fn assert_close_matrix(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        assert_close(*actual_value, *expected_value);
    }
}

fn assert_close_vector(actual: &[f64], expected: &[Option<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        match expected_value {
            Some(expected_value) => assert_close(*actual_value, *expected_value),
            None => assert!(actual_value.is_nan()),
        }
    }
}

fn assert_close(actual: f64, expected: f64) {
    if actual.is_nan() && expected.is_nan() {
        return;
    }
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1e-12,
        "actual={} expected={} delta={}",
        actual,
        expected,
        delta
    );
}
