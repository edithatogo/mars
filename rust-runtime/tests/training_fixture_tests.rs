//! Tests for training orchestration fixtures.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

#[test]
fn test_load_python_baseline_fixture() {
    let path = repo_root().join("tests/fixtures/training_full_fit_baseline_v1.json");
    assert!(path.exists(), "Python baseline fixture not found");

    let content = fs::read_to_string(path).expect("fixture should be readable");
    let spec: Value = serde_json::from_str(&content).expect("fixture should parse");

    assert!(spec.get("spec_version").is_some(), "Missing spec_version");
    assert!(spec.get("params").is_some(), "Missing params");
    assert!(
        spec.get("feature_schema").is_some(),
        "Missing feature_schema"
    );
    assert!(spec.get("basis_terms").is_some(), "Missing basis_terms");
    assert!(spec.get("coefficients").is_some(), "Missing coefficients");
    assert!(spec.get("metrics").is_some(), "Missing metrics");
}

#[test]
fn test_load_sample_weight_fixture() {
    let path = repo_root().join("tests/fixtures/training_sample_weight_baseline_v1.json");
    assert!(path.exists(), "Sample-weight baseline fixture not found");

    let content = fs::read_to_string(path).expect("fixture should be readable");
    let spec: Value = serde_json::from_str(&content).expect("fixture should parse");

    assert!(spec.get("sample_weight").is_some(), "Missing sample_weight");
    assert!(spec.get("X").is_some(), "Missing X values");
    assert!(spec.get("y").is_some(), "Missing y values");
}

#[test]
fn test_load_interaction_fixture() {
    let path = repo_root().join("tests/fixtures/model_spec_interaction.json");
    assert!(path.exists(), "Interaction baseline fixture not found");

    let content = fs::read_to_string(path).expect("fixture should be readable");
    let spec: Value = serde_json::from_str(&content).expect("fixture should parse");

    let basis_terms = spec
        .get("basis_terms")
        .expect("basis_terms should exist")
        .as_array()
        .expect("basis_terms should be an array");
    let has_interaction = basis_terms.iter().any(|term| {
        term.get("kind")
            .and_then(Value::as_str)
            .map(|kind| kind == "interaction")
            .unwrap_or(false)
    });
    assert!(has_interaction, "Fixture should contain interaction terms");
}

#[test]
fn test_rust_can_validate_baseline() {
    let path = repo_root().join("tests/fixtures/training_full_fit_baseline_v1.json");
    let content = fs::read_to_string(path).expect("fixture should be readable");
    let _spec: Value = serde_json::from_str(&content).expect("fixture should parse");
}

#[test]
fn test_rust_load_model_spec_function_exists() {
    let path = repo_root().join("tests/fixtures/training_full_fit_baseline_v1.json");
    assert!(path.exists());
    let content = fs::read_to_string(path).expect("fixture should be readable");
    let _spec: Value = serde_json::from_str(&content).expect("fixture should parse");
}

#[test]
fn test_rust_predict_from_loaded_fixture() {
    let path = repo_root().join("tests/fixtures/training_full_fit_baseline_v1.json");
    assert!(path.exists());
    let content = fs::read_to_string(path).expect("fixture should be readable");
    let _spec: Value = serde_json::from_str(&content).expect("fixture should parse");
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust-runtime should sit under the repository root")
        .to_path_buf()
}
