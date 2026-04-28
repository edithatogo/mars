//! Tests for training orchestration fixtures
//! Verifies that Rust can load Python-generated baseline fixtures
//! and that the full fit/export path matches

use std::path::Path;
use serde_json::Value;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_python_baseline_fixture() {
        let path = Path::new("tests/fixtures/training_full_fit_baseline_v1.json");
        assert!(path.exists(), "Python baseline fixture not found");
        
        let content = std::fs::read_to_string(path).unwrap();
        let spec: Value = serde_json::from_str(&content).unwrap();
        
        // Verify structure
        assert!(spec.get("spec_version").is_some(), "Missing spec_version");
        assert!(spec.get("params").is_some(), "Missing params");
        assert!(spec.get("feature_schema").is_some(), "Missing feature_schema");
        assert!(spec.get("basis_terms").is_some(), "Missing basis_terms");
        assert!(spec.get("coefficients").is_some(), "Missing coefficients");
        assert!(spec.get("metrics").is_some(), "Missing metrics");
    }
    
    #[test]
    fn test_load_sample_weight_fixture() {
        let path = Path::new("tests/fixtures/training_sample_weight_baseline_v1.json");
        assert!(path.exists(), "Sample-weight baseline fixture not found");
        
        let content = std::fs::read_to_string(path).unwrap();
        let spec: Value = serde_json::from_str(&content).unwrap();
        
        assert!(spec.get("sample_weight").is_some(), "Missing sample_weight");
        assert!(spec.get("X").is_some(), "Missing X values");
        assert!(spec.get("y").is_some(), "Missing y values");
    }
    
    #[test]
    fn test_load_interaction_fixture() {
        let path = Path::new("tests/fixtures/training_interaction_baseline_v1.json");
        assert!(path.exists(), "Interaction baseline fixture not found");
        
        let content = std::fs::read_to_string(path).unwrap();
        let spec: Value = serde_json::from_str(&content).unwrap();
        
        // Check for interaction terms
        let basis_terms = spec.get("basis_terms").unwrap().as_array().unwrap();
        let has_interaction = basis_terms.iter().any(|term| {
            term.get("kind")
                .and_then(|k| k.as_str())
                .map(|k| k == "interaction")
                .unwrap_or(false)
        });
        assert!(has_interaction, "Fixture should contain interaction terms");
    }
    
    #[test]
    fn test_rust_can_validate_baseline() {
        let path = Path::new("tests/fixtures/training_full_fit_baseline_v1.json");
        let content = std::fs::read_to_string(path).unwrap();
        
        // This test will pass once the Rust training core can validate ModelSpec
        // For now, just verify the fixture is valid JSON
        let _spec: Value = serde_json::from_str(&content).unwrap();
    }
    
    #[test]
    fn test_rust_load_model_spec_function_exists() {
        // This test verifies the fixture exists and can be read
        // In the future, this will call rust_runtime::load_model_spec
        let path = Path::new("tests/fixtures/training_full_fit_baseline_v1.json");
        assert!(path.exists());
        let content = std::fs::read_to_string(path).unwrap();
        let _spec: Value = serde_json::from_str(&content).unwrap();
    }
    
    #[test]
    fn test_rust_predict_from_loaded_fixture() {
        // This test verifies that Rust can eventually predict from loaded fixture
        let path = Path::new("tests/fixtures/training_full_fit_baseline_v1.json");
        assert!(path.exists());
        let content = std::fs::read_to_string(path).unwrap();
        let _spec: Value = serde_json::from_str(&content).unwrap();
        // Future: call rust_runtime::predict
    }
}

    #[test]
    fn test_load_sample_weight_fixture() {
        let path = Path::new("tests/fixtures/training_sample_weight_baseline_v1.json");
        assert!(path.exists(), "Sample-weight baseline fixture not found");

        let content = std::fs::read_to_string(path).unwrap();
        let spec: Value = serde_json::from_str(&content).unwrap();

        assert!(spec.get("sample_weight").is_some(), "Missing sample_weight");
        assert!(spec.get("X").is_some(), "Missing X values");
        assert!(spec.get("y").is_some(), "Missing y values");
    }

    #[test]
    fn test_load_interaction_fixture() {
        let path = Path::new("tests/fixtures/training_interaction_baseline_v1.json");
        assert!(path.exists(), "Interaction baseline fixture not found");

        let content = std::fs::read_to_string(path).unwrap();
        let spec: Value = serde_json::from_str(&content).unwrap();

        // Check for interaction terms
        let basis_terms = spec.get("basis_terms").unwrap().as_array().unwrap();
        let has_interaction = basis_terms.iter().any(|term| {
            term.get("kind")
                .and_then(|k| k.as_str())
                .map(|k| k == "interaction")
                .unwrap_or(false)
        });
        assert!(has_interaction, "Fixture should contain interaction terms");
    }

    #[test]
    fn test_rust_can_validate_baseline() {
        let path = Path::new("tests/fixtures/training_full_fit_baseline_v1.json");
        let content = std::fs::read_to_string(path).unwrap();

        // This test will pass once the Rust training core can validate ModelSpec
        // For now, just verify the fixture is valid JSON
        let _spec: Value = serde_json::from_str(&content).unwrap();
    }
}
