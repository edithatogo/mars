use std::fs;

use serde::Deserialize;

use pymars_runtime::{
    fit_least_squares, fit_model, forward_pass, model_spec_from_terms, predict, prune_model,
    score_candidate, score_pruning_subset, validate_model_spec, TrainingParams, TrainingRequest,
};

#[derive(Debug, Deserialize)]
struct TrainingFixture {
    design: Vec<Vec<f64>>,
    y: Vec<f64>,
    sample_weight: Vec<f64>,
    candidate_columns: Vec<Vec<f64>>,
    pruning_columns: Vec<usize>,
    penalty: f64,
    num_hinge_terms: usize,
}

#[derive(Debug, Deserialize)]
struct FitFixture {
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    sample_weight: Vec<f64>,
    params: serde_json::Value,
}

#[test]
fn weighted_least_squares_matches_python_baseline_fixture() {
    let fixture = load_fixture();

    let fit = fit_least_squares(
        &fixture.design,
        &fixture.y,
        Some(&fixture.sample_weight),
        true,
    )
    .expect("weighted fit should succeed");

    assert_close(fit.effective_n_samples, 19.0);
    assert_close(fit.rss, 8.790020790020785);
    assert_vector_close(&fit.coefficients, &[4.08108108108108, 2.521829521829523]);
}

#[test]
fn candidate_and_pruning_scoring_are_fixture_backed() {
    let fixture = load_fixture();

    let candidate = score_candidate(
        &fixture.design,
        &fixture.candidate_columns,
        &fixture.y,
        Some(&fixture.sample_weight),
        fixture.penalty,
        fixture.num_hinge_terms,
    )
    .expect("candidate scoring should succeed");
    assert_close(candidate.effective_n_samples, 19.0);
    assert_close(candidate.rss, 1.4638233054074643);
    assert_close(candidate.gcv, 0.16457185090379775);

    let subset = score_pruning_subset(
        &fixture.design,
        &fixture.pruning_columns,
        &fixture.y,
        Some(&fixture.sample_weight),
        fixture.penalty,
        fixture.num_hinge_terms,
    )
    .expect("pruning subset scoring should succeed");
    assert_close(subset.rss, 8.790020790020785);
    assert_close(subset.gcv, 0.8520938520938515);
}

#[test]
fn forward_pass_and_pruning_match_linear_baseline_fixture() {
    let fixture = load_fit_fixture();
    let params = TrainingParams {
        max_terms: fixture
            .params
            .get("max_terms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(5) as usize,
        max_degree: fixture
            .params
            .get("max_degree")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize,
        penalty: fixture
            .params
            .get("penalty")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(3.0),
        minspan: fixture
            .params
            .get("minspan")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        endspan: fixture
            .params
            .get("endspan")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        threshold: fixture
            .params
            .get("threshold")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.001),
        allow_linear: fixture
            .params
            .get("allow_linear")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        allow_missing: fixture
            .params
            .get("allow_missing")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        categorical_features: fixture
            .params
            .get("categorical_features")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|value| value as usize)
                    .collect()
            }),
        feature_names: Some(vec!["x0".to_string()]),
    };

    let forward = forward_pass(
        &fixture.x,
        &fixture.y,
        Some(&fixture.sample_weight),
        &params,
    )
    .expect("forward pass should succeed");
    assert_eq!(forward.basis_terms.len(), 2);
    assert_eq!(forward.basis_terms[0].kind, "constant");
    assert_eq!(forward.basis_terms[1].kind, "linear");
    assert_close(forward.coefficients[0], 1.0);
    assert_close(forward.coefficients[1], 2.0);
    assert_close(forward.rss, 0.0);

    let pruned = prune_model(
        &fixture.x,
        &fixture.y,
        Some(&fixture.sample_weight),
        &forward.basis_terms,
        &forward.coefficients,
        &params,
    )
    .expect("pruning should succeed");
    assert_eq!(pruned.basis_terms.len(), 2);
    assert_eq!(pruned.basis_terms[1].kind, "linear");
    assert_close(pruned.coefficients[0], 1.0);
    assert_close(pruned.coefficients[1], 2.0);

    let spec = model_spec_from_terms(&pruned.basis_terms, &pruned.coefficients, &params);
    assert_eq!(spec.basis_terms.len(), 2);
    assert_eq!(spec.coefficients.len(), 2);
}

#[test]
fn fit_model_replays_linear_baseline_fixture_end_to_end() {
    let fixture = load_fit_fixture();
    let request = TrainingRequest {
        x: fixture.x.clone(),
        y: fixture.y.clone(),
        sample_weight: Some(fixture.sample_weight.clone()),
        params: TrainingParams {
            max_terms: fixture
                .params
                .get("max_terms")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(5) as usize,
            max_degree: fixture
                .params
                .get("max_degree")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1) as usize,
            penalty: fixture
                .params
                .get("penalty")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(3.0),
            minspan: fixture
                .params
                .get("minspan")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
            endspan: fixture
                .params
                .get("endspan")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
            threshold: fixture
                .params
                .get("threshold")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.001),
            allow_linear: fixture
                .params
                .get("allow_linear")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            allow_missing: fixture
                .params
                .get("allow_missing")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            categorical_features: fixture
                .params
                .get("categorical_features")
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(serde_json::Value::as_u64)
                        .map(|value| value as usize)
                        .collect()
                }),
            feature_names: Some(vec!["x0".to_string()]),
        },
    };

    let spec = fit_model(&request).expect("fit_model should succeed");
    validate_model_spec(&spec).expect("fit_model spec should validate");
    let predictions = predict(&spec, &request.x).expect("fit_model spec should replay");
    assert_vector_close(&predictions, &request.y);
}

#[test]
fn forward_pass_supports_numeric_interactions_and_exported_specs_replay() {
    let x = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 1.0],
    ];
    let y = vec![1.0, 1.0, 1.0, 2.0, 3.0];
    let params = TrainingParams {
        max_terms: 10,
        max_degree: 2,
        penalty: 0.0,
        minspan: 0.0,
        endspan: 0.0,
        threshold: 0.001,
        allow_linear: false,
        allow_missing: false,
        categorical_features: Some(vec![]),
        feature_names: Some(vec!["x0".to_string(), "x1".to_string()]),
    };

    let forward =
        forward_pass(&x, &y, None, &params).expect("interaction forward pass should succeed");
    assert!(
        forward
            .basis_terms
            .iter()
            .any(|term| term.kind == "interaction"),
        "expected at least one interaction term"
    );

    assert_close(forward.rss, 0.0);

    let spec = model_spec_from_terms(&forward.basis_terms, &forward.coefficients, &params);
    validate_model_spec(&spec).expect("exported interaction spec should validate");

    let predictions = predict(&spec, &x).expect("exported interaction spec should replay");
    assert_vector_close(&predictions, &y);
}

#[test]
fn forward_pass_supports_categorical_terms_when_linear_terms_are_disabled() {
    let x = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0]];
    let y = vec![2.0, 5.0, 2.0, 5.0, 2.0];
    let params = TrainingParams {
        max_terms: 5,
        max_degree: 1,
        penalty: 0.0,
        minspan: 0.0,
        endspan: 0.0,
        threshold: 0.001,
        allow_linear: false,
        allow_missing: false,
        categorical_features: Some(vec![0]),
        feature_names: Some(vec!["category".to_string()]),
    };

    let forward =
        forward_pass(&x, &y, None, &params).expect("categorical forward pass should succeed");
    assert!(
        forward
            .basis_terms
            .iter()
            .any(|term| term.kind == "categorical"),
        "expected categorical candidate term to be selected"
    );
    let spec = model_spec_from_terms(&forward.basis_terms, &forward.coefficients, &params);
    validate_model_spec(&spec).expect("categorical spec should validate");
    let predictions = predict(&spec, &x).expect("categorical spec should replay");
    assert_vector_close(&predictions, &y);
}

#[test]
fn forward_pass_supports_missingness_terms_when_enabled() {
    let x = vec![
        vec![0.0],
        vec![f64::NAN],
        vec![0.0],
        vec![f64::NAN],
        vec![0.0],
    ];
    let y = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let params = TrainingParams {
        max_terms: 4,
        max_degree: 1,
        penalty: 0.0,
        minspan: 0.0,
        endspan: 0.0,
        threshold: 0.001,
        allow_linear: false,
        allow_missing: true,
        categorical_features: None,
        feature_names: Some(vec!["x0".to_string()]),
    };

    let forward =
        forward_pass(&x, &y, None, &params).expect("missingness forward pass should succeed");
    assert!(
        forward
            .basis_terms
            .iter()
            .any(|term| term.kind == "missingness"),
        "expected missingness candidate term to be selected"
    );
    let spec = model_spec_from_terms(&forward.basis_terms, &forward.coefficients, &params);
    validate_model_spec(&spec).expect("missingness spec should validate");
    let predictions = predict(&spec, &x).expect("missingness spec should replay");
    assert_vector_close(&predictions, &y);
}

fn load_fixture() -> TrainingFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/training_core_fixture_v1.json"
    );
    serde_json::from_str(&fs::read_to_string(path).expect("fixture should exist"))
        .expect("fixture should deserialize")
}

fn load_fit_fixture() -> FitFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/training_sample_weight_baseline_v1.json"
    );
    serde_json::from_str(&fs::read_to_string(path).expect("fit fixture should exist"))
        .expect("fit fixture should deserialize")
}

fn assert_vector_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        assert_close(*actual_value, *expected_value);
    }
}

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1e-10,
        "actual={} expected={} delta={}",
        actual,
        expected,
        delta
    );
}
