use std::fs;

use serde::Deserialize;

use pymars_runtime::{fit_least_squares, score_candidate, score_pruning_subset};

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

fn load_fixture() -> TrainingFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/training_core_fixture_v1.json"
    );
    serde_json::from_str(&fs::read_to_string(path).expect("fixture should exist"))
        .expect("fixture should deserialize")
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
