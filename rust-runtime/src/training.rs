use serde::Deserialize;

use crate::errors::{MarsError, MarsResult};
use crate::model_spec::{BasisTermSpec, FeatureSchema, ModelSpec};
use crate::runtime::evaluate_basis;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TrainingParams {
    pub max_terms: usize,
    pub max_degree: usize,
    pub penalty: f64,
    pub minspan: f64,
    pub endspan: f64,
    pub threshold: f64,
    pub allow_linear: bool,
    pub allow_missing: bool,
    #[serde(default)]
    pub categorical_features: Option<Vec<usize>>,
    pub feature_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TrainingRequest {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    #[serde(default)]
    pub sample_weight: Option<Vec<f64>>,
    pub params: TrainingParams,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            max_terms: 21,
            max_degree: 1,
            penalty: 3.0,
            minspan: 0.0,
            endspan: 0.0,
            threshold: 0.001,
            allow_linear: true,
            allow_missing: false,
            categorical_features: None,
            feature_names: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresFit {
    pub rss: f64,
    pub coefficients: Vec<f64>,
    pub effective_n_samples: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateScore {
    pub rss: f64,
    pub gcv: f64,
    pub coefficients: Vec<f64>,
    pub effective_n_samples: f64,
}

pub fn effective_parameters(num_terms: usize, num_hinge_terms: usize, penalty: f64) -> f64 {
    if num_terms == 0 {
        0.0
    } else {
        num_terms as f64 + penalty * num_hinge_terms as f64
    }
}

pub fn calculate_gcv(rss: f64, num_samples: f64, num_effective_params: f64) -> f64 {
    if num_samples <= 0.0 || num_effective_params >= num_samples {
        return f64::INFINITY;
    }
    let denominator = (1.0 - num_effective_params / num_samples).powi(2);
    if denominator < 1e-9 {
        f64::INFINITY
    } else {
        rss / (num_samples * denominator)
    }
}

pub fn fit_least_squares(
    design: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    drop_nan_rows: bool,
) -> MarsResult<LeastSquaresFit> {
    if design.len() != y.len() {
        return Err(MarsError::MalformedArtifact(
            "design row count must match y length".to_string(),
        ));
    }
    if let Some(weights) = sample_weight {
        if weights.len() != y.len() {
            return Err(MarsError::MalformedArtifact(
                "sample_weight length must match y length".to_string(),
            ));
        }
    }

    let n_terms = design.first().map_or(0, Vec::len);
    if n_terms == 0 {
        return fit_intercept_only(y, sample_weight);
    }

    let mut xtx = vec![vec![0.0; n_terms]; n_terms];
    let mut xty = vec![0.0; n_terms];
    let mut effective_n_samples = 0.0;

    for (row_idx, row) in design.iter().enumerate() {
        if row.len() != n_terms {
            return Err(MarsError::MalformedArtifact(
                "all design rows must have equal length".to_string(),
            ));
        }
        if drop_nan_rows && row.iter().any(|value| value.is_nan()) {
            continue;
        }
        let weight = sample_weight.map_or(1.0, |weights| weights[row_idx]);
        if weight == 0.0 {
            continue;
        }
        let row_values: Vec<f64> = if drop_nan_rows {
            row.clone()
        } else {
            row.iter()
                .map(|value| if value.is_nan() { 0.0 } else { *value })
                .collect()
        };
        effective_n_samples += weight;
        for col in 0..n_terms {
            xty[col] += weight * row_values[col] * y[row_idx];
            for other in 0..n_terms {
                xtx[col][other] += weight * row_values[col] * row_values[other];
            }
        }
    }

    if effective_n_samples == 0.0 {
        return Err(MarsError::NumericalEvaluationFailure(
            "no valid rows for least-squares fit".to_string(),
        ));
    }

    let coefficients = solve_linear_system(xtx, xty)?;
    let mut rss = 0.0;
    for (row_idx, row) in design.iter().enumerate() {
        if drop_nan_rows && row.iter().any(|value| value.is_nan()) {
            continue;
        }
        let weight = sample_weight.map_or(1.0, |weights| weights[row_idx]);
        if weight == 0.0 {
            continue;
        }
        let prediction = row
            .iter()
            .zip(coefficients.iter())
            .map(|(value, coefficient)| {
                let value = if value.is_nan() && !drop_nan_rows {
                    0.0
                } else {
                    *value
                };
                value * coefficient
            })
            .sum::<f64>();
        rss += weight * (y[row_idx] - prediction).powi(2);
    }

    Ok(LeastSquaresFit {
        rss,
        coefficients,
        effective_n_samples,
    })
}

pub fn score_candidate(
    design: &[Vec<f64>],
    candidate_columns: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    penalty: f64,
    num_hinge_terms: usize,
) -> MarsResult<CandidateScore> {
    let combined = append_columns(design, candidate_columns)?;
    let fit = fit_least_squares(&combined, y, sample_weight, true)?;
    let effective = effective_parameters(fit.coefficients.len(), num_hinge_terms, penalty);
    Ok(CandidateScore {
        gcv: calculate_gcv(fit.rss, fit.effective_n_samples, effective),
        rss: fit.rss,
        coefficients: fit.coefficients,
        effective_n_samples: fit.effective_n_samples,
    })
}

pub fn score_pruning_subset(
    design: &[Vec<f64>],
    columns: &[usize],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    penalty: f64,
    num_hinge_terms: usize,
) -> MarsResult<CandidateScore> {
    let subset = select_columns(design, columns)?;
    let fit = fit_least_squares(&subset, y, sample_weight, true)?;
    let effective = effective_parameters(fit.coefficients.len(), num_hinge_terms, penalty);
    Ok(CandidateScore {
        gcv: calculate_gcv(fit.rss, fit.effective_n_samples, effective),
        rss: fit.rss,
        coefficients: fit.coefficients,
        effective_n_samples: fit.effective_n_samples,
    })
}

fn fit_intercept_only(y: &[f64], sample_weight: Option<&[f64]>) -> MarsResult<LeastSquaresFit> {
    if y.is_empty() {
        return Err(MarsError::NumericalEvaluationFailure(
            "cannot fit intercept for empty y".to_string(),
        ));
    }
    let (sum_weight, weighted_sum) =
        y.iter()
            .enumerate()
            .fold((0.0, 0.0), |(sum_weight, weighted_sum), (idx, value)| {
                let weight = sample_weight.map_or(1.0, |weights| weights[idx]);
                (sum_weight + weight, weighted_sum + weight * value)
            });
    if sum_weight == 0.0 {
        return Err(MarsError::NumericalEvaluationFailure(
            "sample weights sum to zero".to_string(),
        ));
    }
    let intercept = weighted_sum / sum_weight;
    let rss = y
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let weight = sample_weight.map_or(1.0, |weights| weights[idx]);
            weight * (value - intercept).powi(2)
        })
        .sum();
    Ok(LeastSquaresFit {
        rss,
        coefficients: vec![intercept],
        effective_n_samples: sum_weight,
    })
}

fn append_columns(
    design: &[Vec<f64>],
    candidate_columns: &[Vec<f64>],
) -> MarsResult<Vec<Vec<f64>>> {
    let mut combined = design.to_vec();
    for column in candidate_columns {
        if column.len() != combined.len() {
            return Err(MarsError::MalformedArtifact(
                "candidate column length must match design row count".to_string(),
            ));
        }
        for (row_idx, value) in column.iter().enumerate() {
            combined[row_idx].push(*value);
        }
    }
    Ok(combined)
}

fn select_columns(design: &[Vec<f64>], columns: &[usize]) -> MarsResult<Vec<Vec<f64>>> {
    design
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|column| {
                    row.get(*column).copied().ok_or_else(|| {
                        MarsError::MalformedArtifact(format!(
                            "column index {column} outside design matrix"
                        ))
                    })
                })
                .collect()
        })
        .collect()
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> MarsResult<Vec<f64>> {
    let n = rhs.len();
    for pivot_idx in 0..n {
        let mut best_row = pivot_idx;
        let mut best_abs = matrix[pivot_idx][pivot_idx].abs();
        for (row_idx, row) in matrix.iter().enumerate().skip(pivot_idx + 1) {
            let candidate = row[pivot_idx].abs();
            if candidate > best_abs {
                best_abs = candidate;
                best_row = row_idx;
            }
        }
        if best_abs <= 1e-12 {
            return Err(MarsError::NumericalEvaluationFailure(
                "least-squares normal equations are singular".to_string(),
            ));
        }
        matrix.swap(pivot_idx, best_row);
        rhs.swap(pivot_idx, best_row);

        let pivot = matrix[pivot_idx][pivot_idx];
        for value in matrix[pivot_idx].iter_mut().skip(pivot_idx) {
            *value /= pivot;
        }
        let pivot_row = matrix[pivot_idx].clone();
        rhs[pivot_idx] /= pivot;

        for row_idx in 0..n {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = matrix[row_idx][pivot_idx];
            for (value, pivot_value) in matrix[row_idx]
                .iter_mut()
                .skip(pivot_idx)
                .zip(pivot_row.iter().skip(pivot_idx))
            {
                *value -= factor * pivot_value;
            }
            rhs[row_idx] -= factor * rhs[pivot_idx];
        }
    }
    Ok(rhs)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForwardPassResult {
    pub basis_terms: Vec<BasisTermSpec>,
    pub coefficients: Vec<f64>,
    pub rss: f64,
    pub gcv: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PruningResult {
    pub basis_terms: Vec<BasisTermSpec>,
    pub coefficients: Vec<f64>,
    pub rss: f64,
    pub gcv: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum CandidateAddition {
    Single(BasisTermSpec),
}

pub fn forward_pass(
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    params: &TrainingParams,
) -> MarsResult<ForwardPassResult> {
    validate_xy(x, y, sample_weight)?;
    let mut basis_terms = vec![constant_term()];
    let mut current = fit_terms(x, y, sample_weight, &basis_terms, params)?;

    loop {
        let candidates = generate_candidate_additions(x, &basis_terms, params)?;
        let mut best_candidate: Option<(CandidateAddition, ForwardPassResult)> = None;

        for candidate in candidates {
            let candidate_terms = basis_terms_with_addition(&basis_terms, &candidate);
            if candidate_terms.len() > params.max_terms {
                continue;
            }
            let candidate_fit = match fit_terms(x, y, sample_weight, &candidate_terms, params) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if candidate_fit.rss >= current.rss - f64::EPSILON {
                continue;
            }
            match &best_candidate {
                Some((_, best_fit))
                    if candidate_fit.gcv > best_fit.gcv + f64::EPSILON
                        || ((candidate_fit.gcv - best_fit.gcv).abs() <= f64::EPSILON
                            && candidate_fit.rss >= best_fit.rss - f64::EPSILON) => {}
                _ => best_candidate = Some((candidate, candidate_fit)),
            }
        }

        let Some((candidate, next_fit)) = best_candidate else {
            break;
        };

        let mut next_terms = basis_terms_with_addition(&basis_terms, &candidate);
        apply_candidate_scores(
            &mut next_terms,
            &candidate,
            current.rss,
            current.gcv,
            &next_fit,
        );
        basis_terms = next_terms;
        current = next_fit;

        if basis_terms.len() >= params.max_terms {
            break;
        }
    }

    Ok(current)
}

pub fn prune_model(
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    basis_terms: &[BasisTermSpec],
    _coefficients: &[f64],
    params: &TrainingParams,
) -> MarsResult<PruningResult> {
    validate_xy(x, y, sample_weight)?;
    if basis_terms.is_empty() {
        return Ok(PruningResult {
            basis_terms: Vec::new(),
            coefficients: Vec::new(),
            rss: f64::INFINITY,
            gcv: f64::INFINITY,
        });
    }

    let mut active = basis_terms.to_vec();
    let mut best = fit_terms(x, y, sample_weight, &active, params)?;
    let min_allowed_terms = if active.iter().any(|term| term.kind == "constant") {
        1
    } else {
        0
    };

    loop {
        if active.len() <= min_allowed_terms {
            break;
        }

        let mut best_subset: Option<(usize, ForwardPassResult)> = None;
        for idx in 0..active.len() {
            if active[idx].kind == "constant" && active.len() == min_allowed_terms {
                continue;
            }
            let subset: Vec<BasisTermSpec> = active
                .iter()
                .enumerate()
                .filter_map(|(term_idx, term)| (term_idx != idx).then_some(term.clone()))
                .collect();
            if subset.len() < min_allowed_terms {
                continue;
            }
            let subset_fit = match fit_terms(x, y, sample_weight, &subset, params) {
                Ok(value) => value,
                Err(_) => continue,
            };
            match &best_subset {
                Some((_, current_best))
                    if subset_fit.gcv > current_best.gcv + f64::EPSILON
                        || ((subset_fit.gcv - current_best.gcv).abs() <= f64::EPSILON
                            && subset_fit.rss >= current_best.rss - f64::EPSILON) => {}
                _ => best_subset = Some((idx, subset_fit)),
            }
        }

        let Some((idx_removed, candidate)) = best_subset else {
            break;
        };

        if candidate.gcv >= best.gcv - f64::EPSILON {
            break;
        }

        active.remove(idx_removed);
        best = candidate;
    }

    Ok(PruningResult {
        basis_terms: active,
        coefficients: best.coefficients,
        rss: best.rss,
        gcv: best.gcv,
    })
}

pub fn model_spec_from_terms(
    basis_terms: &[BasisTermSpec],
    coefficients: &[f64],
    params: &TrainingParams,
) -> ModelSpec {
    let n_features = basis_terms
        .iter()
        .filter_map(term_max_variable_idx)
        .max()
        .map_or(0, |max_idx| max_idx + 1);
    ModelSpec {
        spec_version: "1.0".to_string(),
        params: serde_json::json!({
            "max_terms": params.max_terms,
            "max_degree": params.max_degree,
            "penalty": params.penalty,
            "minspan": params.minspan,
            "endspan": params.endspan,
            "threshold": params.threshold,
            "allow_linear": params.allow_linear,
            "allow_missing": params.allow_missing,
            "categorical_features": params.categorical_features.clone(),
            "feature_names": params.feature_names.clone(),
        }),
        feature_schema: FeatureSchema {
            n_features: Some(n_features),
            feature_names: params.feature_names.clone().unwrap_or_default(),
        },
        basis_terms: basis_terms.to_vec(),
        coefficients: coefficients.to_vec(),
    }
}

pub fn fit_model(request: &TrainingRequest) -> MarsResult<ModelSpec> {
    let forward = forward_pass(
        &request.x,
        &request.y,
        request.sample_weight.as_deref(),
        &request.params,
    )?;
    let pruned = prune_model(
        &request.x,
        &request.y,
        request.sample_weight.as_deref(),
        &forward.basis_terms,
        &forward.coefficients,
        &request.params,
    )?;
    Ok(model_spec_from_terms(
        &pruned.basis_terms,
        &pruned.coefficients,
        &request.params,
    ))
}

fn validate_xy(x: &[Vec<f64>], y: &[f64], sample_weight: Option<&[f64]>) -> MarsResult<()> {
    if x.len() != y.len() {
        return Err(MarsError::MalformedArtifact(
            "x row count must match y length".to_string(),
        ));
    }
    if let Some(weights) = sample_weight {
        if weights.len() != y.len() {
            return Err(MarsError::MalformedArtifact(
                "sample_weight length must match y length".to_string(),
            ));
        }
    }
    if let Some(first_len) = x.first().map(Vec::len) {
        if x.iter().any(|row| row.len() != first_len) {
            return Err(MarsError::MalformedArtifact(
                "x rows must all have equal length".to_string(),
            ));
        }
    }
    Ok(())
}

fn fit_terms(
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    basis_terms: &[BasisTermSpec],
    params: &TrainingParams,
) -> MarsResult<ForwardPassResult> {
    let design = build_design_matrix(x, basis_terms)?;
    let fit = fit_least_squares(&design, y, sample_weight, true)?;
    let hinge_terms = basis_terms
        .iter()
        .filter(|term| term.kind == "hinge")
        .count();
    let gcv = calculate_gcv(
        fit.rss,
        fit.effective_n_samples,
        effective_parameters(fit.coefficients.len(), hinge_terms, params.penalty),
    );
    Ok(ForwardPassResult {
        basis_terms: basis_terms.to_vec(),
        coefficients: fit.coefficients,
        rss: fit.rss,
        gcv,
    })
}

fn build_design_matrix(x: &[Vec<f64>], basis_terms: &[BasisTermSpec]) -> MarsResult<Vec<Vec<f64>>> {
    x.iter()
        .map(|row| {
            basis_terms
                .iter()
                .map(|term| evaluate_basis(term, row))
                .collect()
        })
        .collect()
}

fn generate_candidate_additions(
    x: &[Vec<f64>],
    basis_terms: &[BasisTermSpec],
    params: &TrainingParams,
) -> MarsResult<Vec<CandidateAddition>> {
    let n_features = x.first().map_or(0, Vec::len);
    let mut candidates = Vec::new();

    let categorical_features = params.categorical_features.as_deref().unwrap_or(&[]);

    for parent in basis_terms {
        if term_degree(parent) >= params.max_degree {
            continue;
        }
        let parent_max_var = term_max_variable_idx(parent);
        for var_idx in 0..n_features {
            if !params.allow_linear {
                break;
            }
            if parent_max_var.is_some_and(|max_var| var_idx <= max_var) {
                continue;
            }
            let linear = linear_term(var_idx, params, Some(parent));
            if !has_exact_term(basis_terms, &linear) {
                candidates.push(CandidateAddition::Single(linear));
            }
            for knot in allowable_knots(x, var_idx, params) {
                let left = hinge_term(var_idx, knot, false, params, Some(parent));
                let right = hinge_term(var_idx, knot, true, params, Some(parent));
                if !has_exact_term(basis_terms, &left) {
                    candidates.push(CandidateAddition::Single(left));
                }
                if !has_exact_term(basis_terms, &right) {
                    candidates.push(CandidateAddition::Single(right));
                }
            }
        }
    }

    if params.max_degree > 1 {
        for left_idx in 0..n_features {
            for right_idx in (left_idx + 1)..n_features {
                let interaction = interaction_term(
                    &linear_term(left_idx, params, None),
                    &linear_term(right_idx, params, None),
                );
                if !has_exact_term(basis_terms, &interaction) {
                    candidates.push(CandidateAddition::Single(interaction));
                }
            }
        }
        for (left_idx, left_parent) in basis_terms.iter().enumerate() {
            if term_degree(left_parent) >= params.max_degree {
                continue;
            }
            for right_parent in basis_terms.iter().skip(left_idx + 1) {
                if term_degree(left_parent) + term_degree(right_parent) > params.max_degree {
                    continue;
                }
                if left_parent.kind == "constant" || right_parent.kind == "constant" {
                    continue;
                }
                let interaction = interaction_term(left_parent, right_parent);
                if !has_exact_term(basis_terms, &interaction) {
                    candidates.push(CandidateAddition::Single(interaction));
                }
            }
        }
    }

    if params.allow_linear {
        for var_idx in 0..n_features {
            let linear = linear_term(var_idx, params, None);
            if !has_exact_term(basis_terms, &linear) {
                candidates.push(CandidateAddition::Single(linear));
            }
            for knot in allowable_knots(x, var_idx, params) {
                let left = hinge_term(var_idx, knot, false, params, None);
                let right = hinge_term(var_idx, knot, true, params, None);
                if !has_exact_term(basis_terms, &left) {
                    candidates.push(CandidateAddition::Single(left));
                }
                if !has_exact_term(basis_terms, &right) {
                    candidates.push(CandidateAddition::Single(right));
                }
            }
        }
    }

    for &var_idx in categorical_features {
        for category in categorical_observed_values(x, var_idx) {
            let categorical = categorical_term(var_idx, category, params, None);
            if !has_exact_term(basis_terms, &categorical) {
                candidates.push(CandidateAddition::Single(categorical));
            }
        }
    }

    if params.allow_missing {
        for var_idx in 0..n_features {
            if column_has_missing(x, var_idx) {
                let missingness = missingness_term(var_idx, params, None);
                if !has_exact_term(basis_terms, &missingness) {
                    candidates.push(CandidateAddition::Single(missingness));
                }
            }
        }
    }

    Ok(candidates)
}

fn basis_terms_with_addition(
    basis_terms: &[BasisTermSpec],
    addition: &CandidateAddition,
) -> Vec<BasisTermSpec> {
    let mut terms = basis_terms.to_vec();
    match addition {
        CandidateAddition::Single(term) => terms.push(term.clone()),
    }
    terms
}

fn apply_candidate_scores(
    basis_terms: &mut [BasisTermSpec],
    candidate: &CandidateAddition,
    current_rss: f64,
    current_gcv: f64,
    next_fit: &ForwardPassResult,
) {
    let gcv_score = current_gcv - next_fit.gcv;
    let rss_score = current_rss - next_fit.rss;
    match candidate {
        CandidateAddition::Single(term) => {
            if let Some(slot) = basis_terms.iter_mut().find(|existing| *existing == term) {
                slot.gcv_score = Some(gcv_score);
                slot.rss_score = Some(rss_score);
            }
        }
    }
}

fn constant_term() -> BasisTermSpec {
    BasisTermSpec {
        kind: "constant".to_string(),
        variable_idx: None,
        variable_name: None,
        knot_val: None,
        is_right_hinge: None,
        category: None,
        gcv_score: Some(0.0),
        rss_score: Some(0.0),
        parent1: None,
        parent2: None,
    }
}

fn linear_term(
    variable_idx: usize,
    params: &TrainingParams,
    parent1: Option<&BasisTermSpec>,
) -> BasisTermSpec {
    BasisTermSpec {
        kind: "linear".to_string(),
        variable_idx: Some(variable_idx),
        variable_name: Some(feature_name(params, variable_idx)),
        knot_val: None,
        is_right_hinge: None,
        category: None,
        gcv_score: None,
        rss_score: None,
        parent1: parent1.map(|term| Box::new(term.clone())),
        parent2: None,
    }
}

fn hinge_term(
    variable_idx: usize,
    knot_val: f64,
    is_right_hinge: bool,
    params: &TrainingParams,
    parent1: Option<&BasisTermSpec>,
) -> BasisTermSpec {
    BasisTermSpec {
        kind: "hinge".to_string(),
        variable_idx: Some(variable_idx),
        variable_name: Some(feature_name(params, variable_idx)),
        knot_val: Some(knot_val),
        is_right_hinge: Some(is_right_hinge),
        category: None,
        gcv_score: None,
        rss_score: None,
        parent1: parent1.map(|term| Box::new(term.clone())),
        parent2: None,
    }
}

fn categorical_term(
    variable_idx: usize,
    category: f64,
    params: &TrainingParams,
    parent1: Option<&BasisTermSpec>,
) -> BasisTermSpec {
    BasisTermSpec {
        kind: "categorical".to_string(),
        variable_idx: Some(variable_idx),
        variable_name: Some(feature_name(params, variable_idx)),
        knot_val: None,
        is_right_hinge: None,
        category: Some(serde_json::json!(category)),
        gcv_score: None,
        rss_score: None,
        parent1: parent1.map(|term| Box::new(term.clone())),
        parent2: None,
    }
}

fn missingness_term(
    variable_idx: usize,
    params: &TrainingParams,
    parent1: Option<&BasisTermSpec>,
) -> BasisTermSpec {
    BasisTermSpec {
        kind: "missingness".to_string(),
        variable_idx: Some(variable_idx),
        variable_name: Some(feature_name(params, variable_idx)),
        knot_val: None,
        is_right_hinge: None,
        category: None,
        gcv_score: None,
        rss_score: None,
        parent1: parent1.map(|term| Box::new(term.clone())),
        parent2: None,
    }
}

fn interaction_term(parent1: &BasisTermSpec, parent2: &BasisTermSpec) -> BasisTermSpec {
    BasisTermSpec {
        kind: "interaction".to_string(),
        variable_idx: None,
        variable_name: None,
        knot_val: None,
        is_right_hinge: None,
        category: None,
        gcv_score: None,
        rss_score: None,
        parent1: Some(Box::new(parent1.clone())),
        parent2: Some(Box::new(parent2.clone())),
    }
}

fn feature_name(params: &TrainingParams, variable_idx: usize) -> String {
    params
        .feature_names
        .as_ref()
        .and_then(|names| names.get(variable_idx).cloned())
        .unwrap_or_else(|| format!("x{variable_idx}"))
}

fn allowable_knots(x: &[Vec<f64>], variable_idx: usize, params: &TrainingParams) -> Vec<f64> {
    let mut values = x
        .iter()
        .filter_map(|row| row.get(variable_idx).copied())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup();
    if values.len() <= 2 {
        return Vec::new();
    }

    let endspan = if params.endspan > 0.0 {
        params.endspan.round() as usize
    } else {
        0
    };
    let start = endspan.min(values.len());
    let end = values.len().saturating_sub(endspan);
    if start >= end.saturating_sub(1) {
        return Vec::new();
    }
    values[start..end - 1].to_vec()
}

fn categorical_observed_values(x: &[Vec<f64>], variable_idx: usize) -> Vec<f64> {
    let mut values = x
        .iter()
        .filter_map(|row| row.get(variable_idx).copied())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup();
    values
}

fn column_has_missing(x: &[Vec<f64>], variable_idx: usize) -> bool {
    x.iter()
        .filter_map(|row| row.get(variable_idx).copied())
        .any(|value| value.is_nan())
}

fn term_degree(term: &BasisTermSpec) -> usize {
    match term.kind.as_str() {
        "constant" => 0,
        "linear" | "hinge" | "categorical" | "missingness" => {
            1 + term.parent1.as_deref().map_or(0, term_degree)
        }
        "interaction" => {
            term.parent1.as_deref().map_or(0, term_degree)
                + term.parent2.as_deref().map_or(0, term_degree)
        }
        _ => 1,
    }
}

fn term_max_variable_idx(term: &BasisTermSpec) -> Option<usize> {
    let own = term.variable_idx;
    let left = term.parent1.as_deref().and_then(term_max_variable_idx);
    let right = term.parent2.as_deref().and_then(term_max_variable_idx);
    own.into_iter().chain(left).chain(right).max()
}

fn has_exact_term(existing: &[BasisTermSpec], term: &BasisTermSpec) -> bool {
    existing
        .iter()
        .any(|candidate| same_term_shape(candidate, term))
}

fn same_term_shape(left: &BasisTermSpec, right: &BasisTermSpec) -> bool {
    left.kind == right.kind
        && left.variable_idx == right.variable_idx
        && left.variable_name == right.variable_name
        && left.knot_val == right.knot_val
        && left.is_right_hinge == right.is_right_hinge
        && left.category == right.category
        && match (&left.parent1, &right.parent1) {
            (Some(lhs), Some(rhs)) => same_term_shape(lhs, rhs),
            (None, None) => true,
            _ => false,
        }
        && match (&left.parent2, &right.parent2) {
            (Some(lhs), Some(rhs)) => same_term_shape(lhs, rhs),
            (None, None) => true,
            _ => false,
        }
}
