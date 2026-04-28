use crate::errors::{MarsError, MarsResult};
use crate::model_spec::BasisTermSpec;

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingParams {
    pub max_terms: usize,
    pub max_degree: usize,
    pub penalty: f64,
    pub minspan: f64,
    pub endspan: f64,
    pub threshold: f64,
    pub feature_names: Option<Vec<String>>,
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
            feature_names: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForwardPassResult {
    pub basis_terms: Vec<BasisTermSpec>,
    pub coefficients: Vec<f64>,
    pub rss: f64,
    pub gcv: f64,
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
        for col in pivot_idx..n {
            matrix[pivot_idx][col] /= pivot;
        }
        rhs[pivot_idx] /= pivot;

        for row_idx in 0..n {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = matrix[row_idx][pivot_idx];
            for col in pivot_idx..n {
                matrix[row_idx][col] -= factor * matrix[pivot_idx][col];
            }
            rhs[row_idx] -= factor * rhs[pivot_idx];
        }
    }
    Ok(rhs)
}

/// Generate candidate hinge terms from current basis functions
/// This is a placeholder - full implementation will be added in subsequent tasks
pub fn generate_candidates(
    _x: &[Vec<f64>],
    _basis_terms: &[BasisTermSpec],
    _params: &TrainingParams,
) -> Vec<Vec<f64>> {
    // TODO: Implement candidate generation
    // - Generate hinge pairs (max(0, x - knot), max(0, knot - x))
    // - Apply minspan/endspan constraints
    // - Consider interaction terms up to max_degree
    // - Return candidate columns for evaluation
    vec![]
}

/// Run forward-pass orchestration
/// This is a placeholder - full implementation will be added in subsequent tasks
pub fn forward_pass(
    _x: &[Vec<f64>],
    _y: &[f64],
    _sample_weight: Option<&[f64]>,
    _params: &TrainingParams,
) -> MarsResult<ForwardPassResult> {
    // TODO: Implement forward pass
    // 1. Start with intercept term
    // 2. Loop until max_terms reached or no improvement:
    //    - Generate candidates
    //    - Score each candidate using score_candidate
    //    - Select best candidate (lowest GCV)
    //    - Add to basis set
    // 3. Return ForwardPassResult with basis_terms, coefficients, rss, gcv

    Err(MarsError::NotYetImplemented(
        "forward_pass not yet implemented".to_string(),
    ))
}
