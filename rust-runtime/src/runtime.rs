use std::fs;
use std::path::Path;

use crate::errors::{MarsError, MarsResult};
use crate::model_spec::{BasisTermSpec, ModelSpec};

pub fn load_model_spec_str(raw: &str) -> MarsResult<ModelSpec> {
    let spec: ModelSpec = serde_json::from_str(raw).map_err(|error| {
        MarsError::MalformedArtifact(format!("failed to deserialize model spec JSON: {error}"))
    })?;
    validate_model_spec(&spec)?;
    Ok(spec)
}

pub fn load_model_spec_path(path: impl AsRef<Path>) -> MarsResult<ModelSpec> {
    let path_ref = path.as_ref();
    let raw = fs::read_to_string(path_ref).map_err(|error| {
        MarsError::MalformedArtifact(format!(
            "failed to read model spec from {}: {error}",
            path_ref.display()
        ))
    })?;
    load_model_spec_str(&raw)
}

pub fn validate_model_spec(spec: &ModelSpec) -> MarsResult<()> {
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
    validate_model_spec(spec)?;
    validate_rows(spec, rows)?;

    rows.iter()
        .map(|row| {
            spec.basis_terms
                .iter()
                .map(|basis| evaluate_basis(basis, row))
                .collect()
        })
        .collect()
}

pub fn predict(spec: &ModelSpec, rows: &[Vec<f64>]) -> MarsResult<Vec<f64>> {
    let matrix = design_matrix(spec, rows)?;
    Ok(matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(spec.coefficients.iter())
                .map(|(value, coefficient)| value * coefficient)
                .sum()
        })
        .collect())
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
            Ok(row[variable_idx])
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
            if basis.is_right_hinge.unwrap_or(true) {
                Ok((value - knot_val).max(0.0))
            } else {
                Ok((knot_val - value).max(0.0))
            }
        }
        "categorical" => {
            let variable_idx = basis.variable_idx.ok_or_else(|| {
                MarsError::MissingRequiredField(
                    "categorical basis terms require variable_idx".to_string(),
                )
            })?;
            let category = numeric_category_value(basis)?;
            let value = row[variable_idx];
            if value.is_nan() {
                Ok(f64::NAN)
            } else if value == category {
                Ok(1.0)
            } else {
                Ok(0.0)
            }
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
            if row[variable_idx].is_nan() {
                Ok(1.0)
            } else {
                Ok(0.0)
            }
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
