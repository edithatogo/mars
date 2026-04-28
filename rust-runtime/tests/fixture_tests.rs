use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use pymars_runtime::{design_matrix, load_model_spec_path, predict};

#[derive(Debug, Deserialize)]
struct RuntimeFixture {
    probe: Vec<Vec<Option<f64>>>,
    design_matrix: Vec<Vec<Option<f64>>>,
    predict: Vec<Option<f64>>,
}

#[test]
fn validates_and_matches_checked_in_python_fixture_outputs() -> Result<()> {
    let fixtures_dir = repo_root().join("tests/fixtures");
    let fixture_pairs = fixture_pairs(&fixtures_dir)?;

    assert!(
        !fixture_pairs.is_empty(),
        "expected at least one runtime portability fixture pair in {}",
        fixtures_dir.display()
    );

    for (model_spec_path, runtime_fixture_path) in fixture_pairs {
        let spec = load_model_spec_path(&model_spec_path)?;
        let runtime_fixture = load_runtime_fixture(&runtime_fixture_path)?;
        let probe = normalize_matrix(&runtime_fixture.probe);
        let expected_design_matrix = normalize_matrix(&runtime_fixture.design_matrix);
        let expected_predictions = normalize_vector(&runtime_fixture.predict);

        let matrix = design_matrix(&spec, &probe).with_context(|| {
            format!(
                "failed design_matrix parity for fixture {}",
                model_spec_path.display()
            )
        })?;
        assert_matrix_close(&matrix, &expected_design_matrix);

        let predictions = predict(&spec, &probe).with_context(|| {
            format!(
                "failed predict parity for fixture {}",
                model_spec_path.display()
            )
        })?;
        assert_vector_close(&predictions, &expected_predictions);
    }

    Ok(())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust-runtime should sit under the repository root")
        .to_path_buf()
}

fn fixture_pairs(fixtures_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();
    let mut model_specs: Vec<PathBuf> = fs::read_dir(fixtures_dir)
        .with_context(|| format!("failed to read {}", fixtures_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with("model_spec_") && name.ends_with(".json"))
        })
        .collect();

    model_specs.sort();

    for model_spec_path in model_specs {
        let suffix = model_spec_path
            .file_stem()
            .and_then(OsStr::to_str)
            .and_then(|stem| stem.strip_prefix("model_spec_"))
            .with_context(|| {
                format!(
                    "failed to derive fixture suffix from {}",
                    model_spec_path.display()
                )
            })?;
        let runtime_fixture_path =
            fixtures_dir.join(format!("runtime_portability_fixture_{suffix}.json"));
        if !runtime_fixture_path.exists() {
            anyhow::bail!(
                "missing runtime portability fixture for {}: expected {}",
                model_spec_path.display(),
                runtime_fixture_path.display()
            );
        }
        pairs.push((model_spec_path, runtime_fixture_path));
    }

    Ok(pairs)
}

fn load_runtime_fixture(path: &Path) -> Result<RuntimeFixture> {
    serde_json::from_str(
        &fs::read_to_string(path)
            .with_context(|| format!("failed to read runtime fixture {}", path.display()))?,
    )
    .with_context(|| format!("failed to deserialize runtime fixture {}", path.display()))
}

fn normalize_matrix(rows: &[Vec<Option<f64>>]) -> Vec<Vec<f64>> {
    rows.iter().map(|row| normalize_vector(row)).collect()
}

fn normalize_vector(values: &[Option<f64>]) -> Vec<f64> {
    values
        .iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect()
}

fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len(), "row count mismatch");
    for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_row.len(),
            expected_row.len(),
            "column count mismatch on row {}",
            row_idx
        );
        for (col_idx, (actual_value, expected_value)) in
            actual_row.iter().zip(expected_row.iter()).enumerate()
        {
            assert_close(
                *actual_value,
                *expected_value,
                &format!("matrix[{row_idx}][{col_idx}]"),
            );
        }
    }
}

fn assert_vector_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "vector length mismatch");
    for (idx, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(*actual_value, *expected_value, &format!("vector[{idx}]"));
    }
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    if actual.is_nan() && expected.is_nan() {
        return;
    }
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1e-12,
        "{} mismatch: actual={} expected={} delta={}",
        label,
        actual,
        expected,
        delta
    );
}
