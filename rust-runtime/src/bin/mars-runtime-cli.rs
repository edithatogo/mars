use std::env;
use std::fs;
use std::process;

use pymars_runtime::runtime::{design_matrix, load_model_spec_str, predict, validate_model_spec};
use pymars_runtime::training::{fit_model, TrainingRequest};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct JsonRows(Vec<Vec<Option<f64>>>);

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| {
        "expected a subcommand: validate, design-matrix, predict, or fit".to_string()
    })?;

    match command.as_str() {
        "validate" => {
            let spec_path = parse_flag(&mut args, "--spec-file")?;
            let spec = load_model_spec_str(
                &fs::read_to_string(&spec_path)
                    .map_err(|error| format!("failed to read spec file {spec_path}: {error}"))?,
            )
            .map_err(|error| error.to_string())?;
            validate_model_spec(&spec).map_err(|error| error.to_string())?;
            Ok(())
        }
        "design-matrix" => {
            let spec_path = parse_flag(&mut args, "--spec-file")?;
            let rows_path = parse_flag(&mut args, "--rows-file")?;
            let spec = load_model_spec_str(
                &fs::read_to_string(&spec_path)
                    .map_err(|error| format!("failed to read spec file {spec_path}: {error}"))?,
            )
            .map_err(|error| error.to_string())?;
            let rows = load_rows(&rows_path)?;
            let matrix = design_matrix(&spec, &rows).map_err(|error| error.to_string())?;
            let payload = matrix
                .into_iter()
                .map(|row| row.into_iter().map(jsonify_float).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string(&payload).map_err(|error| error.to_string())?
            );
            Ok(())
        }
        "predict" => {
            let spec_path = parse_flag(&mut args, "--spec-file")?;
            let rows_path = parse_flag(&mut args, "--rows-file")?;
            let spec = load_model_spec_str(
                &fs::read_to_string(&spec_path)
                    .map_err(|error| format!("failed to read spec file {spec_path}: {error}"))?,
            )
            .map_err(|error| error.to_string())?;
            let rows = load_rows(&rows_path)?;
            let values = predict(&spec, &rows).map_err(|error| error.to_string())?;
            let payload = values.into_iter().map(jsonify_float).collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string(&payload).map_err(|error| error.to_string())?
            );
            Ok(())
        }
        "fit" => {
            let request_path = parse_flag(&mut args, "--request-file")?;
            let request: TrainingRequest =
                serde_json::from_str(&fs::read_to_string(&request_path).map_err(|error| {
                    format!("failed to read request file {request_path}: {error}")
                })?)
                .map_err(|error| error.to_string())?;
            let spec = fit_model(&request).map_err(|error| error.to_string())?;
            println!(
                "{}",
                serde_json::to_string(&spec).map_err(|error| error.to_string())?
            );
            Ok(())
        }
        other => Err(format!(
            "unknown subcommand {other}; expected validate, design-matrix, predict, or fit"
        )),
    }
}

fn parse_flag(args: &mut impl Iterator<Item = String>, expected: &str) -> Result<String, String> {
    let flag = args
        .next()
        .ok_or_else(|| format!("expected flag {expected}"))?;
    if flag != expected {
        return Err(format!("expected flag {expected}, got {flag}"));
    }
    args.next()
        .ok_or_else(|| format!("expected value for flag {expected}"))
}

fn load_rows(path: &str) -> Result<Vec<Vec<f64>>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read rows file {path}: {error}"))?;
    let JsonRows(rows): JsonRows = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse rows JSON: {error}"))?;
    Ok(rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect()
        })
        .collect())
}

fn jsonify_float(value: f64) -> serde_json::Value {
    if value.is_nan() {
        serde_json::Value::Null
    } else {
        json!(value)
    }
}
