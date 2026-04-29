import { mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import {
  evaluateBasis,
  validatePure,
  validateRowsPure,
} from "./runtime.pure.js";

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(moduleDir, "../../..");

class RustRuntimeUnavailableError extends Error {}

export function loadModelSpec(raw) {
  const spec = typeof raw === "string" ? JSON.parse(raw) : raw;
  validate(spec);
  return spec;
}

export function fitModel() {
  throw new Error(
    "training is not supported in @mars-earth/runtime; use a training-capable binding or the Rust CLI fit subcommand",
  );
}

export function validate(spec) {
  const binary = rustRuntimeBinary();
  if (binary) {
    try {
      invokeRustRuntime("validate", spec, null, binary);
      return;
    } catch (error) {
      if (!(error instanceof RustRuntimeUnavailableError)) {
        throw error;
      }
    }
  }
  validatePure(spec);
}

export function designMatrix(spec, rows) {
  const binary = rustRuntimeBinary();
  if (binary) {
    try {
      return invokeRustRuntime("design-matrix", spec, rows, binary);
    } catch (error) {
      if (!(error instanceof RustRuntimeUnavailableError)) {
        throw error;
      }
    }
  }
  validatePure(spec);
  validateRowsPure(spec, rows);
  return rows.map((row) => spec.basis_terms.map((basis) => evaluateBasis(basis, row)));
}

export function predict(spec, rows) {
  const binary = rustRuntimeBinary();
  if (binary) {
    try {
      return invokeRustRuntime("predict", spec, rows, binary);
    } catch (error) {
      if (!(error instanceof RustRuntimeUnavailableError)) {
        throw error;
      }
    }
  }
  return designMatrix(spec, rows).map((row) =>
    row.reduce((total, value, index) => total + value * spec.coefficients[index], 0),
  );
}

function invokeRustRuntime(command, spec, rows = null, binary = rustRuntimeBinary()) {
  if (!binary) {
    throw new Error("Rust runtime binary is not available");
  }

  const tmpdir = mkdtempSync(path.join(os.tmpdir(), "mars-ts-"));
  const specFile = path.join(tmpdir, "spec.json");
  writeFileSync(specFile, JSON.stringify(spec));

  const args = [command, "--spec-file", specFile];
  if (rows != null) {
    const rowsFile = path.join(tmpdir, "rows.json");
    writeFileSync(rowsFile, JSON.stringify(nullableRows(rows)));
    args.push("--rows-file", rowsFile);
  }

  try {
    const result = spawnSync(binary, args, { encoding: "utf8" });
    if (result.error) {
      if (result.error.code === "ENOENT" || result.error.code === "EACCES") {
        throw new RustRuntimeUnavailableError(
          `Rust runtime binary is not available: ${result.error.message}`,
        );
      }
      throw new Error(`failed to execute Rust runtime: ${result.error.message}`);
    }
    if (result.status !== 0) {
      const message = (result.stderr || result.stdout || "").trim();
      throw new Error(message || `Rust runtime command failed: ${command}`);
    }

    if (command === "validate") {
      return true;
    }
    const payload = JSON.parse(result.stdout || "null");
    if (command === "design-matrix") {
      return normalizeMatrix(payload);
    }
    if (command === "predict") {
      return normalizeVector(payload);
    }
    return payload;
  } finally {
    rmSync(tmpdir, { recursive: true, force: true });
  }
}

function rustRuntimeBinary() {
  const envBinary = process.env.MARS_RUNTIME_BIN ?? "";
  const candidates = [
    envBinary,
    path.resolve(repoRoot, "rust-runtime/target/debug/mars-runtime-cli"),
    path.resolve(repoRoot, "rust-runtime/target/release/mars-runtime-cli"),
  ];
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) {
      return candidate;
    }
  }
  return "";
}

function nullableRows(rows) {
  return rows.map((row) => row.map((value) => (Number.isNaN(value) ? null : value)));
}

function normalizeMatrix(payload) {
  return payload.map((row) => normalizeVector(row));
}

function normalizeVector(payload) {
  return payload.map((value) => (value === null ? Number.NaN : Number(value)));
}
