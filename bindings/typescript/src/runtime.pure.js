export function validatePure(spec) {
  if (!spec || typeof spec !== "object") {
    throw new Error("malformed artifact: model spec must be an object");
  }
  if (!/^\d+\.\d+$/.test(spec.spec_version ?? "")) {
    throw new Error("malformed artifact: spec_version must be '<major>.<minor>'");
  }
  if (!spec.spec_version.startsWith("1.")) {
    throw new Error(`unsupported artifact version: ${spec.spec_version}`);
  }
  if (!Array.isArray(spec.basis_terms) || !Array.isArray(spec.coefficients)) {
    throw new Error("malformed artifact: basis_terms and coefficients are required");
  }
  if (spec.basis_terms.length !== spec.coefficients.length) {
    throw new Error("malformed artifact: coefficients length must match basis_terms");
  }
  spec.basis_terms.forEach((basis, index) => validateBasis(spec, basis, index));
}

export function validateRowsPure(spec, rows) {
  const nFeatures = spec.feature_schema?.n_features;
  if (nFeatures == null) return;
  rows.forEach((row, index) => {
    if (row.length !== nFeatures) {
      throw new Error(
        `feature-count mismatch: row ${index} has ${row.length} features, expected ${nFeatures}`,
      );
    }
  });
}

export function validateBasis(spec, basis, index) {
  if (!basis.kind) throw new Error(`missing required field: basis term ${index} has empty kind`);
  if (["linear", "hinge", "categorical", "missingness"].includes(basis.kind)) {
    validateVariableIdx(spec, basis.variable_idx, index);
  }
  if (basis.kind === "hinge" && (basis.knot_val == null || basis.is_right_hinge == null)) {
    throw new Error("missing required field: hinge requires knot_val and is_right_hinge");
  }
  if (basis.kind === "categorical" && basis.category == null) {
    throw new Error("missing required field: categorical requires category");
  }
  if (basis.kind === "interaction" && (!basis.parent1 || !basis.parent2)) {
    throw new Error("missing required field: interaction requires parent1 and parent2");
  }
  if (!["constant", "linear", "hinge", "categorical", "interaction", "missingness"].includes(basis.kind)) {
    throw new Error(`unsupported basis term: ${basis.kind}`);
  }
}

export function validateVariableIdx(spec, variableIdx, basisIdx) {
  if (variableIdx == null) throw new Error("missing required field: basis term requires variable_idx");
  const nFeatures = spec.feature_schema?.n_features;
  if (nFeatures != null && variableIdx >= nFeatures) {
    throw new Error(`malformed artifact: basis term ${basisIdx} references variable outside feature count`);
  }
}

export function evaluateBasis(basis, row) {
  switch (basis.kind) {
    case "constant":
      return 1;
    case "linear":
      return row[basis.variable_idx];
    case "hinge": {
      const value = row[basis.variable_idx];
      return basis.is_right_hinge
        ? Math.max(value - basis.knot_val, 0)
        : Math.max(basis.knot_val - value, 0);
    }
    case "categorical": {
      const value = row[basis.variable_idx];
      if (Number.isNaN(value)) return Number.NaN;
      return value === basis.category ? 1 : 0;
    }
    case "interaction": {
      const left = evaluateBasis(basis.parent1, row);
      const right = evaluateBasis(basis.parent2, row);
      return Number.isNaN(left) || Number.isNaN(right) ? Number.NaN : left * right;
    }
    case "missingness":
      return Number.isNaN(row[basis.variable_idx]) ? 1 : 0;
    default:
      throw new Error(`unsupported basis term: ${basis.kind}`);
  }
}

