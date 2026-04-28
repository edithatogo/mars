package pymars

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

type FeatureSchema struct {
	NFeatures    *int     `json:"n_features"`
	FeatureNames []string `json:"feature_names"`
}

type BasisTerm struct {
	Kind         string          `json:"kind"`
	VariableIdx *int            `json:"variable_idx"`
	KnotVal     *float64        `json:"knot_val"`
	IsRight     *bool           `json:"is_right_hinge"`
	Category    json.RawMessage `json:"category"`
	Parent1     *BasisTerm      `json:"parent1"`
	Parent2     *BasisTerm      `json:"parent2"`
}

type ModelSpec struct {
	SpecVersion   string        `json:"spec_version"`
	Params        any           `json:"params"`
	FeatureSchema FeatureSchema `json:"feature_schema"`
	BasisTerms    []BasisTerm   `json:"basis_terms"`
	Coefficients  []float64     `json:"coefficients"`
}

func LoadModelSpec(raw []byte) (*ModelSpec, error) {
	var spec ModelSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, fmt.Errorf("malformed artifact: %w", err)
	}
	if err := Validate(&spec); err != nil {
		return nil, err
	}
	return &spec, nil
}

func Validate(spec *ModelSpec) error {
	if spec == nil {
		return errors.New("malformed artifact: model spec is nil")
	}
	if spec.SpecVersion == "" || len(spec.SpecVersion) < 3 || spec.SpecVersion[1] != '.' {
		return errors.New("malformed artifact: spec_version must be '<major>.<minor>'")
	}
	if spec.SpecVersion[0] != '1' {
		return fmt.Errorf("unsupported artifact version: %s", spec.SpecVersion)
	}
	if len(spec.BasisTerms) != len(spec.Coefficients) {
		return errors.New("malformed artifact: coefficients length must match basis_terms")
	}
	if spec.FeatureSchema.NFeatures != nil && len(spec.FeatureSchema.FeatureNames) > 0 {
		if len(spec.FeatureSchema.FeatureNames) != *spec.FeatureSchema.NFeatures {
			return errors.New("malformed artifact: feature_names length must match n_features")
		}
	}
	for idx := range spec.BasisTerms {
		if err := validateBasis(spec, &spec.BasisTerms[idx], idx); err != nil {
			return err
		}
	}
	return nil
}

func DesignMatrix(spec *ModelSpec, rows [][]float64) ([][]float64, error) {
	if err := Validate(spec); err != nil {
		return nil, err
	}
	if err := validateRows(spec, rows); err != nil {
		return nil, err
	}
	matrix := make([][]float64, len(rows))
	for i, row := range rows {
		matrix[i] = make([]float64, len(spec.BasisTerms))
		for j := range spec.BasisTerms {
			value, err := evaluateBasis(&spec.BasisTerms[j], row)
			if err != nil {
				return nil, err
			}
			matrix[i][j] = value
		}
	}
	return matrix, nil
}

func Predict(spec *ModelSpec, rows [][]float64) ([]float64, error) {
	matrix, err := DesignMatrix(spec, rows)
	if err != nil {
		return nil, err
	}
	predictions := make([]float64, len(matrix))
	for i, row := range matrix {
		for j, value := range row {
			predictions[i] += value * spec.Coefficients[j]
		}
	}
	return predictions, nil
}

func validateBasis(spec *ModelSpec, basis *BasisTerm, idx int) error {
	if basis.Kind == "" {
		return fmt.Errorf("missing required field: basis term %d has empty kind", idx)
	}
	switch basis.Kind {
	case "constant":
		return nil
	case "linear", "missingness":
		return validateVariableIdx(spec, basis.VariableIdx, idx)
	case "hinge":
		if err := validateVariableIdx(spec, basis.VariableIdx, idx); err != nil {
			return err
		}
		if basis.KnotVal == nil {
			return errors.New("missing required field: hinge requires knot_val")
		}
		if basis.IsRight == nil {
			return errors.New("missing required field: hinge requires is_right_hinge")
		}
		return nil
	case "categorical":
		if err := validateVariableIdx(spec, basis.VariableIdx, idx); err != nil {
			return err
		}
		_, err := categoryValue(basis)
		return err
	case "interaction":
		if basis.Parent1 == nil || basis.Parent2 == nil {
			return errors.New("missing required field: interaction requires parent1 and parent2")
		}
		return nil
	default:
		return fmt.Errorf("unsupported basis term: %s", basis.Kind)
	}
}

func validateVariableIdx(spec *ModelSpec, variableIdx *int, basisIdx int) error {
	if variableIdx == nil {
		return errors.New("missing required field: basis term requires variable_idx")
	}
	if spec.FeatureSchema.NFeatures != nil && *variableIdx >= *spec.FeatureSchema.NFeatures {
		return fmt.Errorf("malformed artifact: basis term %d references variable outside feature count", basisIdx)
	}
	return nil
}

func validateRows(spec *ModelSpec, rows [][]float64) error {
	if spec.FeatureSchema.NFeatures == nil {
		return nil
	}
	for idx, row := range rows {
		if len(row) != *spec.FeatureSchema.NFeatures {
			return fmt.Errorf("feature-count mismatch: row %d has %d features, expected %d", idx, len(row), *spec.FeatureSchema.NFeatures)
		}
	}
	return nil
}

func evaluateBasis(basis *BasisTerm, row []float64) (float64, error) {
	switch basis.Kind {
	case "constant":
		return 1, nil
	case "linear":
		return row[*basis.VariableIdx], nil
	case "hinge":
		value := row[*basis.VariableIdx]
		if *basis.IsRight {
			return math.Max(value-*basis.KnotVal, 0), nil
		}
		return math.Max(*basis.KnotVal-value, 0), nil
	case "categorical":
		category, err := categoryValue(basis)
		if err != nil {
			return 0, err
		}
		value := row[*basis.VariableIdx]
		if math.IsNaN(value) {
			return math.NaN(), nil
		}
		if value == category {
			return 1, nil
		}
		return 0, nil
	case "interaction":
		left, err := evaluateBasis(basis.Parent1, row)
		if err != nil {
			return 0, err
		}
		right, err := evaluateBasis(basis.Parent2, row)
		if err != nil {
			return 0, err
		}
		if math.IsNaN(left) || math.IsNaN(right) {
			return math.NaN(), nil
		}
		return left * right, nil
	case "missingness":
		if math.IsNaN(row[*basis.VariableIdx]) {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("unsupported basis term: %s", basis.Kind)
	}
}

func categoryValue(basis *BasisTerm) (float64, error) {
	if len(basis.Category) == 0 || string(basis.Category) == "null" {
		return 0, errors.New("missing required field: categorical requires category")
	}
	var value float64
	if err := json.Unmarshal(basis.Category, &value); err != nil {
		return 0, fmt.Errorf("invalid categorical encoding: %w", err)
	}
	return value, nil
}
