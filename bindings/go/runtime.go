package pymars

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type FeatureSchema struct {
	NFeatures    *int     `json:"n_features"`
	FeatureNames []string `json:"feature_names"`
}

type BasisTerm struct {
	Kind        string          `json:"kind"`
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

type TrainingParams struct {
	MaxTerms            int      `json:"max_terms"`
	MaxDegree           int      `json:"max_degree"`
	Penalty             float64  `json:"penalty"`
	Minspan             float64  `json:"minspan"`
	Endspan             float64  `json:"endspan"`
	Threshold           float64  `json:"threshold"`
	AllowLinear         bool     `json:"allow_linear"`
	AllowMissing        bool     `json:"allow_missing"`
	CategoricalFeatures []int    `json:"categorical_features,omitempty"`
	FeatureNames        []string `json:"feature_names,omitempty"`
}

type TrainingRequest struct {
	X            [][]float64    `json:"x"`
	Y            []float64      `json:"y"`
	SampleWeight []float64      `json:"sample_weight,omitempty"`
	Params       TrainingParams `json:"params"`
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
	if ok, err := tryValidateWithRust(spec); ok {
		return nil
	} else if err != nil {
		return err
	}
	return validatePure(spec)
}

func DesignMatrix(spec *ModelSpec, rows [][]float64) ([][]float64, error) {
	if ok, matrix, err := tryDesignMatrixWithRust(spec, rows); ok {
		return matrix, nil
	} else if err != nil {
		return nil, err
	}
	return designMatrixPure(spec, rows)
}

func Predict(spec *ModelSpec, rows [][]float64) ([]float64, error) {
	if ok, predictions, err := tryPredictWithRust(spec, rows); ok {
		return predictions, nil
	} else if err != nil {
		return nil, err
	}
	return predictPure(spec, rows)
}

func FitModel(request TrainingRequest) (*ModelSpec, error) {
	payload, available, err := invokeRustTraining(request)
	if !available || err != nil {
		if err != nil {
			return nil, err
		}
		return nil, errors.New("Rust training binary is not available")
	}
	spec, err := LoadModelSpec(payload)
	if err != nil {
		return nil, err
	}
	return spec, nil
}

func tryValidateWithRust(spec *ModelSpec) (bool, error) {
	_, available, err := invokeRustRuntime("validate", spec, nil)
	return available, err
}

func tryDesignMatrixWithRust(spec *ModelSpec, rows [][]float64) (bool, [][]float64, error) {
	payload, available, err := invokeRustRuntime("design-matrix", spec, rows)
	if !available || err != nil {
		return available, nil, err
	}
	var raw [][]*float64
	if err := json.Unmarshal(payload, &raw); err != nil {
		return true, nil, fmt.Errorf("failed to parse Rust runtime output: %w", err)
	}
	return true, runtimeMatrixToFloat(raw), nil
}

func tryPredictWithRust(spec *ModelSpec, rows [][]float64) (bool, []float64, error) {
	payload, available, err := invokeRustRuntime("predict", spec, rows)
	if !available || err != nil {
		return available, nil, err
	}
	var raw []*float64
	if err := json.Unmarshal(payload, &raw); err != nil {
		return true, nil, fmt.Errorf("failed to parse Rust runtime output: %w", err)
	}
	return true, runtimeVectorToFloat(raw), nil
}

func invokeRustRuntime(command string, spec *ModelSpec, rows [][]float64) ([]byte, bool, error) {
	binary := rustRuntimeBinary()
	if binary == "" {
		return nil, false, nil
	}

	specPath, cleanupSpec, ok := writeTempJSON(spec)
	if !ok {
		return nil, false, nil
	}
	defer cleanupSpec()

	var rowsPath string
	var cleanupRows func()
	if rows != nil {
		var okRows bool
		rowsPath, cleanupRows, okRows = writeTempJSON(nullableRows(rows))
		if !okRows {
			return nil, false, nil
		}
		defer cleanupRows()
	}

	args := []string{command, "--spec-file", specPath}
	if rowsPath != "" {
		args = append(args, "--rows-file", rowsPath)
	}

	cmd := exec.Command(binary, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, false, nil
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			message := strings.TrimSpace(stderr.String())
			if message == "" {
				message = exitErr.Error()
			}
			return nil, true, errors.New(message)
		}
		return nil, false, nil
	}

	return stdout.Bytes(), true, nil
}

func invokeRustTraining(request TrainingRequest) ([]byte, bool, error) {
	binary := rustRuntimeBinary()
	if binary == "" {
		return nil, false, nil
	}

	requestPath, cleanupRequest, ok := writeTempJSON(request)
	if !ok {
		return nil, false, nil
	}
	defer cleanupRequest()

	args := []string{"fit", "--request-file", requestPath}

	cmd := exec.Command(binary, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, false, nil
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			message := strings.TrimSpace(stderr.String())
			if message == "" {
				message = exitErr.Error()
			}
			return nil, true, errors.New(message)
		}
		return nil, false, nil
	}

	return stdout.Bytes(), true, nil
}

func rustRuntimeBinary() string {
	if envBinary := os.Getenv("MARS_RUNTIME_BIN"); envBinary != "" {
		if fileExists(envBinary) {
			return envBinary
		}
	}

	start, err := os.Getwd()
	if err != nil {
		return ""
	}

	for dir := filepath.Clean(start); ; dir = filepath.Dir(dir) {
		for _, candidate := range candidateRuntimeBinaries(dir) {
			if fileExists(candidate) {
				return candidate
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}

	return ""
}

func candidateRuntimeBinaries(root string) []string {
	name := "mars-runtime-cli"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return []string{
		filepath.Join(root, "rust-runtime", "target", "debug", name),
		filepath.Join(root, "rust-runtime", "target", "release", name),
	}
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func writeTempJSON(value any) (string, func(), bool) {
	raw, err := json.Marshal(value)
	if err != nil {
		return "", func() {}, false
	}
	file, err := os.CreateTemp("", "mars-runtime-*.json")
	if err != nil {
		return "", func() {}, false
	}
	if _, err := file.Write(raw); err != nil {
		file.Close()
		os.Remove(file.Name())
		return "", func() {}, false
	}
	if err := file.Close(); err != nil {
		os.Remove(file.Name())
		return "", func() {}, false
	}
	return file.Name(), func() { os.Remove(file.Name()) }, true
}

func nullableRows(rows [][]float64) [][]*float64 {
	out := make([][]*float64, len(rows))
	for i, row := range rows {
		out[i] = make([]*float64, len(row))
		for j, value := range row {
			if math.IsNaN(value) {
				continue
			}
			v := value
			out[i][j] = &v
		}
	}
	return out
}

func runtimeMatrixToFloat(rows [][]*float64) [][]float64 {
	out := make([][]float64, len(rows))
	for i, row := range rows {
		out[i] = runtimeVectorToFloat(row)
	}
	return out
}

func runtimeVectorToFloat(values []*float64) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		if value == nil {
			out[i] = math.NaN()
		} else {
			out[i] = *value
		}
	}
	return out
}

func validatePure(spec *ModelSpec) error {
	if spec.SpecVersion == "" || len(spec.SpecVersion) < 3 || spec.SpecVersion[1] != '.' {
		return errors.New("malformed artifact: spec_version must be '<major>.<minor>'")
	}
	if !strings.HasPrefix(spec.SpecVersion, "1.") {
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

func designMatrixPure(spec *ModelSpec, rows [][]float64) ([][]float64, error) {
	if err := validatePure(spec); err != nil {
		return nil, err
	}
	if err := validateRowsPure(spec, rows); err != nil {
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

func predictPure(spec *ModelSpec, rows [][]float64) ([]float64, error) {
	matrix, err := designMatrixPure(spec, rows)
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

func validateRowsPure(spec *ModelSpec, rows [][]float64) error {
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
