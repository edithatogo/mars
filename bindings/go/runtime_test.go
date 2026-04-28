package pymars

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type runtimeFixture struct {
	Probe        [][]*float64 `json:"probe"`
	DesignMatrix [][]*float64 `json:"design_matrix"`
	Predict      []*float64   `json:"predict"`
}

func TestFixtureParity(t *testing.T) {
	fixturesDir := filepath.Join("..", "..", "tests", "fixtures")
	entries, err := os.ReadDir(fixturesDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, "model_spec_") || !strings.HasSuffix(name, ".json") {
			continue
		}
		suffix := strings.TrimSuffix(strings.TrimPrefix(name, "model_spec_"), ".json")
		specRaw, err := os.ReadFile(filepath.Join(fixturesDir, name))
		if err != nil {
			t.Fatal(err)
		}
		spec, err := LoadModelSpec(specRaw)
		if err != nil {
			t.Fatalf("%s load failed: %v", suffix, err)
		}
		fixtureRaw, err := os.ReadFile(filepath.Join(fixturesDir, "runtime_portability_fixture_"+suffix+".json"))
		if err != nil {
			t.Fatal(err)
		}
		var fixture runtimeFixture
		if err := json.Unmarshal(fixtureRaw, &fixture); err != nil {
			t.Fatal(err)
		}
		probe := nullableMatrixToFloat(fixture.Probe)
		matrix, err := DesignMatrix(spec, probe)
		if err != nil {
			t.Fatalf("%s design_matrix failed: %v", suffix, err)
		}
		assertMatrixClose(t, matrix, nullableMatrixToFloat(fixture.DesignMatrix))
		predictions, err := Predict(spec, probe)
		if err != nil {
			t.Fatalf("%s predict failed: %v", suffix, err)
		}
		assertVectorClose(t, predictions, nullableVectorToFloat(fixture.Predict))
	}
}

func nullableMatrixToFloat(rows [][]*float64) [][]float64 {
	out := make([][]float64, len(rows))
	for i, row := range rows {
		out[i] = nullableVectorToFloat(row)
	}
	return out
}

func nullableVectorToFloat(values []*float64) []float64 {
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

func assertMatrixClose(t *testing.T, actual, expected [][]float64) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("row count mismatch: %d != %d", len(actual), len(expected))
	}
	for i := range actual {
		assertVectorClose(t, actual[i], expected[i])
	}
}

func assertVectorClose(t *testing.T, actual, expected []float64) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("length mismatch: %d != %d", len(actual), len(expected))
	}
	for i := range actual {
		if math.IsNaN(expected[i]) && math.IsNaN(actual[i]) {
			continue
		}
		if math.Abs(actual[i]-expected[i]) > 1e-12 {
			t.Fatalf("value %d mismatch: %.17g != %.17g", i, actual[i], expected[i])
		}
	}
}
