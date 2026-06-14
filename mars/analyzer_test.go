package mars

import (
	"math"
	"testing"
	"time"
)

func makeTestRecords() []Record {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	return []Record{
		{Timestamp: base, Label: "a", Value: 1.0},
		{Timestamp: base.Add(1 * time.Hour), Label: "a", Value: 2.0},
		{Timestamp: base.Add(2 * time.Hour), Label: "b", Value: 3.0},
		{Timestamp: base.Add(3 * time.Hour), Label: "a", Value: 4.0},
		{Timestamp: base.Add(4 * time.Hour), Label: "b", Value: 5.0},
	}
}

func TestAnalyze(t *testing.T) {
	records := makeTestRecords()
	s, err := Analyze(records)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if s.Count != 5 {
		t.Errorf("Count = %d, want 5", s.Count)
	}
	if s.Mean != 3.0 {
		t.Errorf("Mean = %f, want 3.0", s.Mean)
	}
	if s.Median != 3.0 {
		t.Errorf("Median = %f, want 3.0", s.Median)
	}
	if s.Min != 1.0 {
		t.Errorf("Min = %f, want 1.0", s.Min)
	}
	if s.Max != 5.0 {
		t.Errorf("Max = %f, want 5.0", s.Max)
	}
	if s.P25 != 2.0 {
		t.Errorf("P25 = %f, want 2.0", s.P25)
	}
	if s.P75 != 4.0 {
		t.Errorf("P75 = %f, want 4.0", s.P75)
	}
}

func TestAnalyze_Empty(t *testing.T) {
	_, err := Analyze([]Record{})
	if err != ErrEmptyInput {
		t.Errorf("got error %v, want ErrEmptyInput", err)
	}
}

func TestAnalyze_Nil(t *testing.T) {
	_, err := Analyze(nil)
	if err != ErrNilInput {
		t.Errorf("got error %v, want ErrNilInput", err)
	}
}

func TestAnalyzeGrouped(t *testing.T) {
	records := makeTestRecords()
	gs, err := AnalyzeGrouped(records)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(gs) != 2 {
		t.Fatalf("got %d groups, want 2", len(gs))
	}

	aSum, ok := gs["a"]
	if !ok {
		t.Fatal("missing group 'a'")
	}
	if aSum.Count != 3 {
		t.Errorf("group a Count = %d, want 3", aSum.Count)
	}
	if math.Abs(aSum.Mean-7.0/3.0) > 1e-9 {
		t.Errorf("group a Mean = %f, want ~2.333", aSum.Mean)
	}

	bSum, ok := gs["b"]
	if !ok {
		t.Fatal("missing group 'b'")
	}
	if bSum.Count != 2 {
		t.Errorf("group b Count = %d, want 2", bSum.Count)
	}
}

func TestFilter(t *testing.T) {
	records := makeTestRecords()

	t.Run("label filter", func(t *testing.T) {
		filtered, err := Filter(records, FilterOptions{Labels: []string{"a"}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(filtered) != 3 {
			t.Errorf("got %d records, want 3", len(filtered))
		}
		for _, r := range filtered {
			if r.Label != "a" {
				t.Errorf("unexpected label %q", r.Label)
			}
		}
	})

	t.Run("time range filter", func(t *testing.T) {
		start := time.Date(2024, 1, 1, 1, 0, 0, 0, time.UTC)
		end := time.Date(2024, 1, 1, 3, 0, 0, 0, time.UTC)
		filtered, err := Filter(records, FilterOptions{Start: start, End: end})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(filtered) != 2 {
			t.Errorf("got %d records, want 2", len(filtered))
		}
	})

	t.Run("no match returns error", func(t *testing.T) {
		_, err := Filter(records, FilterOptions{Labels: []string{"z"}})
		if err != ErrNoMatch {
			t.Errorf("got error %v, want ErrNoMatch", err)
		}
	})
}

func TestMerge(t *testing.T) {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	set1 := []Record{
		{Timestamp: base.Add(2 * time.Hour), Label: "a", Value: 3},
	}
	set2 := []Record{
		{Timestamp: base, Label: "b", Value: 1},
		{Timestamp: base.Add(1 * time.Hour), Label: "b", Value: 2},
	}

	merged := Merge(set1, set2)
	if len(merged) != 3 {
		t.Fatalf("got %d records, want 3", len(merged))
	}

	for i := 1; i < len(merged); i++ {
		if merged[i].Timestamp.Before(merged[i-1].Timestamp) {
			t.Error("records not sorted by timestamp")
		}
	}
}

func TestPercentileBounds(t *testing.T) {
	records := makeTestRecords()
	lower, upper, err := PercentileBounds(records, 0.1, 0.9)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lower != 1.4 {
		t.Errorf("lower = %f, want 1.4", lower)
	}
	if upper != 4.6 {
		t.Errorf("upper = %f, want 4.6", upper)
	}
}

func TestMovingAverage(t *testing.T) {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	records := []Record{
		{Timestamp: base, Value: 1},
		{Timestamp: base.Add(time.Hour), Value: 2},
		{Timestamp: base.Add(2 * time.Hour), Value: 3},
		{Timestamp: base.Add(3 * time.Hour), Value: 4},
		{Timestamp: base.Add(4 * time.Hour), Value: 5},
	}

	ma, err := MovingAverage(records, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []float64{math.NaN(), math.NaN(), 2.0, 3.0, 4.0}
	for i, exp := range expected {
		if math.IsNaN(exp) {
			if !math.IsNaN(ma[i]) {
				t.Errorf("ma[%d] = %f, want NaN", i, ma[i])
			}
		} else if ma[i] != exp {
			t.Errorf("ma[%d] = %f, want %f", i, ma[i], exp)
		}
	}
}

func TestTimeBucket(t *testing.T) {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	records := []Record{
		{Timestamp: base, Value: 1},
		{Timestamp: base.Add(30 * time.Minute), Value: 2},
		{Timestamp: base.Add(1 * time.Hour), Value: 10},
		{Timestamp: base.Add(1*time.Hour + 30*time.Minute), Value: 20},
	}

	times, means, err := TimeBucket(records, 1*time.Hour)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(times) != 2 {
		t.Fatalf("got %d buckets, want 2", len(times))
	}

	if means[0] != 1.5 {
		t.Errorf("bucket 0 mean = %f, want 1.5", means[0])
	}
	if means[1] != 15.0 {
		t.Errorf("bucket 1 mean = %f, want 15.0", means[1])
	}
}
