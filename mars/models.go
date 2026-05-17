// Package mars provides data analysis and processing utilities
// for structured datasets. It defines core data models used
// throughout the application.
package mars

import "time"

// Record represents a single data record with a timestamp,
// label, and associated value.
type Record struct {
	// Timestamp is the time at which the record was captured.
	Timestamp time.Time `json:"timestamp"`

	// Label is a human-readable category or identifier for the record.
	Label string `json:"label"`

	// Value is the numeric measurement associated with this record.
	Value float64 `json:"value"`

	// Tags is an optional set of key-value metadata pairs.
	Tags map[string]string `json:"tags,omitempty"`
}

// Summary contains aggregated statistics computed from a collection
// of Records.
type Summary struct {
	// Count is the total number of records in the input.
	Count int `json:"count"`

	// Mean is the arithmetic mean of all record values.
	Mean float64 `json:"mean"`

	// Median is the median value of the record set.
	Median float64 `json:"median"`

	// StdDev is the population standard deviation.
	StdDev float64 `json:"stddev"`

	// Min is the minimum observed value.
	Min float64 `json:"min"`

	// Max is the maximum observed value.
	Max float64 `json:"max"`

	// P25 is the 25th percentile value.
	P25 float64 `json:"p25"`

	// P75 is the 75th percentile value.
	P75 float64 `json:"p75"`

	// Duration is the total time span covered by the records.
	Duration time.Duration `json:"duration"`
}

// GroupedSummary stores per-label summaries keyed by label value.
type GroupedSummary map[string]Summary

// FilterOptions defines criteria for filtering a set of Records.
type FilterOptions struct {
	// Start is the inclusive start of the time range filter.
	Start time.Time `json:"start,omitempty"`

	// End is the exclusive end of the time range filter.
	End time.Time `json:"end,omitempty"`

	// Labels restricts results to records matching any of these labels.
	// An empty slice means no label filtering is applied.
	Labels []string `json:"labels,omitempty"`

	// MinValue excludes records with a value below this threshold.
	MinValue *float64 `json:"min_value,omitempty"`

	// MaxValue excludes records with a value above this threshold.
	MaxValue *float64 `json:"max_value,omitempty"`
}

// Validate checks that FilterOptions is internally consistent
// and returns an error if the time range is inverted.
func (f FilterOptions) Validate() error {
	if !f.Start.IsZero() && !f.End.IsZero() && f.Start.After(f.End) {
		return ErrInvalidTimeRange
	}
	return nil
}

// ImportResult holds the outcome of importing records from a data source.
type ImportResult struct {
	// Imported is the number of records successfully imported.
	Imported int `json:"imported"`

	// Skipped is the number of records that were skipped due to errors.
	Skipped int `json:"skipped"`

	// Errors contains any per-record error messages encountered.
	Errors []string `json:"errors,omitempty"`
}
