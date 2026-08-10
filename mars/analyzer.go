// Package mars provides data analysis and processing utilities
// for structured datasets.
package mars

import (
	"math"
	"sort"
	"time"
)

// Analyze computes a Summary from the provided records. It returns
// ErrEmptyInput if records is empty and ErrNilInput if records is nil.
//
// The summary includes count, mean, median, standard deviation,
// min, max, quartiles (P25, P75), and total duration.
func Analyze(records []Record) (Summary, error) {
	if records == nil {
		return Summary{}, ErrNilInput
	}
	if len(records) == 0 {
		return Summary{}, ErrEmptyInput
	}

	n := len(records)
	values := make([]float64, n)
	minTime := records[0].Timestamp
	maxTime := records[0].Timestamp

	var sum float64
	for i, r := range records {
		values[i] = r.Value
		sum += r.Value
		if r.Timestamp.Before(minTime) {
			minTime = r.Timestamp
		}
		if r.Timestamp.After(maxTime) {
			maxTime = r.Timestamp
		}
	}

	sort.Float64s(values)

	mean := sum / float64(n)

	var varianceSum float64
	for _, v := range values {
		diff := v - mean
		varianceSum += diff * diff
	}
	stdDev := math.Sqrt(varianceSum / float64(n))

	return Summary{
		Count:    n,
		Mean:     mean,
		Median:   percentile(values, 0.5),
		StdDev:   stdDev,
		Min:      values[0],
		Max:      values[n-1],
		P25:      percentile(values, 0.25),
		P75:      percentile(values, 0.75),
		Duration: maxTime.Sub(minTime),
	}, nil
}

// AnalyzeGrouped computes a GroupedSummary by grouping records by
// their Label field and calling Analyze on each group.
//
// Groups with no records are omitted from the result. Returns
// ErrNilInput if records is nil.
func AnalyzeGrouped(records []Record) (GroupedSummary, error) {
	if records == nil {
		return nil, ErrNilInput
	}

	groups := make(map[string][]Record)
	for _, r := range records {
		groups[r.Label] = append(groups[r.Label], r)
	}

	result := make(GroupedSummary, len(groups))
	for label, recs := range groups {
		s, err := Analyze(recs)
		if err != nil {
			return nil, err
		}
		result[label] = s
	}
	return result, nil
}

// Filter returns a subset of records that match the given FilterOptions.
// An empty or zero-value FilterOptions returns all input records unchanged.
func Filter(records []Record, opts FilterOptions) ([]Record, error) {
	if records == nil {
		return nil, ErrNilInput
	}
	if err := opts.Validate(); err != nil {
		return nil, err
	}

	var filtered []Record
	for _, r := range records {
		if !opts.Start.IsZero() && r.Timestamp.Before(opts.Start) {
			continue
		}
		if !opts.End.IsZero() && !r.Timestamp.Before(opts.End) {
			continue
		}
		if len(opts.Labels) > 0 && !contains(opts.Labels, r.Label) {
			continue
		}
		if opts.MinValue != nil && r.Value < *opts.MinValue {
			continue
		}
		if opts.MaxValue != nil && r.Value > *opts.MaxValue {
			continue
		}
		filtered = append(filtered, r)
	}

	if len(filtered) == 0 {
		return nil, ErrNoMatch
	}
	return filtered, nil
}

// Merge combines multiple record slices into a single sorted slice.
// Records are sorted by timestamp in ascending order. If all inputs
// are empty, an empty (non-nil) slice is returned.
func Merge(sets ...[]Record) []Record {
	total := 0
	for _, s := range sets {
		total += len(s)
	}
	if total == 0 {
		return []Record{}
	}
	merged := make([]Record, 0, total)
	for _, s := range sets {
		merged = append(merged, s...)
	}
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Timestamp.Before(merged[j].Timestamp)
	})
	return merged
}

// PercentileBounds returns the record values at the lower and upper
// percentile bounds (e.g., 0.05 and 0.95). It is a convenience wrapper
// around percentile calculations. Returns ErrEmptyInput if records is empty.
func PercentileBounds(records []Record, lower, upper float64) (lowerVal, upperVal float64, err error) {
	if len(records) == 0 {
		return 0, 0, ErrEmptyInput
	}
	values := make([]float64, len(records))
	for i, r := range records {
		values[i] = r.Value
	}
	sort.Float64s(values)
	return percentile(values, lower), percentile(values, upper), nil
}

// MovingAverage computes a simple moving average over the records'
// values using the specified window size. The output slice has the
// same length as the input; the first windowSize-1 entries are NaN.
//
// Returns ErrEmptyInput if records is empty or ErrNilInput if nil.
func MovingAverage(records []Record, windowSize int) ([]float64, error) {
	if records == nil {
		return nil, ErrNilInput
	}
	if len(records) == 0 {
		return nil, ErrEmptyInput
	}
	if windowSize < 1 {
		windowSize = 1
	}
	if windowSize > len(records) {
		windowSize = len(records)
	}

	result := make([]float64, len(records))
	for i := range result {
		result[i] = math.NaN()
	}

	var runningSum float64
	for i, r := range records {
		runningSum += r.Value
		if i >= windowSize {
			runningSum -= records[i-windowSize].Value
		}
		if i >= windowSize-1 {
			result[i] = runningSum / float64(windowSize)
		}
	}
	return result, nil
}

// TimeBucket aggregates records into fixed-duration time buckets and
// returns the mean value per bucket. Buckets are determined by dividing
// the full time range into equal intervals of the given duration.
//
// Returns ErrEmptyInput if records is empty or ErrNilInput if nil.
func TimeBucket(records []Record, bucketDur time.Duration) ([]time.Time, []float64, error) {
	if records == nil {
		return nil, nil, ErrNilInput
	}
	if len(records) == 0 {
		return nil, nil, ErrEmptyInput
	}
	if bucketDur <= 0 {
		bucketDur = time.Hour
	}

	minTime := records[0].Timestamp
	maxTime := records[0].Timestamp
	for _, r := range records {
		if r.Timestamp.Before(minTime) {
			minTime = r.Timestamp
		}
		if r.Timestamp.After(maxTime) {
			maxTime = r.Timestamp
		}
	}

	start := minTime.Truncate(bucketDur)
	end := maxTime.Truncate(bucketDur).Add(bucketDur)

	numBuckets := int(end.Sub(start) / bucketDur)
	if numBuckets < 1 {
		numBuckets = 1
	}

	buckets := make([][]float64, numBuckets)
	times := make([]time.Time, numBuckets)

	for i := range buckets {
		times[i] = start.Add(time.Duration(i) * bucketDur)
	}

	for _, r := range records {
		idx := int(r.Timestamp.Sub(start) / bucketDur)
		if idx >= 0 && idx < numBuckets {
			buckets[idx] = append(buckets[idx], r.Value)
		}
	}

	means := make([]float64, numBuckets)
	for i, vals := range buckets {
		if len(vals) == 0 {
			means[i] = math.NaN()
			continue
		}
		var s float64
		for _, v := range vals {
			s += v
		}
		means[i] = s / float64(len(vals))
	}

	return times, means, nil
}

// percentile computes the p-th percentile (0 <= p <= 1) from a
// sorted slice of float64 values using linear interpolation.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}

	index := p * float64(len(sorted)-1)
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))

	if lower == upper {
		return sorted[lower]
	}

	fraction := index - float64(lower)
	return sorted[lower]*(1-fraction) + sorted[upper]*fraction
}

// contains checks whether a string slice contains a given value.
func contains(slice []string, val string) bool {
	for _, s := range slice {
		if s == val {
			return true
		}
	}
	return false
}