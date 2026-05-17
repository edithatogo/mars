// Command mars is a CLI tool for analyzing structured time-series
// data. It supports CSV and JSON input, statistical analysis,
// filtering, grouping, and other data transformations.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/edithatogo/mars/mars"
)

var (
	inputFile  = flag.String("input", "", "Path to input data file (CSV or JSON)")
	outputFile = flag.String("output", "", "Path to write output (default: stdout)")
	format     = flag.String("format", "json", "Output format: json or table")
	filterLabel = flag.String("label", "", "Filter by label (comma-separated)")
	filterStart = flag.String("start", "", "Filter start time (RFC3339)")
	filterEnd   = flag.String("end", "", "Filter end time (RFC3339)")
	groupBy     = flag.Bool("group", false, "Group analysis by label")
	movingAvg   = flag.Int("moving-avg", 0, "Compute moving average with given window size")
	bucketDur   = flag.String("bucket", "", "Time bucket duration for aggregation (e.g., 1h, 30m)")
	version     = flag.Bool("version", false, "Print version and exit")
)

const appVersion = "0.1.0"

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `mars v%s - Data analysis CLI

Usage: mars [flags]

Flags:
`, appVersion)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *version {
		fmt.Printf("mars v%s\n", appVersion)
		os.Exit(0)
	}

	if *inputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: --input is required")
		flag.Usage()
		os.Exit(2)
	}

	records, err := loadRecords(*inputFile)
	if err != nil {
		log.Fatalf("Error loading records: %v", err)
	}

	opts := buildFilterOptions()
	if opts != nil {
		records, err = mars.Filter(records, *opts)
		if err != nil {
			log.Fatalf("Error filtering records: %v", err)
		}
	}

	if *movingAvg > 0 {
		ma, err := mars.MovingAverage(records, *movingAvg)
		if err != nil {
			log.Fatalf("Error computing moving average: %v", err)
		}
		outputMovingAverage(records, ma)
		return
	}

	if *bucketDur != "" {
		dur, err := time.ParseDuration(*bucketDur)
		if err != nil {
			log.Fatalf("Invalid bucket duration %q: %v", *bucketDur, err)
		}
		times, means, err := mars.TimeBucket(records, dur)
		if err != nil {
			log.Fatalf("Error bucketing: %v", err)
		}
		outputBuckets(times, means)
		return
	}

	if *groupBy {
		grouped, err := mars.AnalyzeGrouped(records)
		if err != nil {
			log.Fatalf("Error analyzing grouped: %v", err)
		}
		outputGroupedSummary(grouped)
	} else {
		summary, err := mars.Analyze(records)
		if err != nil {
			log.Fatalf("Error analyzing: %v", err)
		}
		outputSummary(summary)
	}
}

func loadRecords(path string) ([]mars.Record, error) {
	switch {
	case strings.HasSuffix(strings.ToLower(path), ".csv"):
		return mars.ReadCSV(path)
	case strings.HasSuffix(strings.ToLower(path), ".json"):
		return mars.ReadJSON(path)
	default:
		return nil, fmt.Errorf("unsupported file format: %s (supported: .csv, .json)", path)
	}
}

func buildFilterOptions() *mars.FilterOptions {
	var opts mars.FilterOptions
	hasFilter := false

	if *filterLabel != "" {
		opts.Labels = strings.Split(*filterLabel, ",")
		for i := range opts.Labels {
			opts.Labels[i] = strings.TrimSpace(opts.Labels[i])
		}
		hasFilter = true
	}
	if *filterStart != "" {
		t, err := time.Parse(time.RFC3339, *filterStart)
		if err != nil {
			log.Fatalf("Invalid --start time: %v", err)
		}
		opts.Start = t
		hasFilter = true
	}
	if *filterEnd != "" {
		t, err := time.Parse(time.RFC3339, *filterEnd)
		if err != nil {
			log.Fatalf("Invalid --end time: %v", err)
		}
		opts.End = t
		hasFilter = true
	}

	if !hasFilter {
		return nil
	}
	return &opts
}

func outputSummary(s mars.Summary) {
	switch *format {
	case "table":
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintf(w, "Metric\tValue\n-----\t-----\n")
		fmt.Fprintf(w, "Count\t%d\n", s.Count)
		fmt.Fprintf(w, "Mean\t%.4f\n", s.Mean)
		fmt.Fprintf(w, "Median\t%.4f\n", s.Median)
		fmt.Fprintf(w, "StdDev\t%.4f\n", s.StdDev)
		fmt.Fprintf(w, "Min\t%.4f\n", s.Min)
		fmt.Fprintf(w, "Max\t%.4f\n", s.Max)
		fmt.Fprintf(w, "P25\t%.4f\n", s.P25)
		fmt.Fprintf(w, "P75\t%.4f\n", s.P75)
		fmt.Fprintf(w, "Duration\t%s\n", s.Duration)
		w.Flush()
	default:
		writeJSON(s)
	}
}

func outputGroupedSummary(gs mars.GroupedSummary) {
	switch *format {
	case "table":
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintf(w, "Label\tCount\tMean\tMedian\tStdDev\tMin\tMax\n-----\t-----\t----\t------\t------\t---\t---\n")
		for label, s := range gs {
			fmt.Fprintf(w, "%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
				label, s.Count, s.Mean, s.Median, s.StdDev, s.Min, s.Max)
		}
		w.Flush()
	default:
		writeJSON(gs)
	}
}

func outputMovingAverage(records []mars.Record, ma []float64) {
	switch *format {
	case "table":
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintf(w, "Timestamp\tValue\tMovingAvg\n---------\t-----\t---------\n")
		for i, r := range records {
			maStr := "NaN"
			if !isNaN(ma[i]) {
				maStr = fmt.Sprintf("%.4f", ma[i])
			}
			fmt.Fprintf(w, "%s\t%.4f\t%s\n", r.Timestamp.Format(time.RFC3339), r.Value, maStr)
		}
		w.Flush()
	default:
		type row struct {
			Timestamp  string   `json:"timestamp"`
			Value      float64  `json:"value"`
			MovingAvg  *float64 `json:"moving_avg,omitempty"`
		}
		rows := make([]row, len(records))
		for i, r := range records {
			rows[i].Timestamp = r.Timestamp.Format(time.RFC3339)
			rows[i].Value = r.Value
			if !isNaN(ma[i]) {
				rows[i].MovingAvg = &ma[i]
			}
		}
		writeJSON(rows)
	}
}

func outputBuckets(times []time.Time, means []float64) {
	switch *format {
	case "table":
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintf(w, "Bucket\tMean\n------\t----\n")
		for i, t := range times {
			meanStr := "NaN"
			if !isNaN(means[i]) {
				meanStr = fmt.Sprintf("%.4f", means[i])
			}
			fmt.Fprintf(w, "%s\t%s\n", t.Format(time.RFC3339), meanStr)
		}
		w.Flush()
	default:
		type bucket struct {
			Time string   `json:"time"`
			Mean *float64 `json:"mean,omitempty"`
		}
		buckets := make([]bucket, len(times))
		for i, t := range times {
			buckets[i].Time = t.Format(time.RFC3339)
			if !isNaN(means[i]) {
				buckets[i].Mean = &means[i]
			}
		}
		writeJSON(buckets)
	}
}

func writeJSON(v interface{}) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		log.Fatalf("Error encoding output: %v", err)
	}
}

func isNaN(f float64) bool {
	return f != f
}