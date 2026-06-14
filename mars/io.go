package mars

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
)

// ReadCSV reads records from a CSV file path. The CSV must have
// headers: timestamp,label,value. The timestamp column must be in
// RFC3339 format. Optional fourth column "tag_key=tag_val" pairs
// are parsed as Tags.
func ReadCSV(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()
	return ParseCSV(f)
}

// ParseCSV reads records from a CSV reader. Expected headers:
// timestamp,label,value. Timestamps must be RFC3339 format.
func ParseCSV(r io.Reader) ([]Record, error) {
	reader := csv.NewReader(bufio.NewReader(r))
	reader.TrimLeadingSpace = true

	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("read csv headers: %w", err)
	}

	colIndex := map[string]int{}
	for i, h := range headers {
		colIndex[strings.ToLower(strings.TrimSpace(h))] = i
	}

	if _, ok := colIndex["timestamp"]; !ok {
		return nil, fmt.Errorf("csv missing required column: timestamp")
	}
	if _, ok := colIndex["label"]; !ok {
		return nil, fmt.Errorf("csv missing required column: label")
	}
	if _, ok := colIndex["value"]; !ok {
		return nil, fmt.Errorf("csv missing required column: value")
	}

	var records []Record
	for line := 2; ; line++ {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("csv line %d: %w", line, err)
		}

		ts, err := time.Parse(time.RFC3339, row[colIndex["timestamp"]])
		if err != nil {
			return nil, fmt.Errorf("csv line %d: invalid timestamp %q: %w", line, row[colIndex["timestamp"]], err)
		}

		val, err := strconv.ParseFloat(row[colIndex["value"]], 64)
		if err != nil {
			return nil, fmt.Errorf("csv line %d: invalid value %q: %w", line, row[colIndex["value"]], err)
		}

		r := Record{
			Timestamp: ts,
			Label:     row[colIndex["label"]],
			Value:     val,
			Tags:      make(map[string]string),
		}

		// Parse additional columns as tag=value pairs
		for h, idx := range colIndex {
			if h == "timestamp" || h == "label" || h == "value" {
				continue
			}
			if idx < len(row) {
				parts := strings.SplitN(row[idx], "=", 2)
				if len(parts) == 2 {
					r.Tags[parts[0]] = parts[1]
				}
			}
		}

		if len(r.Tags) == 0 {
			r.Tags = nil
		}

		records = append(records, r)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("csv contains no data records")
	}

	return records, nil
}

// WriteCSV writes records to a CSV file at the given path.
func WriteCSV(path string, records []Record) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create csv: %w", err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	defer writer.Flush()

	if err := writer.Write([]string{"timestamp", "label", "value"}); err != nil {
		return fmt.Errorf("write csv header: %w", err)
	}

	for _, r := range records {
		row := []string{
			r.Timestamp.Format(time.RFC3339),
			r.Label,
			strconv.FormatFloat(r.Value, 'f', -1, 64),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("write csv row: %w", err)
		}
	}

	return nil
}

// ReadJSON reads records from a JSON file path. Expects an array
// of Record objects.
func ReadJSON(path string) ([]Record, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read json file: %w", err)
	}

	var records []Record
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("unmarshal json: %w", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("json contains no records")
	}

	return records, nil
}

// WriteJSON writes records as a JSON array to the given file path.
func WriteJSON(path string, records []Record) error {
	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal json: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write json file: %w", err)
	}
	return nil
}

// WriteSummaryJSON writes a Summary as formatted JSON to the given path.
func WriteSummaryJSON(path string, s Summary) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal summary: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write summary: %w", err)
	}
	return nil
}
