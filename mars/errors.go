package mars

import "errors"

// Sentinel errors returned by mars package functions.
var (
	// ErrInvalidTimeRange is returned when a time range has Start after End.
	ErrInvalidTimeRange = errors.New("invalid time range: start is after end")

	// ErrEmptyInput is returned when a function receives no records.
	ErrEmptyInput = errors.New("empty input: at least one record is required")

	// ErrNilInput is returned when a nil slice is passed to a function.
	ErrNilInput = errors.New("nil input: a non-nil slice is required")

	// ErrNoMatch is returned when no records match the given filter.
	ErrNoMatch = errors.New("no records match the provided filter criteria")
)
