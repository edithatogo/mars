# State-of-the-Art Analysis & Scope Contract

## SOTA Analysis

### Existing Go Data Analysis Libraries

| Library | Description | Comparison |
|---------|-------------|------------|
| [gonum/stat](https://github.com/gonum/gonum) | Comprehensive statistical library | More features but heavier; Mars focuses on time-series specifically |
| [go-gota/gota](https://github.com/go-gota/gota) | DataFrame-like API | Inspired by pandas; Mars is simpler and more opinionated |
| [mattn/go-json](https://github.com/mattn/go-json) | JSON processing | Mars uses stdlib; no additional dependency needed |
| [gocarina/gocsv](https://github.com/gocarina/gocsv) | CSV marshaling | Mars uses stdlib + custom parsing for simplicity |

### Differentiation

Mars differentiates from existing libraries by:

1. **Focus on time-series**: First-class `Timestamp` field, time-range filtering, time bucketing
2. **Simplicity**: Single package, no external dependencies (stdlib only)
3. **CLI-first design**: Built-in CLI with multiple output formats
4. **Doc generation**: `starlight-polyglot` integration for auto-generated API docs
5. **Minimal API surface**: ~10 public functions for the core use cases

## Scope Contract

### In Scope

- Time-series record data model with timestamp, label, value
- Statistical analysis: mean, median, stddev, percentiles
- Record filtering (time range, label, value thresholds)
- Grouped analysis per label
- CSV and JSON I/O
- Moving average computation
- Time bucketing / aggregation
- CLI with table and JSON output
- Starlight documentation with polyglot integration
- CI/CD pipelines

### Out of Scope (explicitly)

- Real-time/streaming data processing
- Distributed computing
- Database backends
- Machine learning model training/evaluation
- Visualization/charting
- Plugin system
- Async/concurrent processing within the library
- DataFrame/Series API (like pandas)

## API Stability Guarantee

The `mars` package API is considered **unstable** (v0.x). Breaking changes may occur without major version bumps until v1.0. Minor additions will follow semantic versioning.
