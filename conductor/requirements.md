# Requirements (MoSCoW)

## Must Have

- [x] Go module with `mars` library package
- [x] `Record`, `Summary`, `GroupedSummary` data models
- [x] Statistical analysis: mean, median, stddev, min, max, quartiles
- [x] Record filtering by time range, label, value
- [x] Grouped analysis by label
- [x] CSV I/O (read and write)
- [x] JSON I/O (read and write)
- [x] CLI entry point (`cmd/mars`)
- [x] Comprehensive test coverage
- [x] Starlight documentation site with index, getting-started, API reference
- [x] CI pipeline (lint, test, build cross-platform)
- [x] GitHub Pages deployment workflow

## Should Have

- [x] Moving average computation
- [x] Time bucketing/aggregation
- [x] Table-formatted CLI output
- [x] Sentinel error types for all failure modes
- [x] `.gitignore` covering Go and Node patterns
- [x] `starlight-polyglot` integration for auto-generated Go docs

## Could Have

- [ ] Streaming I/O for large datasets
- [ ] Prometheus metrics integration
- [ ] Docker image for CLI
- [ ] gRPC service exposing analysis endpoints
- [ ] Web dashboard (separate frontend)
- [ ] Benchmark suite
- [ ] Fuzz testing

## Won't Have (current phase)

- [ ] Real-time streaming analysis
- [ ] Distributed processing
- [ ] Machine learning model integration
- [ ] Database backend (Postgres, etc.)
