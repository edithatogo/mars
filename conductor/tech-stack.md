# Technology Stack

## Go (Core Library)

- **Language**: Go 1.22+
- **Module**: `github.com/edithatogo/mars`
- **Purpose**: Data analysis library and CLI tool
- **Rationale**: Performance, simplicity, excellent standard library, cross-compilation

### Key Go Packages Used

| Package | Purpose |
|---------|---------|
| `encoding/csv` | CSV I/O for data import/export |
| `encoding/json` | JSON I/O for data import/export |
| `math` | Statistical computations (stddev, percentiles) |
| `sort` | Sorting records and values |
| `time` | Timestamp handling and duration calculations |
| `flag` | CLI argument parsing |
| `text/tabwriter` | Table-formatted output |

## Documentation (Starlight)

- **Framework**: [Astro](https://astro.build/) + [Starlight](https://starlight.astro.build/)
- **Version**: Starlight >=0.39.0, Astro >=6.0.0
- **Plugin**: `starlight-polyglot` (local: `file:../../starlight-polyglot/packages/starlight-polyglot`)
- **Purpose**: Auto-generated and hand-written API documentation
- **Deployment**: GitHub Pages via GitHub Actions

### Documentation Structure

```
docs/astro-site/
  src/content/docs/
    index.mdx          — Landing page
    getting-started.mdx — Quick start guide
    api-reference.mdx   — Manual API reference
  src/styles/
    custom.css          — Custom theme overrides
  astro.config.mjs      — Astro/Starlight configuration
  package.json          — Dependencies and scripts
```

## CI/CD

- **CI**: GitHub Actions — `ci.yml` (lint, test, build matrix)
- **Docs**: GitHub Actions — `docs.yml` (build + deploy to Pages)
- **Package Manager**: pnpm (for docs)

## Development Tools

| Tool | Purpose |
|------|---------|
| `golangci-lint` | Go linting |
| `gofmt` | Go code formatting |
| `pnpm` | Node.js package management |
| `gomarkdoc` | Go doc extraction (for starlight-polyglot) |
