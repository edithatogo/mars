# Design

## Architecture Overview

```mermaid
graph TB
    subgraph "CLI Layer"
        CLI[cmd/mars/main.go]
    end

    subgraph "Library Layer"
        MARS[mars package]
        MODELS[models.go]
        ANALYZER[analyzer.go]
        IO[io.go]
        ERRORS[errors.go]
    end

    subgraph "External"
        CSV[CSV Files]
        JSON[JSON Files]
        STDOUT[STDOUT/File]
    end

    CLI --> MARS
    MARS --> MODELS
    MARS --> ANALYZER
    MARS --> IO
    MARS --> ERRORS
    IO --> CSV
    IO --> JSON
    CLI --> STDOUT
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as mars CLI
    participant Lib as mars Library
    participant File as File System

    User->>CLI: mars --input data.csv --format table
    CLI->>File: Read data.csv
    File-->>CLI: Raw bytes
    CLI->>Lib: ParseCSV(reader)
    Lib-->>CLI: []Record
    CLI->>Lib: Analyze(records)
    Lib-->>CLI: Summary
    CLI->>CLI: Format output
    CLI-->>User: Table output
```

## Package Structure

```mermaid
graph LR
    subgraph "mars/"
    A[models.go]
    B[analyzer.go]
    C[io.go]
    D[errors.go]
    E[analyzer_test.go]
    end

    subgraph "cmd/mars/"
    F[main.go]
    end

    subgraph "docs/astro-site/"
    G[Starlight Docs]
    end

    A --> B
    A --> C
    B --> D
    C --> D
    E --> B
    F --> A
    F --> B
    F --> C
```

## Key Design Decisions

### 1. Flat Package Structure

The `mars` library uses a single flat package rather than sub-packages. This keeps the API surface simple and discoverable. The package is small enough that sub-packages would add unnecessary nesting.

### 2. Error Handling

Sentinel errors (`ErrEmptyInput`, `ErrNilInput`, etc.) allow callers to use `errors.Is()` for precise error matching. All functions return errors rather than panicking.

### 3. Immutable Input

Analysis functions never modify input slices. `Filter` returns a new slice, and `Analyze` copies values internally. This prevents side effects and makes concurrent usage safe.

### 4. CLI as Thin Wrapper

The CLI (`cmd/mars`) is a thin wrapper around the library. All logic lives in the `mars` package, making it reusable in other Go programs.

### 5. Documentation-Driven

Documentation is treated as a first-class artifact. Starlight with `starlight-polyglot` enables both hand-written guides and auto-generated API docs from Go source.
