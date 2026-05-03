# Performance Optimization Plan

Performance work stays secondary to correctness and parity.

The current priorities are:

- keep the Rust runtime as the shared acceleration path
- preserve deterministic behavior
- avoid changing the public API to chase micro-optimizations
