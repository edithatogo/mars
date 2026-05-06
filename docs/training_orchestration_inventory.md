# Training Orchestration Inventory

This page records the current state of the training boundary between Python and
Rust.

The inventory is intentionally conservative:

- the Rust core owns the portable `ModelSpec` evaluation surface
- Python routes supported training and replay through Rust first and only
  owns fallback preprocessing or unsupported cases when Rust cannot handle a
  request
- the remaining parity gaps stay listed here until they are fully closed

See [Rust Core](rust_core.md) and [Training Core Migration](training_core_migration.md).
