# Training Orchestration Inventory

This page records the current state of the training boundary between Python and
Rust.

The inventory is intentionally conservative:

- the Rust core owns the portable `ModelSpec` evaluation surface
- Python still owns some fallback preprocessing and unsupported cases
- the remaining parity gaps stay listed here until they are fully closed

See [Rust Core](rust_core.md) and [Training Core Migration](training_core_migration.md).
