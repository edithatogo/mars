# Training Core Migration

The training migration is moving from Python orchestration toward a Rust core.

Current state:

- Rust-backed training exists for the supported baseline cases
- Python still owns unsupported edge handling and fallback behavior
- full search and pruning parity is still tracked separately

See [Training Orchestration Inventory](training_orchestration_inventory.md) for
the current boundary notes.
