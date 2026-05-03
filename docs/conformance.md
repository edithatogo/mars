# Conformance

Conformance is checked through the shared fixture corpus and the binding
parity tests.

The main rules are:

- validate `ModelSpec` before evaluation
- keep runtime-only packages runtime-only until training is supported
- run the fixture suites for the supported bindings

See [Training Orchestration Inventory](training_orchestration_inventory.md)
for the remaining parity gaps.
