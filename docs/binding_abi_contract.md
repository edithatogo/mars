# Binding ABI and API Contract

This contract defines the shared rules for the language bindings:

- the Rust core owns the shared computational semantics
- host wrappers own parsing and native conversion
- buffers and handles must have explicit ownership rules
- runtime-only packages must reject training until supported
- `ModelSpec` validation happens before evaluation
