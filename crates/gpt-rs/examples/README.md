# Runtime Override Examples

This folder contains runnable snippets that demonstrate how to register runtime implementations for functional primitives.

- `runtime_override.rs` shows how to force the `matmul` functional to use a custom implementation. Run it with
  `cargo run -p gpt-rs --example runtime_override`.

Each example uses `FunctionalOverrides` to configure the registry and registers a bespoke implementation before executing the functional through the standard API.
