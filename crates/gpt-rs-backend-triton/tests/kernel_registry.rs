use std::collections::HashSet;

use gpt_rs_backend_triton::kernels::{builtin_kernel_sources, builtin_kernel_specs};

#[test]
fn builtin_kernel_registry_has_unique_ids_and_symbols() {
    let kernels = builtin_kernel_specs();
    assert!(!kernels.is_empty(), "builtin kernel registry is empty");

    let mut ids = HashSet::new();
    let mut symbols = HashSet::new();
    for kernel in kernels {
        assert!(
            ids.insert(kernel.id.clone()),
            "duplicate kernel id in registry: {}",
            kernel.id
        );
        assert!(
            symbols.insert(kernel.symbol.clone()),
            "duplicate kernel symbol in registry: {}",
            kernel.symbol
        );
    }
}

#[test]
fn builtin_kernel_sources_are_non_empty() {
    for (idx, source) in builtin_kernel_sources().iter().enumerate() {
        assert!(
            !source.trim().is_empty(),
            "builtin kernel source at index {idx} is empty"
        );
    }
}
