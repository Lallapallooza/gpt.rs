use gpt_rs::backend::registry::{create_backend, has_backend, list_backends};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

#[test]
fn test_backend_registry() {
    // Ensure the backend crate is linked so its registrar lands in the slice.
    let _ = CpuPortableBackend::new();

    // List backends
    let backends = list_backends();
    println!("Available backends: {:?}", backends);

    // CPU backend should always be available
    assert!(has_backend("cpu"), "cpu backend not registered");
    assert!(backends.contains(&"cpu".to_string()));

    // Create CPU backend (registered as "cpu" for convenience)
    let cpu_backend = create_backend("cpu").expect("failed to create cpu backend");
    // Backend's self-reported name is "cpu-portable", but it's registered as "cpu"
    assert_eq!(cpu_backend.backend_name(), "cpu-portable");

    // Non-existent backend should return None
    assert!(!has_backend("nonexistent"));
    assert!(create_backend("nonexistent").is_none());
}
