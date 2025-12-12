gpt_rs_backend_tests::define_backend_tests!(ref_cpu_backend_tests, || std::sync::Arc::new(
    gpt_rs_backend_ref_cpu::CpuPortableBackend::new()
));
