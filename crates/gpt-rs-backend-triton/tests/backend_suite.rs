gpt_rs_backend_tests::define_backend_tests!(triton_backend_tests, || std::sync::Arc::new(
    gpt_rs_backend_triton::TritonBackend::new()
));
