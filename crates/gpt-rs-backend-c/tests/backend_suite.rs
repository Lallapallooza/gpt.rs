gpt_rs_backend_tests::define_backend_tests!(c_backend_tests, || std::sync::Arc::new(
    gpt_rs_backend_c::CBackend::new()
));
