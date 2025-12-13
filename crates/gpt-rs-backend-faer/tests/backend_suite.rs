gpt_rs_backend_tests::define_backend_tests!(faer_backend_tests, || {
    std::sync::Arc::new(gpt_rs_backend_faer::FaerCpuBackend::create())
});
