gpt_rs_backend_tests::define_backend_tests!(x86_backend_tests, || {
    let target = gpt_rs_ir::targets::Target::resolve("host").expect("the host target");
    let options = gpt_rs_backend_x86::X86Options::default();
    std::sync::Arc::new(
        gpt_rs_backend_x86::X86Backend::new(target, options).expect("an x86 backend for the host"),
    )
});
