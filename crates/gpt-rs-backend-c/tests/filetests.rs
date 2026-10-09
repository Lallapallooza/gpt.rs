//! Runs the filetests of the C backend's passes, under `filetests/backend/c/`.

#[test]
fn filetests() {
    let backend = gpt_rs_backend_c::CBackend::new();
    gpt_rs_filetests::backend::run(&backend);
}
