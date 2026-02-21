use std::sync::Arc;

use gpt_rs_backend_triton::TritonBackend;

fn backend_or_skip() -> Option<Arc<TritonBackend>> {
    if TritonBackend::is_available() {
        Some(Arc::new(TritonBackend::new()))
    } else {
        eprintln!("skipping triton backend test: CUDA runtime unavailable");
        None
    }
}

#[test]
fn smoke_matmul_matches_expected() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    gpt_rs_backend_tests::smoke::matmul_matches_expected(&backend);
}

#[cfg(feature = "torch")]
#[test]
fn torch_parity_add_matches_torch_shape_2x3x4() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    gpt_rs_backend_tests::torch_parity::harness::run_parity_test_with_modes(
        Arc::clone(&backend),
        "triton_torch_add_matches_torch_shape_2x3x4",
        |backend| {
            gpt_rs_backend_tests::torch_parity::arithmetic::add_matches_torch_shape_2x3x4(backend);
        },
    );
}

#[cfg(feature = "torch")]
#[test]
fn torch_parity_matmul_matches_torch_1x5_5x3() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    gpt_rs_backend_tests::torch_parity::harness::run_parity_test_with_modes(
        Arc::clone(&backend),
        "triton_torch_matmul_matches_torch_1x5_5x3",
        |backend| {
            gpt_rs_backend_tests::torch_parity::matmul::matmul_matches_torch_1x5_5x3(backend);
        },
    );
}
