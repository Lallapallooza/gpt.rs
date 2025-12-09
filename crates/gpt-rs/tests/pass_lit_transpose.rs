mod lit_support;

use gpt_rs::backend::passes::TransposeCanonicalizationPass;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use lit_support::{load_cases, run_case_with_passes};

#[test]
fn transpose_cases() {
    let backend = CpuPortableBackend::new();
    let cases = load_cases("transpose.lit");
    let pass = TransposeCanonicalizationPass::default();
    for case in cases {
        run_case_with_passes(&backend, &[&pass], &case);
    }
}
