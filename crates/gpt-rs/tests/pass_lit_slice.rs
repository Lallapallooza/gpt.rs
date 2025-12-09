mod lit_support;

use gpt_rs::backend::passes::SliceCanonicalizationPass;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use lit_support::{load_cases, run_case_with_passes};

#[test]
fn slice_cases() {
    let backend = CpuPortableBackend::new();
    let cases = load_cases("slice.lit");
    let pass = SliceCanonicalizationPass::default();
    for case in cases {
        run_case_with_passes(&backend, &[&pass], &case);
    }
}
