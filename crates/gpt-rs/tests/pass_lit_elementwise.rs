mod lit_support;

use gpt_rs::backend::passes::ElementwiseSimplificationPass;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use lit_support::{load_cases, run_case_with_passes};

#[test]
fn elementwise_cases() {
    let backend = CpuPortableBackend::new();
    let cases = load_cases("elementwise.lit");
    let pass = ElementwiseSimplificationPass::default();
    for case in cases {
        run_case_with_passes(&backend, &[&pass], &case);
    }
}
