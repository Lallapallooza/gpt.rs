mod lit_support;

use gpt_rs::backend::passes::DotScaleFoldPass;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use lit_support::{load_cases, run_case_with_passes};

#[test]
fn dot_scale_fold_cases() {
    let backend = CpuPortableBackend::new();
    let pass = DotScaleFoldPass::default();
    for case in load_cases("dot_scale_fold.lit") {
        run_case_with_passes(&backend, &[&pass], &case);
    }
}
