mod lit_support;

use gpt_rs::backend::passes::CommonSubexpressionEliminationPass;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use lit_support::{load_cases, run_case_with_passes};

#[test]
fn dedup_cases() {
    let backend = CpuPortableBackend::new();
    let cases = load_cases("dedup.lit");
    let pass = CommonSubexpressionEliminationPass;
    for case in cases {
        run_case_with_passes(&backend, &[&pass], &case);
    }
}
