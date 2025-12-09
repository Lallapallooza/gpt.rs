mod lit_support;

use gpt_rs::backend::passes::{DeadCodeEliminationPass, ParamOnlyFoldToParamPass};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use lit_support::{load_cases, run_case_with_passes};

#[test]
fn param_only_fold_to_param_cases() {
    let backend = CpuPortableBackend::new();
    let cases = load_cases("param_only_fold_to_param.lit");
    let hoist = ParamOnlyFoldToParamPass;
    let dce = DeadCodeEliminationPass;
    for case in cases {
        run_case_with_passes(&backend, &[&hoist, &dce], &case);
    }
}
