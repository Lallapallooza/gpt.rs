mod fusion_policy;
mod pipeline;

use gpt_rs::backend::conversion::ConversionResult;
use gpt_rs::backend::optimizer::{
    default_optimizer, EntryParam, EntrySignature, OptimizeConfig, OptimizeContext,
    OptimizeServices,
};
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::tensor::InputRole;

pub use fusion_policy::{TritonHintCostModel, TritonHintLegalizer};
pub use pipeline::TritonPipeline;

pub fn optimize_program_for_triton(
    program: &gpt_rs::backend::spec::Program,
) -> ConversionResult<gpt_rs::backend::spec::Program> {
    let mut optimized = program.clone();
    let backend = crate::TritonBackend::new();
    let optimizer = default_optimizer(backend.pipeline());
    let params = backend.param_resolver();

    for function in optimized.functions.iter_mut() {
        let entry_params = function
            .parameter_ids
            .iter()
            .copied()
            .zip(function.parameters.iter().cloned())
            .map(|(id, ty)| EntryParam {
                id,
                ty,
                role: InputRole::Arg,
                stable_id: None,
            })
            .collect::<Vec<_>>();
        let entry = EntrySignature::new(entry_params);
        let services = OptimizeServices {
            params: params.as_deref(),
        };
        let cfg = OptimizeConfig::default();
        let mut cx = OptimizeContext::new(&backend, services, entry, cfg);
        let _ = optimizer.optimize(function, &mut cx);
    }

    Ok(optimized)
}
