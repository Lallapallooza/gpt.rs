use gpt_rs::backend::optimizer::{FunctionPass, OptimizeContext, PassResult};
use gpt_rs::backend::spec::{Function, Operation};

/// Skeleton fusion pass used to establish Triton-specific rewrite hooks.
///
/// This pass is intentionally conservative in the bootstrap stage: it detects candidate
/// elementwise chains but does not rewrite IR yet. Follow-up work will replace candidates with
/// Triton custom calls and serialized launch metadata.
#[derive(Debug, Default)]
pub struct TritonElementwiseFusionPass;

impl FunctionPass<crate::TritonBackend> for TritonElementwiseFusionPass {
    fn name(&self) -> &'static str {
        "triton_elementwise_fusion"
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::TritonBackend>,
    ) -> PassResult {
        let candidates = function
            .body
            .iter()
            .filter(|inst| {
                matches!(
                    inst.op,
                    Operation::ElementwiseBinary(_) | Operation::ElementwiseUnary(_)
                )
            })
            .count();

        PassResult {
            changed: false,
            iterations: 1,
            rewrites_applied: candidates,
            erased_insts: 0,
        }
    }
}
