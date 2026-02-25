use std::sync::Arc;

use gpt_rs::backend::passes::DeadCodeEliminationPass;
use gpt_rs::backend::passes::FusionHintPass;
use gpt_rs::backend::pipeline::{BackendPipeline, PipelineBuilder};

use super::{
    TritonHintCostModel, TritonHintLegalizer, TritonLayerNormFusionPass, TritonSoftmaxFusionPass,
};

#[derive(Debug, Default)]
pub struct TritonPipeline;

impl BackendPipeline<crate::TritonBackend> for TritonPipeline {
    fn populate_pre(&self, p: &mut PipelineBuilder<crate::TritonBackend>) {
        // LayerNorm pattern extraction expects explicit gamma/beta broadcast nodes.
        // Run this before ParamOnlyFoldToParam, which can fold those broadcasts
        // into derived parameters and erase the structural pattern.
        p.pass(Arc::new(TritonLayerNormFusionPass));
    }

    fn populate_legalize(&self, _p: &mut PipelineBuilder<crate::TritonBackend>) {
        // Reserved for layout/dtype normalization passes.
    }

    fn populate_fuse(&self, p: &mut PipelineBuilder<crate::TritonBackend>) {
        // Softmax fusion is safe after param-only folding and before post-canonicalization.
        p.pass(Arc::new(TritonSoftmaxFusionPass));
    }

    fn populate_cleanup(&self, p: &mut PipelineBuilder<crate::TritonBackend>) {
        let pass =
            FusionHintPass::new(Arc::new(TritonHintLegalizer), Arc::new(TritonHintCostModel))
                .with_min_score(0);
        p.pass(Arc::new(pass));
        p.pass(Arc::new(DeadCodeEliminationPass));
    }
}
