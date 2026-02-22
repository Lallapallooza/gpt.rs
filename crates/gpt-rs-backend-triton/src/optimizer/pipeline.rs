use std::sync::Arc;

use gpt_rs::backend::passes::ElementwiseDagFusionPass;
use gpt_rs::backend::pipeline::{BackendPipeline, PipelineBuilder};

use crate::targets::TARGET_ELEMENTWISE_FUSED_F32_V1;

#[derive(Debug, Default)]
pub struct TritonPipeline;

impl BackendPipeline<crate::TritonBackend> for TritonPipeline {
    fn populate_legalize(&self, _p: &mut PipelineBuilder<crate::TritonBackend>) {
        // Reserved for layout/dtype normalization passes.
    }

    fn populate_fuse(&self, p: &mut PipelineBuilder<crate::TritonBackend>) {
        p.pass(Arc::new(ElementwiseDagFusionPass::new(
            TARGET_ELEMENTWISE_FUSED_F32_V1,
        )));
    }

    fn populate_cleanup(&self, _p: &mut PipelineBuilder<crate::TritonBackend>) {
        // Reserved for post-fusion canonicalization passes.
    }
}
