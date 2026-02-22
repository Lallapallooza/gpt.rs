use std::sync::Arc;

use gpt_rs::backend::passes::FusionHintPass;
use gpt_rs::backend::pipeline::{BackendPipeline, PipelineBuilder};

use super::{TritonHintCostModel, TritonHintLegalizer};

#[derive(Debug, Default)]
pub struct TritonPipeline;

impl BackendPipeline<crate::TritonBackend> for TritonPipeline {
    fn populate_legalize(&self, _p: &mut PipelineBuilder<crate::TritonBackend>) {
        // Reserved for layout/dtype normalization passes.
    }

    fn populate_fuse(&self, p: &mut PipelineBuilder<crate::TritonBackend>) {
        let pass =
            FusionHintPass::new(Arc::new(TritonHintLegalizer), Arc::new(TritonHintCostModel))
                .with_min_score(0);
        p.pass(Arc::new(pass));
    }

    fn populate_cleanup(&self, _p: &mut PipelineBuilder<crate::TritonBackend>) {
        // Reserved for post-fusion canonicalization passes.
    }
}
