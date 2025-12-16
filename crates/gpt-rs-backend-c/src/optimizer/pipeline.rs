use std::sync::Arc;

use gpt_rs::backend::pipeline::{BackendPipeline, PipelineBuilder};

use super::conv2d::CConv2dCustomCallFusionPass;
use super::elementwise::CElementwiseFusionPass;

pub struct CPipeline;

impl BackendPipeline<crate::CBackend> for CPipeline {
    fn populate_legalize(&self, p: &mut PipelineBuilder<crate::CBackend>) {
        p.pass(Arc::new(CConv2dCustomCallFusionPass::default()));
    }

    fn populate_fuse(&self, p: &mut PipelineBuilder<crate::CBackend>) {
        p.pass(Arc::new(CElementwiseFusionPass));
    }
}
