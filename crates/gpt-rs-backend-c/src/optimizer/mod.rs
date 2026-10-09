//! The C backend's pipeline: the weights of its GEMMs are packed into the GEMM's panels before the
//! load-time passes, whose `outline-static` moves that packing to load time.

mod pack;

use anyhow::Result;
use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::{self, pass, Greedy, Pass, Pipeline, Rewrites};
use gpt_rs_ir::rewrite::PatternSet;

/// Packs the weights of the dots that codegen sends to the GEMM into the GEMM's panel layout.
///
/// The packing is a reshape and a transpose of the weight, and the dot then contracts the packed
/// weight, so the IR keeps its meaning. When the weight is static, `outline-static` moves the
/// packing to load time, and the GEMM reads the packed parameter. Other weights, activations
/// among them, stay unpacked, and the GEMM packs the panels it needs on each call.
#[pass(
    name = "c-pack-weights",
    stats(packed = "GEMM weights packed into panels")
)]
pub struct PackWeights;

/// Adds the pattern that packs GEMM weights into panels.
pub fn populate_weight_layout_patterns(set: &mut PatternSet) {
    set.add(pack::PackGemmWeight);
}

impl Pass for PackWeights {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_weight_layout_patterns(&mut patterns);
        Rewrites {
            folds: false,
            patterns,
        }
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        let report = Greedy::new(self.rewrites()).erase_dead().run(module)?;
        PackWeights::PACKED.add(report.rewritten);
        Ok(report.changed())
    }
}

pub(crate) fn pipeline() -> Pipeline {
    let mut pm = Pipeline::new();
    passes::populate_simplification_passes(&mut pm);
    passes::populate_tensor_passes(&mut pm);
    pm.add(PackWeights);
    passes::populate_load_time_passes(&mut pm);
    pm
}
