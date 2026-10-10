//! The C backend's pipeline: the weights of its GEMMs are packed into the GEMM's panels before the
//! load-time passes, whose `outline-static` moves that packing to load time.

mod pack;

use anyhow::Result;
use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::{self, pass, Greedy, Pass, Pipeline, Rewrites};
use gpt_rs_ir::rewrite::PatternSet;

/// Puts the weights of the dots that C kernels compute in the layout each kernel reads: GEMM
/// weights in the GEMM's panels, and grouped convolution weights tap by tap.
///
/// Each layout is a transpose of the weight, after a reshape for the GEMM's panels. The dot then
/// contracts the weight in that layout, so the IR keeps its meaning. When the weight is static,
/// `outline-static` moves the layout to load time, and the kernel reads the laid-out parameter.
/// GEMM weights that are not static, activations among them, stay as they are, and the GEMM packs
/// the panels it needs on each call.
#[pass(
    name = "c-pack-weights",
    stats(packed = "weights laid out for a kernel")
)]
pub struct PackWeights;

/// Adds the patterns that lay out weights for the C kernels.
pub fn populate_weight_layout_patterns(set: &mut PatternSet) {
    set.add(pack::PackGemmWeight).add(pack::GroupedConvWeight);
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
