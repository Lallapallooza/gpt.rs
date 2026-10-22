//! The C backend's pipeline: the weights of its GEMMs are packed into the GEMM's panels before the
//! load-time passes, whose `outline-static` moves that packing to load time.

mod pack_weights;

use gpt_rs_ir::dialects::tensor::MergeContractedAxes;
use gpt_rs_ir::dialects::tile::Numerics;
use gpt_rs_ir::passes::{self, Pipeline};

pub use self::pack_weights::PackWeights;

pub(crate) fn pipeline() -> Pipeline {
    let mut pm = Pipeline::new();
    passes::populate_simplification_passes(&mut pm);
    let numerics = Numerics {
        scale_fold: true,
        ..Numerics::default()
    };
    passes::populate_tensor_passes(&mut pm, &numerics);
    pm.add(MergeContractedAxes).add(PackWeights);
    passes::populate_load_time_passes(&mut pm);
    pm
}
