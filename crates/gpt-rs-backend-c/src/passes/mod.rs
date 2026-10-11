//! The C backend's pipeline: the weights of its GEMMs are packed into the GEMM's panels before the
//! load-time passes, whose `outline-static` moves that packing to load time.

mod pack_weights;

use gpt_rs_ir::passes::{self, Pipeline};

pub use self::pack_weights::PackWeights;

pub(crate) fn pipeline() -> Pipeline {
    let mut pm = Pipeline::new();
    passes::populate_simplification_passes(&mut pm);
    passes::populate_tensor_passes(&mut pm);
    pm.add(PackWeights);
    passes::populate_load_time_passes(&mut pm);
    pm
}
