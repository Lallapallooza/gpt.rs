//! Embedding layer (PyTorch `nn.Embedding`) built on [`functional::embedding_lookup`].

use crate::nn::{self, Params};
use crate::ops::functional;
use crate::tensor::DType;
use anyhow::Result;

/// Embedding layer that maps integer token IDs to dense vectors.
#[nn::module]
pub struct Embedding {
    pub weight: Tensor,
}

#[nn::module]
impl Embedding {
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Ok(Self { weight: p.param()? })
    }

    /// Looks up the rows of `indices` (`I32`) and returns them in f32. The layer widens
    /// reduced-precision tables after the lookup.
    fn forward(&self, indices: &Tensor) -> Tensor {
        functional::embedding_lookup(&self.weight, indices).cast(DType::F32)
    }
}
