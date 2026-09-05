//! Embedding layer (PyTorch `nn.Embedding`) built on [`functional::embedding_lookup`].

use crate::nn;
use crate::ops::functional;
use crate::tensor::DType;
use anyhow::{ensure, Result};

/// Embedding layer that maps integer token IDs to dense vectors.
#[nn::module]
pub struct Embedding {
    pub weight: Tensor,
}

#[nn::module]
impl Embedding {
    /// Looks up the rows of `indices` (`I32`) and returns them in f32. The layer widens
    /// reduced-precision tables after the lookup.
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        ensure!(
            indices.dtype() == DType::I32,
            "embedding indices must have dtype I32, got {:?}",
            indices.dtype()
        );
        let rows = functional::embedding_lookup(&self.weight, indices)?;
        functional::cast(&rows, DType::F32)
    }
}
