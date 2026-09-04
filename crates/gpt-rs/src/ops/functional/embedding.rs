//! Embedding lookup lowered to a portable `take`.

use anyhow::Result;

use crate::{capture, ensure_dtype, ensure_rank, ensure_same_backend, functional};

/// Gathers the rows of `weight` (`[vocab, embed_dim]`) at `indices` (`i32 [seq]`), like
/// `torch.nn.Embedding`.
#[functional]
pub fn embedding_lookup(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    ensure_rank!(weight, 2);
    ensure_rank!(indices, 1);
    ensure_dtype!(indices, I32);
    ensure_same_backend!(weight, indices);
    capture!(|weight, indices| weight.take(&indices))
}
