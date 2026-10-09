//! Embedding lookup lowered to a portable `take`.

use crate::{ensure_dtype, ensure_rank, functional};

/// Gathers the rows of `weight` (`[vocab, embed_dim]`) at `indices` (`i32 [seq]`), like
/// `torch.nn.Embedding`.
#[functional]
pub fn embedding_lookup(weight: &Tensor, indices: &Tensor) -> Tensor {
    ensure_rank!(weight, 2);
    ensure_rank!(indices, 1);
    ensure_dtype!(indices, Si32);
    weight.take(indices)
}
