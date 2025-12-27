//! Embedding lookup helpers that lower to backend take operations.
//!
//! The routines validate tensor metadata and emit reshape/gather
//! sequences that work across supported backends.

use std::sync::Arc;

use anyhow::{bail, Result};
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::common::{
    ensure_dtype_equals, ensure_last_dim, ensure_rank, ensure_same_backend, CaptureIntoDeviceTensor,
};
use crate::ops::graph::GraphArena;
use crate::tensor::{DType as TensorDType, DeviceTensor};

struct EmbeddingPlan {
    seq_len: usize,
    squeeze_indices: bool,
    requires_grad: bool,
}

/// Validates embedding tensors and records gather metadata/broadcasted shapes.
///
/// Ensures the table is `[vocab, embed_dim]` and indices are rank-1 `[seq]` int32 tensors.
/// Tests live in the backend torch_parity suite.
fn validate_embedding_lookup<B: PortableBackend + 'static>(
    weight: &DeviceTensor<B>,
    indices: &DeviceTensor<B>,
) -> Result<EmbeddingPlan> {
    ensure_rank("embedding weight", weight, 2)?;
    ensure_dtype_equals("embedding indices", indices, TensorDType::I32)?;

    // Accept both rank-1 [seq_len] and rank-2 [seq_len, 1] for backward compatibility.
    let rank = indices.shape().rank();
    let (seq_len, squeeze_indices) = if rank == 1 {
        (indices.shape().dims()[0], false)
    } else if rank == 2 {
        ensure_last_dim("embedding indices", indices, 1)?;
        (indices.shape().dims()[0], true)
    } else {
        bail!(
            "embedding indices must be rank 1 [seq_len] or rank 2 [seq_len, 1], got rank {}",
            rank
        );
    };

    ensure_same_backend("embedding_lookup", weight, indices)?;

    Ok(EmbeddingPlan {
        seq_len,
        squeeze_indices,
        requires_grad: weight.requires_grad_flag() || indices.requires_grad_flag(),
    })
}

/// Fetches embedding vectors for integer indices by emitting a portable gather program.
/// The helper mirrors the behaviour of `torch.nn.Embedding` while remaining backend agnostic.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.embedding_lookup")]
pub fn embedding_lookup<B: PortableBackend + 'static>(
    _backend: &B,
    weight: &DeviceTensor<B>,
    indices: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_embedding_lookup(weight, indices)?;
    let captured = capture_ptir! {
        { weight, indices },
        |_session| {
            let indices_1d = if plan.squeeze_indices {
                indices.reshape(vec![plan.seq_len])
            } else {
                indices
            };
            let gathered = weight.take(&indices_1d);
            Ok(gathered.id())
        }
    }?;

    captured.into_device_tensor(plan.requires_grad)
}

/// Fetches embedding vectors with explicit graph arena control for optimization.
///
/// This variant allows reusing an existing graph arena, which is useful when you want
/// multiple embedding lookups to share the same computation graph for better optimization.
///
/// # Example
/// ```rust,ignore
/// let token_emb = embedding_lookup(backend, token_weights, token_indices)?;
/// // Reuse token_emb's graph for position embeddings
/// let pos_emb = embedding_lookup_with_graph(
///     backend, pos_weights, pos_indices, token_emb.graph()
/// )?;
/// ```
#[ptir_pattern(target = "gpt_rs.embedding_lookup_with_graph")]
pub fn embedding_lookup_with_graph<B: PortableBackend + 'static>(
    _backend: &B,
    weight: &DeviceTensor<B>,
    indices: &DeviceTensor<B>,
    graph: Option<Arc<GraphArena<B>>>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_embedding_lookup(weight, indices)?;
    let captured = if let Some(graph) = graph {
        capture_ptir! {
            graph = Arc::clone(&graph);
            { weight, indices },
            |_session| {
                let indices_1d = if plan.squeeze_indices {
                    indices.reshape(vec![plan.seq_len])
                } else {
                    indices
                };
                let gathered = weight.take(&indices_1d);
                Ok(gathered.id())
            }
        }?
    } else {
        capture_ptir! {
            { weight, indices },
            |_session| {
                let indices_1d = if plan.squeeze_indices {
                    indices.reshape(vec![plan.seq_len])
                } else {
                    indices
                };
                let gathered = weight.take(&indices_1d);
                Ok(gathered.id())
            }
        }?
    };

    captured.into_device_tensor(plan.requires_grad)
}
