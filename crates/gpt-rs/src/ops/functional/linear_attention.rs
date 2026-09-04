//! Gated DeltaNet (linear attention) primitives.
//!
//! Two functionals cover the token-mixing path of Gated DeltaNet layers:
//! - [`causal_conv1d`]: a depthwise causal 1-D convolution with a carried input window.
//! - [`gated_delta_rule`]: the gated delta-rule recurrence over a chunk of tokens, with a carried
//!   state matrix per head.
//!
//! Both functionals take and return their carried state explicitly. So the same entry points serve
//! prompt prefill (many tokens) and decode (one token).

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::attention::MASKED_BIAS;
use crate::ops::ptir::{self, DotAttrs, DotDims};
use crate::tensor::{DType, DeviceTensor, Shape};
use crate::{
    capture, ensure_dtype, ensure_rank, ensure_same_backend, ensure_same_shape, functional,
};

/// Chunk length of the WY-form evaluation in [`gated_delta_rule`].
const CHUNK: usize = 64;

/// Epsilon of the L2 normalisation of queries and keys. It matches the FLA reference.
const L2NORM_EPS: f32 = 1e-6;

/// Carried state of one Gated DeltaNet layer between decode steps.
pub struct LinearAttentionCache<B: PortableBackend + 'static> {
    conv_state: DeviceTensor<B>,
    recurrent_state: DeviceTensor<B>,
}

impl<B: PortableBackend + 'static> Clone for LinearAttentionCache<B> {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
        }
    }
}

impl<B: PortableBackend + 'static> LinearAttentionCache<B> {
    /// `conv_state` is `[kernel - 1, channels]`, with the most recent inputs last.
    /// `recurrent_state` is `[value_heads, key_dim, value_dim]`.
    pub fn new(conv_state: DeviceTensor<B>, recurrent_state: DeviceTensor<B>) -> Result<Self> {
        ensure!(
            conv_state.shape().rank() == 2 && conv_state.dtype() == DType::F32,
            "linear attention conv state must be f32 of rank 2, got {:?} {:?}",
            conv_state.dtype(),
            conv_state.shape().dims()
        );
        ensure!(
            recurrent_state.shape().rank() == 3 && recurrent_state.dtype() == DType::F32,
            "linear attention recurrent state must be f32 of rank 3, got {:?} {:?}",
            recurrent_state.dtype(),
            recurrent_state.shape().dims()
        );
        Ok(Self {
            conv_state,
            recurrent_state,
        })
    }

    /// Creates the zero state of a `kernel_size`-tap convolution over `channels`, and of
    /// `num_value_heads` recurrent matrices of shape `[key_head_dim, value_head_dim]`.
    pub fn zeros(
        backend: &Arc<B>,
        kernel_size: usize,
        channels: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Self> {
        ensure!(kernel_size >= 2, "convolution kernel size must be >= 2");
        let conv =
            DeviceTensor::zeros(Arc::clone(backend), Shape::new([kernel_size - 1, channels]))?;
        let recurrent = DeviceTensor::zeros(
            Arc::clone(backend),
            Shape::new([num_value_heads, key_head_dim, value_head_dim]),
        )?;
        Self::new(conv, recurrent)
    }

    pub fn conv_state(&self) -> &DeviceTensor<B> {
        &self.conv_state
    }

    pub fn recurrent_state(&self) -> &DeviceTensor<B> {
        &self.recurrent_state
    }
}

/// Output of [`causal_conv1d`].
pub struct CausalConv1dResult<B: PortableBackend + 'static> {
    /// `conv(x)`, shape `[T, C]`.
    pub output: DeviceTensor<B>,
    /// Last `kernel - 1` inputs of `[state; x]`, shape `[kernel - 1, C]`.
    pub state: DeviceTensor<B>,
}

/// Depthwise causal convolution with an explicit input window carried across calls.
///
/// With `window = [state; x]` of shape `[T + K - 1, C]`, output row `t` is
/// `sum_j weight[:, j] * window[t + j]`. When `state` is zero, this is PyTorch `conv1d` with
/// `groups = C` and `K - 1` leading zeros. The returned state holds the last `K - 1` window rows.
#[functional]
pub fn causal_conv1d(x: &Tensor, state: &Tensor, weight: &Tensor) -> Result<CausalConv1dResult<B>> {
    ensure_same_backend!(x, state, weight);
    ensure_rank!(x, 2);
    ensure_rank!(state, 2);
    ensure_rank!(weight, 2);
    ensure_dtype!(x, F32);
    ensure_dtype!(state, F32);
    ensure_dtype!(weight, F32);
    let [seq_len, channels] = [x.shape().dims()[0], x.shape().dims()[1]];
    let [w_channels, kernel] = [weight.shape().dims()[0], weight.shape().dims()[1]];
    ensure!(
        w_channels == channels,
        "{FUNCTIONAL}: weight must have the {channels} channels of x, got {w_channels}"
    );
    ensure!(
        kernel >= 2,
        "{FUNCTIONAL}: kernel must be >= 2, got {kernel}"
    );
    ensure!(
        state.shape().dims() == [kernel - 1, channels],
        "{FUNCTIONAL}: state must be [{}, {channels}], got {:?}",
        kernel - 1,
        state.shape().dims()
    );

    let (output, state) = capture!(session, |x, state, weight| {
        let window = ptir::Tensor::concat(0, &[state, x]);
        let taps = weight.transpose(vec![1, 0]);
        let mut acc = None;
        for j in 0..kernel {
            let rows = window.slice(vec![j, 0], vec![seq_len, channels]);
            let tap = taps
                .slice(vec![j, 0], vec![1, channels])
                .broadcast_to(vec![seq_len, channels]);
            let term = rows * tap;
            acc = Some(match acc {
                None => term,
                Some(prev) => prev + term,
            });
        }
        let output = acc.expect("kernel >= 2");
        let new_state = window.slice(vec![seq_len, 0], vec![kernel - 1, channels]);
        session.export(new_state);
        (output, new_state)
    })?;
    Ok(CausalConv1dResult { output, state })
}

/// Output of [`gated_delta_rule`].
pub struct GatedDeltaRuleResult<B: PortableBackend + 'static> {
    /// Per-token outputs, shape `[T, Hv, Dv]`.
    pub output: DeviceTensor<B>,
    /// Final recurrent state, shape `[Hv, Dk, Dv]`.
    pub state: DeviceTensor<B>,
}

#[derive(Clone, Copy)]
struct DeltaRulePlan {
    seq_len: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
}

/// Gated delta-rule recurrence (Gated DeltaNet) over `T` tokens with carried state.
///
/// The functional L2-normalises `q` and `k` over the key dimension and scales `q` by `Dk^-1/2`. Value head `h` reads key head `h / (Hv / Hk)`. Each head has the
/// state `S` (`[Dk, Dv]`), with `S_0 = state`. Token `t` computes the following, like Hugging Face
/// `torch_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel`:
///
/// ```text
/// S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - exp(g_t) S_{t-1}^T k_t))^T
/// o_t = S_t^T q_t
/// ```
///
/// The implementation evaluates this recurrence in fixed-length chunks in the WY form, as the FLA
/// chunked kernels do. So the captured graph uses only dense core PTIR ops. Its size grows with the
/// number of chunks, not with the number of tokens.
#[functional]
pub fn gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<GatedDeltaRuleResult<B>> {
    ensure_same_backend!(q, k, v, g, beta, state);
    ensure_dtype!(q, F32);
    ensure_dtype!(k, F32);
    ensure_dtype!(v, F32);
    ensure_dtype!(g, F32);
    ensure_dtype!(beta, F32);
    ensure_dtype!(state, F32);
    ensure_rank!(q, 3);
    ensure_rank!(v, 3);
    ensure_same_shape!(q, k);
    let (q_dims, v_dims) = (q.shape().dims(), v.shape().dims());
    let (seq_len, key_heads, key_dim) = (q_dims[0], q_dims[1], q_dims[2]);
    let (value_heads, value_dim) = (v_dims[1], v_dims[2]);
    ensure!(
        v_dims[0] == seq_len,
        "{FUNCTIONAL}: v must have the {seq_len} tokens of q, got {}",
        v_dims[0]
    );
    ensure!(
        value_heads.is_multiple_of(key_heads),
        "{FUNCTIONAL}: value heads ({value_heads}) must be a multiple of key heads ({key_heads})"
    );
    for (name, t) in [("g", g), ("beta", beta)] {
        ensure!(
            t.shape().dims() == [seq_len, value_heads],
            "{FUNCTIONAL}: {name} must be [{seq_len}, {value_heads}], got {:?}",
            t.shape().dims()
        );
    }
    ensure!(
        state.shape().dims() == [value_heads, key_dim, value_dim],
        "{FUNCTIONAL}: state must be [{value_heads}, {key_dim}, {value_dim}], got {:?}",
        state.shape().dims()
    );
    let plan = DeltaRulePlan {
        seq_len,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    };

    let (output, state) = capture!(session, |q, k, v, g, beta, state| {
        // Functionals do not name the PTIR session type, so pass the mask builder as a closure.
        let tril =
            |n, diagonal, inside, outside| ptir::tril_mask(&session, n, diagonal, inside, outside);
        let (heads_first, state) = gated_delta_rule_chunked(tril, plan, [q, k, v, g, beta, state]);
        let output = heads_first.transpose(vec![1, 0, 2]);
        session.export(state);
        (output, state)
    })?;
    Ok(GatedDeltaRuleResult { output, state })
}

fn l2_normalize_last<'ctx, 'gb, B: PortableBackend + 'static>(
    x: ptir::Tensor<'ctx, 'gb, B>,
    dims: &[usize],
) -> ptir::Tensor<'ctx, 'gb, B> {
    let last = dims.len() - 1;
    let sum_sq = (x * x).reduce_sum(vec![last], true);
    let inv = sum_sq.add_scalar(L2NORM_EPS).rsqrt();
    x * inv.broadcast_to(dims.to_vec())
}

fn batched_matmul<'ctx, 'gb, B: PortableBackend + 'static>(
    lhs: &ptir::Tensor<'ctx, 'gb, B>,
    rhs: &ptir::Tensor<'ctx, 'gb, B>,
    contract_lhs: usize,
    contract_rhs: usize,
) -> ptir::Tensor<'ctx, 'gb, B> {
    lhs.dot_general(
        rhs,
        &DotDims::new(
            crate::axes!(0),
            ptir::axes_iter([contract_lhs]),
            ptir::axes_iter([contract_rhs]),
        ),
        &DotAttrs::default(),
    )
}

/// Chunked (WY-form) gated delta rule built from core PTIR ops.
///
/// Take a chunk of length `L` with cumulative log-decays `G`. The intra-chunk system is
/// `(I + A) X = B`, with the strictly lower-triangular `A_ij = beta_i (k_i . k_j) exp(G_i - G_j)`.
/// The function solves it with `(I + A)^-1 = prod_m (I + (-A)^(2^m))`. This is exact because `A`
/// is nilpotent. Returns the outputs as `[Hv, T, Dv]` and the final state.
fn gated_delta_rule_chunked<'ctx, 'gb, B: PortableBackend + 'static>(
    tril: impl Fn(usize, i64, f32, f32) -> ptir::Tensor<'ctx, 'gb, B>,
    plan: DeltaRulePlan,
    [q, k, v, g, beta, state]: [ptir::Tensor<'ctx, 'gb, B>; 6],
) -> (ptir::Tensor<'ctx, 'gb, B>, ptir::Tensor<'ctx, 'gb, B>) {
    let DeltaRulePlan {
        seq_len: t,
        key_heads: hk,
        value_heads: hv,
        key_dim: dk,
        value_dim: dv,
    } = plan;
    let group = hv / hk;
    let scale = 1.0f32 / (dk as f32).sqrt();

    let qk_dims = [t, hk, dk];
    let qn = l2_normalize_last(q, &qk_dims) * scale;
    let kn = l2_normalize_last(k, &qk_dims);
    let expand = |x: ptir::Tensor<'ctx, 'gb, B>| {
        x.reshape(vec![t, hk, 1, dk])
            .broadcast_to(vec![t, hk, group, dk])
            .reshape(vec![t, hv, dk])
            .transpose(vec![1, 0, 2])
    };
    let q_all = expand(qn);
    let k_all = expand(kn);
    let v_all = v.transpose(vec![1, 0, 2]);
    let g_all = g.transpose(vec![1, 0]);
    let beta_all = beta.transpose(vec![1, 0]);

    let mut s = state;
    let mut outputs = Vec::new();
    let mut start = 0;
    while start < t {
        let l = CHUNK.min(t - start);
        let qc = q_all.slice(vec![0, start, 0], vec![hv, l, dk]);
        let kc = k_all.slice(vec![0, start, 0], vec![hv, l, dk]);
        let vc = v_all.slice(vec![0, start, 0], vec![hv, l, dv]);
        let gc = g_all.slice(vec![0, start], vec![hv, l]);
        let bc = beta_all.slice(vec![0, start], vec![hv, l]);

        // Inclusive cumulative sum over the chunk: G = g @ U with U[j, i] = [i >= j].
        let lower = tril(l, 0, 1.0, 0.0);
        let upper = lower.transpose(vec![1, 0]);
        let cum = gc.dot_general(
            &upper,
            &DotDims::new(ptir::axes_iter([]), crate::axes!(1), crate::axes!(0)),
            &DotAttrs::default(),
        );

        // The bias is 0 on and below the diagonal and a large negative value above it. `exp` maps
        // the masked entries to exactly 0, and their positive differences never overflow.
        let causal_bias = tril(l, 0, 0.0, MASKED_BIAS)
            .reshape(vec![1, l, l])
            .broadcast_to(vec![hv, l, l]);
        let strict = tril(l, -1, 1.0, 0.0)
            .reshape(vec![1, l, l])
            .broadcast_to(vec![hv, l, l]);
        let diff = cum.reshape(vec![hv, l, 1]).broadcast_to(vec![hv, l, l])
            - cum.reshape(vec![hv, 1, l]).broadcast_to(vec![hv, l, l]);
        // exp(G_i - G_j) on and below the diagonal, 0 above.
        let decay = (diff + causal_bias).exp();

        let beta_col = bc.reshape(vec![hv, l, 1]);
        let kk = batched_matmul(&kc, &kc, 2, 2);
        let a = kk * decay * beta_col.broadcast_to(vec![hv, l, l]) * strict;
        let neg_a = a * -1.0f32;
        let eye = (lower * upper)
            .reshape(vec![1, l, l])
            .broadcast_to(vec![hv, l, l]);
        let mut inv = eye + neg_a;
        let mut power = neg_a;
        let mut covered = 2usize;
        while covered < l {
            power = batched_matmul(&power, &power, 2, 1);
            inv = batched_matmul(&inv, &(eye + power), 2, 1);
            covered *= 2;
        }

        let exp_cum = cum.exp().reshape(vec![hv, l, 1]);
        let v_beta = vc * beta_col.broadcast_to(vec![hv, l, dv]);
        let k_beta_decay = kc * (beta_col * exp_cum).broadcast_to(vec![hv, l, dk]);
        let u = batched_matmul(&inv, &v_beta, 2, 1);
        let w = batched_matmul(&inv, &k_beta_decay, 2, 1);
        let v_new = u - batched_matmul(&w, &s, 2, 1);
        let q_decay = qc * exp_cum.broadcast_to(vec![hv, l, dk]);
        let o_inter = batched_matmul(&q_decay, &s, 2, 1);
        let qk = batched_matmul(&qc, &kc, 2, 2) * decay;
        outputs.push(o_inter + batched_matmul(&qk, &v_new, 2, 1));

        let last = cum.slice(vec![0, l - 1], vec![hv, 1]);
        let tail_decay = (last.broadcast_to(vec![hv, l]) - cum)
            .exp()
            .reshape(vec![hv, l, 1]);
        let k_tail = kc * tail_decay.broadcast_to(vec![hv, l, dk]);
        let carry = last
            .exp()
            .reshape(vec![hv, 1, 1])
            .broadcast_to(vec![hv, dk, dv]);
        s = s * carry + batched_matmul(&k_tail, &v_new, 1, 1);
        start += l;
    }

    (ptir::Tensor::concat(1, &outputs), s)
}
