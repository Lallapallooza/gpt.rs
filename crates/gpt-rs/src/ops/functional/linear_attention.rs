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
use gpt_rs_ir::dialects::tensor::DotGeneral;

use crate::backend::PortableBackend;
use crate::ops::functional::matmul;
use crate::tensor::{DType, DeviceTensor, Shape};
use crate::{ensure_dtype, ensure_rank, ensure_same_shape, functional, s};

/// Additive bias of the masked entries of a decay matrix: `exp` maps it to exactly 0.
const MASKED_BIAS: f32 = -1e9;

/// Chunk length of the WY-form evaluation in [`gated_delta_rule`].
const CHUNK: usize = 64;

/// Epsilon of the L2 normalization of queries and keys. It matches the FLA reference.
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
    #[track_caller]
    pub fn new(conv_state: DeviceTensor<B>, recurrent_state: DeviceTensor<B>) -> Self {
        assert!(
            conv_state.shape().rank() == 2 && conv_state.dtype() == DType::F32,
            "linear attention conv state must be f32 of rank 2, got {:?} {:?}",
            conv_state.dtype(),
            conv_state.shape().dims()
        );
        assert!(
            recurrent_state.shape().rank() == 3 && recurrent_state.dtype() == DType::F32,
            "linear attention recurrent state must be f32 of rank 3, got {:?} {:?}",
            recurrent_state.dtype(),
            recurrent_state.shape().dims()
        );
        Self {
            conv_state,
            recurrent_state,
        }
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
        Ok(Self::new(conv, recurrent))
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
pub fn causal_conv1d(x: &Tensor, state: &Tensor, weight: &Tensor) -> CausalConv1dResult<B> {
    ensure_rank!(x, 2);
    ensure_rank!(state, 2);
    ensure_rank!(weight, 2);
    ensure_dtype!(x, F32);
    ensure_dtype!(state, F32);
    ensure_dtype!(weight, F32);
    let [seq_len, channels] = [x.size(0), x.size(1)];
    let [w_channels, kernel] = [weight.size(0), weight.size(1)];
    assert!(
        w_channels == channels,
        "{FUNCTIONAL}: weight must have the {channels} channels of x, got {w_channels}"
    );
    assert!(
        kernel >= 2,
        "{FUNCTIONAL}: kernel must be >= 2, got {kernel}"
    );
    assert!(
        state.shape().dims() == [kernel - 1, channels],
        "{FUNCTIONAL}: state must be [{}, {channels}], got {:?}",
        kernel - 1,
        state.shape().dims()
    );

    let window = DeviceTensor::concat(0, &[state, x]);
    let taps = weight.transpose([1, 0]);
    let output = (0..kernel)
        .map(|j| window.slice(s![j..j + seq_len]) * taps.slice(s![j..=j]))
        .reduce(|sum, term| sum + term)
        .expect("kernel >= 2");
    let state = window.slice(s![seq_len..]);
    state.export();
    CausalConv1dResult { output, state }
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
}

/// Gated delta-rule recurrence (Gated DeltaNet) over `T` tokens with carried state.
///
/// The functional L2-normalizes `q` and `k` over the key dimension and scales `q` by `Dk^-1/2`.
/// Value head `h` reads key head `h / (Hv / Hk)`. Each head has the state `S` (`[Dk, Dv]`), with
/// `S_0 = state`. Token `t` computes the following, like Hugging Face
/// `torch_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel`:
///
/// ```text
/// S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - exp(g_t) S_{t-1}^T k_t))^T
/// o_t = S_t^T q_t
/// ```
///
/// The implementation evaluates this recurrence in fixed-length chunks in the WY form, as the FLA
/// chunked kernels do. So the recorded graph uses only dense core ops. Its size grows with the
/// number of chunks, not with the number of tokens. A chunk of one token, such as a decode step,
/// has no intra-chunk system, so it records the recurrence above for its token.
#[functional]
pub fn gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> GatedDeltaRuleResult<B> {
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
    assert!(
        v_dims[0] == seq_len,
        "{FUNCTIONAL}: v must have the {seq_len} tokens of q, got {}",
        v_dims[0]
    );
    assert!(
        value_heads.is_multiple_of(key_heads),
        "{FUNCTIONAL}: value heads ({value_heads}) must be a multiple of key heads ({key_heads})"
    );
    for (name, t) in [("g", g), ("beta", beta)] {
        assert!(
            t.shape().dims() == [seq_len, value_heads],
            "{FUNCTIONAL}: {name} must be [{seq_len}, {value_heads}], got {:?}",
            t.shape().dims()
        );
    }
    assert!(
        state.shape().dims() == [value_heads, key_dim, value_dim],
        "{FUNCTIONAL}: state must be [{value_heads}, {key_dim}, {value_dim}], got {:?}",
        state.shape().dims()
    );
    let plan = DeltaRulePlan {
        seq_len,
        key_heads,
        value_heads,
        key_dim,
    };

    let (heads_first, state) = gated_delta_rule_chunked(plan, [q, k, v, g, beta, state]);
    state.export();
    GatedDeltaRuleResult {
        output: heads_first.transpose([1, 0, 2]),
        state,
    }
}

/// An `[n, n]` lower-triangular selection in the graph of `like`: `inside` where
/// `col <= row + diagonal`, and `outside` elsewhere. With `inside = 1` and `outside = 0`, this is
/// `torch.tril(torch.ones(n, n), diagonal)`.
fn tril_mask<B: PortableBackend + 'static>(
    like: &DeviceTensor<B>,
    n: usize,
    diagonal: i64,
    inside: f32,
    outside: f32,
) -> DeviceTensor<B> {
    // Row `i` and column `j` of the result read position `(i + r, j + c)` of a larger plane. So
    // `row >= col` in that plane means `i + diagonal >= j`.
    let (r, c) = (diagonal.max(0) as usize, (-diagonal).max(0) as usize);
    let plane = [n + r, n + c];
    let rows = DeviceTensor::iota(like, plane, 0, DType::Si32);
    let cols = DeviceTensor::iota(like, plane, 1, DType::Si32);
    rows.greater_equal(&cols)
        .select(inside, outside)
        .slice(s![r..r + n, c..c + n])
}

/// `x` divided by its L2 norm over the last axis.
fn l2_normalize_last<B: PortableBackend + 'static>(x: &DeviceTensor<B>) -> DeviceTensor<B> {
    x * ((x * x).reduce_sum([-1], true) + L2NORM_EPS).rsqrt()
}

/// The contraction of axis `contract_lhs` of `lhs` with axis `contract_rhs` of `rhs`, batched over
/// axis 0.
fn batched_dot<B: PortableBackend + 'static>(
    lhs: &DeviceTensor<B>,
    rhs: &DeviceTensor<B>,
    contract_lhs: i64,
    contract_rhs: i64,
) -> DeviceTensor<B> {
    lhs.dot_general(
        rhs,
        DotGeneral::contract([contract_lhs], [contract_rhs]).batch([0], [0]),
    )
}

/// `[l, l]` mask of the lower-left `b x b` block inside each `2b x 2b` diagonal block.
///
/// Built at a size that `2b` divides, as `eye(m) (x) [[0, 0], [1, 0]] (x) ones(b, b)`, then cut to
/// `l`. The leading block of a product of lower-triangular matrices is the product of their leading
/// blocks, so the doubling with the cut mask is also exact when `l` is not a power of two.
fn off_diagonal_block_mask<B: PortableBackend + 'static>(
    like: &DeviceTensor<B>,
    l: usize,
    b: usize,
) -> DeviceTensor<B> {
    let m = l.div_ceil(2 * b);
    let n = 2 * b * m;
    let lower = tril_mask(like, m, 0, 1.0, 0.0);
    let eye = &lower * lower.transpose([1, 0]);
    let blocks = eye.reshape([m, 1, 1, m, 1, 1]);
    let lower_left = tril_mask(like, 2, -1, 1.0, 0.0).reshape([1, 2, 1, 1, 2, 1]);
    (blocks * lower_left)
        .broadcast_to([m, 2, b, m, 2, b])
        .reshape([n, n])
        .slice(s![..l, ..l])
}

/// Chunked (WY-form) gated delta rule built from core ops.
///
/// Take a chunk of length `L` with cumulative log-decays `G`. The intra-chunk system is
/// `(I + A) X = B`, with the strictly lower-triangular `A_ij = beta_i (k_i . k_j) exp(G_i - G_j)`.
/// The function inverts `I + A` by doubling the block size: if `P` inverts the `b x b` diagonal
/// blocks, `P - P (A * M_b) P` inverts the `2b x 2b` ones, where `M_b` is
/// [`off_diagonal_block_mask`]. The diagonal blocks of every `P` equal those of the true inverse,
/// and its other entries are zero, so `P` never holds entries larger than the inverse's. Powers of
/// `A` would grow: when keys repeat, their entries reach binomial coefficients that cancel, and
/// f32 loses most of the precision.
/// Returns the outputs as `[Hv, T, Dv]` and the final state.
fn gated_delta_rule_chunked<B: PortableBackend + 'static>(
    plan: DeltaRulePlan,
    [q, k, v, g, beta, state]: [&DeviceTensor<B>; 6],
) -> (DeviceTensor<B>, DeviceTensor<B>) {
    let DeltaRulePlan {
        seq_len: t,
        key_heads: hk,
        value_heads: hv,
        key_dim: dk,
    } = plan;
    let group = hv / hk;
    let scale = 1.0f32 / (dk as f32).sqrt();

    let expand = |x: DeviceTensor<B>| {
        x.reshape([t, hk, 1, dk])
            .broadcast_to([t, hk, group, dk])
            .reshape([t, hv, dk])
            .transpose([1, 0, 2])
    };
    let q_all = expand(l2_normalize_last(q) * scale);
    let k_all = expand(l2_normalize_last(k));
    let v_all = v.transpose([1, 0, 2]);
    let g_all = g.transpose([1, 0]);
    let beta_all = beta.transpose([1, 0]);

    let mut s = state.clone();
    let mut outputs = Vec::new();
    let mut start = 0;
    while start < t {
        let l = CHUNK.min(t - start);
        let qc = q_all.slice(s![.., start..start + l]);
        let kc = k_all.slice(s![.., start..start + l]);
        let vc = v_all.slice(s![.., start..start + l]);
        let gc = g_all.slice(s![.., start..start + l]);
        let bc = beta_all.slice(s![.., start..start + l]);

        let beta_col = bc.reshape([hv, l, 1]);
        let ChunkTerms {
            exp_cum,
            u,
            w,
            scores,
            k_tail,
        } = if l == 1 {
            one_token_terms(&qc, &kc, &vc, &gc, &beta_col)
        } else {
            wy_terms(&qc, &kc, &vc, &gc, &beta_col, l)
        };
        let v_new = u - matmul(&w, &s);
        let o_inter = matmul(&(&qc * &exp_cum), &s);
        outputs.push(o_inter + matmul(&scores, &v_new));

        let carry = exp_cum.slice(s![.., l - 1..l, ..]);
        s = s * carry + batched_dot(&k_tail, &v_new, 1, 1);
        start += l;
    }

    let outputs: Vec<_> = outputs.iter().collect();
    (DeviceTensor::concat(1, &outputs), s)
}

/// What a chunk contributes besides the carried state: the cumulative decays `exp(G)` (`[Hv, L, 1]`),
/// `U` and `W` with `V_new = U - W S`, the decayed scores `(Q K^T) * exp(G_i - G_j)` that weigh
/// `V_new`, and the keys decayed to the chunk's end, which update the state.
struct ChunkTerms<B: PortableBackend + 'static> {
    exp_cum: DeviceTensor<B>,
    u: DeviceTensor<B>,
    w: DeviceTensor<B>,
    scores: DeviceTensor<B>,
    k_tail: DeviceTensor<B>,
}

/// The terms of a chunk of `l` tokens: the intra-chunk system `(I + A) X = B` of
/// [`gated_delta_rule_chunked`], solved by doubling.
fn wy_terms<B: PortableBackend + 'static>(
    qc: &DeviceTensor<B>,
    kc: &DeviceTensor<B>,
    vc: &DeviceTensor<B>,
    gc: &DeviceTensor<B>,
    beta_col: &DeviceTensor<B>,
    l: usize,
) -> ChunkTerms<B> {
    let hv = gc.shape().dims()[0];
    // Inclusive cumulative sum over the chunk: G = g @ U with U[j, i] = [i >= j].
    let lower = tril_mask(gc, l, 0, 1.0, 0.0);
    let upper = lower.transpose([1, 0]);
    let cum = gc.dot_general(&upper, DotGeneral::contract([1], [0]));

    // The bias is 0 on and below the diagonal and a large negative value above it. `exp` maps
    // the masked entries to exactly 0, and their positive differences never overflow.
    let causal_bias = tril_mask(gc, l, 0, 0.0, MASKED_BIAS);
    let strict = tril_mask(gc, l, -1, 1.0, 0.0);
    let diff = cum.reshape([hv, l, 1]) - cum.reshape([hv, 1, l]);
    // exp(G_i - G_j) on and below the diagonal, 0 above.
    let decay = (diff + causal_bias).exp();

    let a = batched_dot(kc, kc, 2, 2) * &decay * beta_col * strict;
    let eye = &lower * &upper;
    let mut inv = eye - &a * off_diagonal_block_mask(gc, l, 1);
    let mut block = 2usize;
    while block < l {
        let coupling = &a * off_diagonal_block_mask(gc, l, block);
        inv = &inv - matmul(&matmul(&inv, &coupling), &inv);
        block *= 2;
    }

    let exp_cum = cum.exp().reshape([hv, l, 1]);
    let v_beta = vc * beta_col;
    let k_beta_decay = kc * (beta_col * &exp_cum);
    let last = cum.slice(s![.., l - 1..l]);
    let tail_decay = (&last - &cum).exp().reshape([hv, l, 1]);
    ChunkTerms {
        u: matmul(&inv, &v_beta),
        w: matmul(&inv, &k_beta_decay),
        scores: batched_dot(qc, kc, 2, 2) * decay,
        k_tail: kc * tail_decay,
        exp_cum,
    }
}

/// The terms of a chunk of one token, the recurrent step. The cumulative log-decay of the chunk is
/// the log-decay of its one token.
/// `A` has no entry below the diagonal, so `I + A` is `I`. The decay from the token to itself and
/// to the chunk's end is `exp(0) = 1`.
fn one_token_terms<B: PortableBackend + 'static>(
    qc: &DeviceTensor<B>,
    kc: &DeviceTensor<B>,
    vc: &DeviceTensor<B>,
    gc: &DeviceTensor<B>,
    beta_col: &DeviceTensor<B>,
) -> ChunkTerms<B> {
    let hv = gc.shape().dims()[0];
    let exp_g = gc.exp().reshape([hv, 1, 1]);
    ChunkTerms {
        u: vc * beta_col,
        w: kc * (beta_col * &exp_g),
        scores: batched_dot(qc, kc, 2, 2),
        k_tail: kc.clone(),
        exp_cum: exp_g,
    }
}
