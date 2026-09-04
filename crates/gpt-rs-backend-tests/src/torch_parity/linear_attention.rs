//! Parity tests for the Gated DeltaNet functionals.

use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::ops::functional;
use tch::{Kind, Tensor as TchTensor};

use super::common::*;

// ---------------------------------------------------------------------------------------------
// causal depthwise conv1d
// ---------------------------------------------------------------------------------------------

/// Reference: `conv1d([state; x]^T, weight[C, 1, K], groups=C)` and the last `K - 1` rows.
fn conv_reference(
    x: &[f32],
    state: &[f32],
    weight: &[f32],
    seq: usize,
    channels: usize,
    kernel: usize,
) -> (Vec<f32>, Vec<f32>) {
    let x_t = tch_tensor_from_vec(&[seq, channels], x);
    let s_t = tch_tensor_from_vec(&[kernel - 1, channels], state);
    let window = TchTensor::cat(&[s_t, x_t], 0);
    let w_t = tch_tensor_from_vec(&[channels, 1, kernel], weight);
    let conv = window.transpose(0, 1).unsqueeze(0).conv1d(
        &w_t,
        None::<TchTensor>,
        [1],
        [0],
        [1],
        channels as i64,
    );
    let out = conv.squeeze_dim(0).transpose(0, 1).contiguous();
    let new_state = window
        .narrow(0, seq as i64, (kernel - 1) as i64)
        .contiguous();
    (tensor_to_vec(&out), tensor_to_vec(&new_state))
}

fn run_conv_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    seq: usize,
    channels: usize,
    kernel: usize,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x = random_vec(&mut rng, seq * channels);
    let state = random_vec(&mut rng, (kernel - 1) * channels);
    let weight = random_vec(&mut rng, channels * kernel);
    let (expected_out, expected_state) =
        timed_torch(|| conv_reference(&x, &state, &weight, seq, channels, kernel));
    let (actual_out, actual_state) = timed_gpt(|| {
        let res = functional::causal_conv1d(
            &device_tensor_from_data(backend, &[seq, channels], &x),
            &device_tensor_from_data(backend, &[kernel - 1, channels], &state),
            &device_tensor_from_data(backend, &[channels, kernel], &weight),
        )
        .unwrap();
        (to_host_vec(&res.output), to_host_vec(&res.state))
    });
    assert_close(&expected_out, &actual_out);
    assert_close(&expected_state, &actual_state);
}

pub fn causal_conv1d_decode_step_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv_case(backend, 1, 12, 4, 0x1109);
}

pub fn causal_conv1d_prefill_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv_case(backend, 9, 20, 4, 0x110A);
}

pub fn causal_conv1d_kernel2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv_case(backend, 5, 7, 2, 0x110B);
}

// ---------------------------------------------------------------------------------------------
// gated delta rule
// ---------------------------------------------------------------------------------------------

struct DeltaRuleInputs {
    seq: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

impl DeltaRuleInputs {
    /// Slow decays with `g` near 0 and `beta` near 1, so tokens far apart in a chunk still interact
    /// and the higher powers of the intra-chunk system contribute above the parity tolerance.
    fn with_slow_decay(mut self, seed: u64) -> Self {
        let mut rng = seeded_rng(seed);
        let n = self.seq * self.value_heads;
        self.g = random_vec_range(&mut rng, n, -0.05, -0.001);
        self.beta = random_vec_range(&mut rng, n, 0.8, 1.0);
        self
    }

    fn random(
        seq: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        zero_state: bool,
        seed: u64,
    ) -> Self {
        let mut rng = seeded_rng(seed);
        let q = random_vec(&mut rng, seq * key_heads * key_dim);
        let k = random_vec(&mut rng, seq * key_heads * key_dim);
        let v = random_vec(&mut rng, seq * value_heads * value_dim);
        // Log-decays as produced by `-exp(A_log) * softplus(a + dt_bias)`: strictly negative.
        let g = random_vec_range(&mut rng, seq * value_heads, -2.5, -0.01);
        let beta = random_vec_range(&mut rng, seq * value_heads, 0.05, 0.95);
        let state = if zero_state {
            vec![0.0; value_heads * key_dim * value_dim]
        } else {
            random_vec(&mut rng, value_heads * key_dim * value_dim)
        };
        Self {
            seq,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            q,
            k,
            v,
            g,
            beta,
            state,
        }
    }

    /// Token-by-token recurrence, as in `torch_recurrent_gated_delta_rule`.
    fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let (t, hk, hv, dk, dv) = (
            self.seq as i64,
            self.key_heads as i64,
            self.value_heads as i64,
            self.key_dim as i64,
            self.value_dim as i64,
        );
        let l2 = |x: &TchTensor| {
            let sum_sq = (x * x).sum_dim_intlist([-1i64].as_slice(), true, Kind::Float);
            x * (sum_sq + 1e-6).rsqrt()
        };
        let group = hv / hk;
        let q = l2(&tch_tensor_from_vec(
            &[self.seq, self.key_heads, self.key_dim],
            &self.q,
        ))
        .repeat_interleave_self_int(group, 1, None::<i64>)
            * (1.0 / (dk as f64).sqrt());
        let k = l2(&tch_tensor_from_vec(
            &[self.seq, self.key_heads, self.key_dim],
            &self.k,
        ))
        .repeat_interleave_self_int(group, 1, None::<i64>);
        let v = tch_tensor_from_vec(&[self.seq, self.value_heads, self.value_dim], &self.v);
        let g = tch_tensor_from_vec(&[self.seq, self.value_heads], &self.g);
        let beta = tch_tensor_from_vec(&[self.seq, self.value_heads], &self.beta);
        let mut s = tch_tensor_from_vec(
            &[self.value_heads, self.key_dim, self.value_dim],
            &self.state,
        );
        let mut outs = Vec::with_capacity(self.seq);
        for step in 0..t {
            let q_t = q.get(step); // [Hv, Dk]
            let k_t = k.get(step);
            let v_t = v.get(step); // [Hv, Dv]
            let decay = g.get(step).exp().view([hv, 1, 1]);
            let b_t = beta.get(step).view([hv, 1]);
            s *= decay;
            let kv_mem =
                (&s * k_t.unsqueeze(-1)).sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
            let delta = (v_t - kv_mem) * b_t;
            s = &s + k_t.unsqueeze(-1) * delta.unsqueeze(1);
            outs.push((&s * q_t.unsqueeze(-1)).sum_dim_intlist(
                [1i64].as_slice(),
                false,
                Kind::Float,
            ));
        }
        let out = TchTensor::stack(&outs, 0).view([t, hv, dv]).contiguous();
        (tensor_to_vec(&out), tensor_to_vec(&s.contiguous()))
    }
}

fn run_delta_rule_case<B: PortableBackend + 'static>(backend: &Arc<B>, inputs: DeltaRuleInputs) {
    let (expected_out, expected_state) = timed_torch(|| inputs.reference());
    let (actual_out, actual_state) = timed_gpt(|| {
        let DeltaRuleInputs {
            seq,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            ..
        } = inputs;
        let res = functional::gated_delta_rule(
            &device_tensor_from_data(backend, &[seq, key_heads, key_dim], &inputs.q),
            &device_tensor_from_data(backend, &[seq, key_heads, key_dim], &inputs.k),
            &device_tensor_from_data(backend, &[seq, value_heads, value_dim], &inputs.v),
            &device_tensor_from_data(backend, &[seq, value_heads], &inputs.g),
            &device_tensor_from_data(backend, &[seq, value_heads], &inputs.beta),
            &device_tensor_from_data(backend, &[value_heads, key_dim, value_dim], &inputs.state),
        )
        .unwrap();
        (to_host_vec(&res.output), to_host_vec(&res.state))
    });
    assert_close(&expected_out, &actual_out);
    assert_close(&expected_state, &actual_state);
}

pub fn gated_delta_rule_decode_step_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_delta_rule_case(
        backend,
        DeltaRuleInputs::random(1, 2, 4, 8, 6, false, 0x110D),
    );
}

pub fn gated_delta_rule_prefill_zero_state_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_delta_rule_case(
        backend,
        DeltaRuleInputs::random(7, 2, 4, 8, 6, true, 0x110E),
    );
}

pub fn gated_delta_rule_prefill_with_state_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_delta_rule_case(
        backend,
        DeltaRuleInputs::random(13, 1, 3, 5, 4, false, 0x110F),
    );
}

pub fn gated_delta_rule_multi_chunk_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    // 70 tokens span two chunks of the chunked evaluation: 64 + 6.
    run_delta_rule_case(
        backend,
        DeltaRuleInputs::random(70, 2, 4, 8, 6, false, 0x1110),
    );
}

pub fn gated_delta_rule_slow_decay_two_chunks_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    // Slow decay keeps the state carried across the chunk boundary significant.
    run_delta_rule_case(
        backend,
        DeltaRuleInputs::random(128, 1, 2, 16, 5, false, 0x1122).with_slow_decay(0x1123),
    );
}
