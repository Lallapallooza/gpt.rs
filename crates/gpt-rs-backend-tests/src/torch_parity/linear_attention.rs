//! Parity tests for the Gated DeltaNet functionals.

use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
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
        );
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
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

impl DeltaRuleInputs {
    /// Slow decays with the log-decay near 0 and `beta` near 1, so tokens far apart in a chunk
    /// still interact and the higher powers of the intra-chunk system contribute above the parity
    /// tolerance.
    fn with_slow_decay(mut self, seed: u64) -> Self {
        let mut rng = seeded_rng(seed);
        let len = self.seq * self.value_heads;
        self.log_decay = random_vec_range(&mut rng, len, -0.05, -0.001);
        self.beta = random_vec_range(&mut rng, len, 0.8, 1.0);
        self
    }

    /// Keys repeat every `period` tokens, as in a repeated prompt. With slow decay, the powers of
    /// the strictly lower intra-chunk matrix `A` then reach large entries that cancel in the
    /// inverse of `I + A`.
    fn with_repeated_keys(mut self, period: usize) -> Self {
        let row = self.key_heads * self.key_dim;
        for token in period..self.seq {
            self.key.copy_within(
                (token - period) * row..(token - period + 1) * row,
                token * row,
            );
        }
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
        let query = random_vec(&mut rng, seq * key_heads * key_dim);
        let key = random_vec(&mut rng, seq * key_heads * key_dim);
        let value = random_vec(&mut rng, seq * value_heads * value_dim);
        // Log-decays as produced by `-exp(A_log) * softplus(a + dt_bias)`: strictly negative.
        let log_decay = random_vec_range(&mut rng, seq * value_heads, -2.5, -0.01);
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
            query,
            key,
            value,
            log_decay,
            beta,
            state,
        }
    }

    /// Token-by-token recurrence, as in `torch_recurrent_gated_delta_rule`.
    fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let (seq_len, hk, hv, dk, dv) = (
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
        let query = l2(&tch_tensor_from_vec(
            &[self.seq, self.key_heads, self.key_dim],
            &self.query,
        ))
        .repeat_interleave_self_int(group, 1, None::<i64>)
            * (1.0 / (dk as f64).sqrt());
        let key = l2(&tch_tensor_from_vec(
            &[self.seq, self.key_heads, self.key_dim],
            &self.key,
        ))
        .repeat_interleave_self_int(group, 1, None::<i64>);
        let value = tch_tensor_from_vec(&[self.seq, self.value_heads, self.value_dim], &self.value);
        let log_decay = tch_tensor_from_vec(&[self.seq, self.value_heads], &self.log_decay);
        let beta = tch_tensor_from_vec(&[self.seq, self.value_heads], &self.beta);
        let mut state = tch_tensor_from_vec(
            &[self.value_heads, self.key_dim, self.value_dim],
            &self.state,
        );
        let mut outs = Vec::with_capacity(self.seq);
        for step in 0..seq_len {
            let q_t = query.get(step); // [Hv, Dk]
            let k_t = key.get(step);
            let v_t = value.get(step); // [Hv, Dv]
            let decay = log_decay.get(step).exp().view([hv, 1, 1]);
            let b_t = beta.get(step).view([hv, 1]);
            state *= decay;
            let kv_mem =
                (&state * k_t.unsqueeze(-1)).sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
            let delta = (v_t - kv_mem) * b_t;
            state = &state + k_t.unsqueeze(-1) * delta.unsqueeze(1);
            outs.push((&state * q_t.unsqueeze(-1)).sum_dim_intlist(
                [1i64].as_slice(),
                false,
                Kind::Float,
            ));
        }
        let out = TchTensor::stack(&outs, 0)
            .view([seq_len, hv, dv])
            .contiguous();
        (tensor_to_vec(&out), tensor_to_vec(&state.contiguous()))
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
            &device_tensor_from_data(backend, &[seq, key_heads, key_dim], &inputs.query),
            &device_tensor_from_data(backend, &[seq, key_heads, key_dim], &inputs.key),
            &device_tensor_from_data(backend, &[seq, value_heads, value_dim], &inputs.value),
            &device_tensor_from_data(backend, &[seq, value_heads], &inputs.log_decay),
            &device_tensor_from_data(backend, &[seq, value_heads], &inputs.beta),
            &device_tensor_from_data(backend, &[value_heads, key_dim, value_dim], &inputs.state),
        );
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

pub fn gated_delta_rule_repeated_keys_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_delta_rule_case(
        backend,
        DeltaRuleInputs::random(64, 1, 2, 16, 5, true, 0x1124)
            .with_slow_decay(0x1125)
            .with_repeated_keys(2),
    );
}
