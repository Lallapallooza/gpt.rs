use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::ops::functional::{self, DecodeKvCache};
use tch::{Kind, Tensor as TchTensor};

use super::common::*;

#[allow(clippy::too_many_arguments)]
fn run_kv_cache_attention_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    prefix: usize,
    seq: usize,
    capacity: usize,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let q = random_vec(&mut rng, query_heads * seq * head_dim);
    let k = random_vec(&mut rng, kv_heads * seq * head_dim);
    let v = random_vec(&mut rng, kv_heads * seq * head_dim);
    // Cache rows past `prefix` hold garbage that attention must never read.
    let cache_k = random_vec(&mut rng, kv_heads * capacity * head_dim);
    let cache_v = random_vec(&mut rng, kv_heads * capacity * head_dim);

    let (expected_ctx, expected_k, expected_v) = timed_torch(|| {
        let (hq, hkv, d, p, t) = (
            query_heads as i64,
            kv_heads as i64,
            head_dim as i64,
            prefix as i64,
            seq as i64,
        );
        let q_t = tch_tensor_from_vec(&[query_heads, seq, head_dim], &q);
        let k_t = tch_tensor_from_vec(&[kv_heads, seq, head_dim], &k);
        let v_t = tch_tensor_from_vec(&[kv_heads, seq, head_dim], &v);
        let ck = tch_tensor_from_vec(&[kv_heads, capacity, head_dim], &cache_k);
        let cv = tch_tensor_from_vec(&[kv_heads, capacity, head_dim], &cache_v);
        let keys = TchTensor::cat(&[ck.narrow(1, 0, p), k_t.shallow_clone()], 1);
        let values = TchTensor::cat(&[cv.narrow(1, 0, p), v_t.shallow_clone()], 1);
        let group = hq / hkv;
        let keys_q = keys.repeat_interleave_self_int(group, 0, None::<i64>);
        let values_q = values.repeat_interleave_self_int(group, 0, None::<i64>);
        let scores = q_t.matmul(&keys_q.transpose(1, 2)) * (1.0 / (d as f64).sqrt());
        let mask = TchTensor::ones([t, p + t], (Kind::Float, tch::Device::Cpu))
            .tril(p)
            .view([1, t, p + t]);
        let masked = scores.masked_fill(&mask.eq(0.0), f64::NEG_INFINITY);
        let ctx = masked
            .softmax(-1, Kind::Float)
            .matmul(&values_q)
            .transpose(0, 1)
            .reshape([t, hq * d])
            .contiguous();
        let new_k = ck.copy();
        let new_v = cv.copy();
        new_k.narrow(1, p, t).copy_(&k_t);
        new_v.narrow(1, p, t).copy_(&v_t);
        (
            tensor_to_vec(&ctx),
            tensor_to_vec(&new_k.contiguous()),
            tensor_to_vec(&new_v.contiguous()),
        )
    });

    let (actual_ctx, actual_k, actual_v, actual_len) = timed_gpt(|| {
        let cache = DecodeKvCache::new(
            device_tensor_from_data(backend, &[kv_heads, capacity, head_dim], &cache_k),
            device_tensor_from_data(backend, &[kv_heads, capacity, head_dim], &cache_v),
            prefix,
        )
        .unwrap();
        let res = functional::attention_kv_cache(
            &device_tensor_from_data(backend, &[query_heads, seq, head_dim], &q),
            &device_tensor_from_data(backend, &[kv_heads, seq, head_dim], &k),
            &device_tensor_from_data(backend, &[kv_heads, seq, head_dim], &v),
            &cache,
            &kv_cache_update_starts(backend, prefix),
        )
        .unwrap();
        (
            to_host_vec(&res.output),
            to_host_vec(res.cache.keys()),
            to_host_vec(res.cache.values()),
            res.cache.len(),
        )
    });

    assert_eq!(actual_len, prefix + seq);
    assert_close(&expected_ctx, &actual_ctx);
    assert_close_tol(&expected_k, &actual_k, 0.0, 0.0);
    assert_close_tol(&expected_v, &actual_v, 0.0, 0.0);
}

pub fn attention_kv_cache_prefill_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_kv_cache_attention_case(backend, 4, 4, 8, 0, 5, 8, 0x1111);
}

pub fn attention_kv_cache_multi_query_full_prefill_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_kv_cache_attention_case(backend, 8, 1, 4, 0, 8, 8, 0x1115);
}

pub fn attention_kv_cache_decode_step_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_kv_cache_attention_case(backend, 4, 2, 8, 3, 1, 8, 0x1112);
}

pub fn attention_kv_cache_grouped_chunk_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_kv_cache_attention_case(backend, 6, 2, 16, 5, 4, 16, 0x1113);
}

pub fn attention_kv_cache_long_cache_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    // A cache this long reaches the parallel cache-update paths of backends.
    run_kv_cache_attention_case(backend, 8, 4, 16, 700, 3, 1024, 0x1114);
}
