use std::sync::Arc;

use gpt_rs::backend::spec::{
    Dimension, ElementwiseBinaryOp, ElementwiseUnaryOp, Function, Operation, PortableBackend,
    ReduceKind,
};
use gpt_rs::nn::layers::AttentionConfig;
use gpt_rs::ops::functional;
use gpt_rs::ops::functional::softmax_last_dim;
use gpt_rs::tensor::{DeviceTensor, Shape as DeviceShape};
use gpt_rs::DType;
use gpt_rs_backend_tests::recording_backend::RecordingBackend;

#[test]
fn softmax_last_dim_uses_ptir_dsl() {
    let backend = Arc::new(RecordingBackend::default());
    let input_shape = DeviceShape::new(vec![2, 4]);
    let input =
        DeviceTensor::from_handle(Arc::clone(&backend), input_shape.clone(), DType::F32, ());

    let result = softmax_last_dim(backend.as_ref(), &input).expect("softmax should succeed");
    assert_eq!(result.shape(), &input_shape);
    assert_eq!(result.dtype(), DType::F32);

    // Trigger program emission to record the captured IR.
    result
        .clone()
        .materialize()
        .expect("materialize lazy softmax");

    let recorded = backend
        .recorded_program()
        .expect("backend should record emitted program");
    let function = recorded
        .functions
        .iter()
        .find(|func| func.name == recorded.entry)
        .expect("captured function present");

    assert_eq!(
        function.parameters.len(),
        1,
        "softmax captures a single parameter"
    );
    assert_eq!(
        function.body.len(),
        7,
        "softmax should emit seven operations"
    );

    // reduce_max
    match &function.body[0].op {
        Operation::Reduce(spec) => {
            assert_eq!(spec.kind, ReduceKind::Max);
            assert_eq!(spec.axes, vec![1]);
            assert!(spec.keepdims);
        }
        other => panic!("expected leading reduce_max, got {other:?}"),
    }

    // broadcast max
    match &function.body[1].op {
        Operation::BroadcastTo(spec) => {
            let dims = spec.result_shape.dims().to_vec();
            assert_eq!(
                dims,
                vec![Dimension::Static(2), Dimension::Static(4)],
                "broadcast result shape should match input"
            );
        }
        other => panic!("expected broadcast after max, got {other:?}"),
    }

    // subtraction
    match &function.body[2].op {
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Sub) => {}
        other => panic!("expected subtraction, got {other:?}"),
    }

    // exponentiation
    match &function.body[3].op {
        Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp) => {}
        other => panic!("expected exp, got {other:?}"),
    }

    // reduce_sum
    match &function.body[4].op {
        Operation::Reduce(spec) => {
            assert_eq!(spec.kind, ReduceKind::Sum);
            assert_eq!(spec.axes, vec![1]);
            assert!(spec.keepdims);
        }
        other => panic!("expected reduce_sum, got {other:?}"),
    }

    // broadcast denominator
    match &function.body[5].op {
        Operation::BroadcastTo(spec) => {
            let dims = spec.result_shape.dims().to_vec();
            assert_eq!(
                dims,
                vec![Dimension::Static(2), Dimension::Static(4)],
                "broadcast result shape should match input",
            );
        }
        other => panic!("expected broadcast of reduce_sum, got {other:?}"),
    }

    // division
    match &function.body[6].op {
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Div) => {}
        other => panic!("expected final division, got {other:?}"),
    }

    assert_eq!(
        function.result_ids.last().copied(),
        Some(function.body.last().expect("body non-empty").id),
        "softmax result should correspond to final division",
    );

    let allowed = function.body.iter().all(|inst| {
        matches!(
            inst.op,
            Operation::Reduce(_)
                | Operation::BroadcastTo(_)
                | Operation::ElementwiseBinary(_)
                | Operation::ElementwiseUnary(_)
        )
    });
    assert!(
        allowed,
        "softmax_last_dim should only emit PTIR DSL-friendly ops"
    );
}

#[test]
fn dropout_emits_rng_mask_sequence() {
    let backend = Arc::new(RecordingBackend::default());
    let input = make_tensor(&backend, &[2, 4]);

    let dropped = functional::dropout(backend.as_ref(), &input, 0.25, true)
        .expect("dropout capture should succeed");
    flush_tensor(&dropped);

    let function = recorded_entry_function(&backend);
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::RngUniform(_))),
        "dropout graph should emit rng_uniform to sample masks"
    );
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::Select)),
        "dropout graph should select between scaled mask values"
    );
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::Compare(_))),
        "dropout graph must compare RNG outputs against the threshold"
    );
}

#[test]
fn matmul_emits_dot_general_instruction() {
    let backend = Arc::new(RecordingBackend::default());
    let lhs = make_tensor(&backend, &[2, 3]);
    let rhs = make_tensor(&backend, &[3, 4]);

    let result = functional::matmul(backend.as_ref(), &lhs, &rhs).expect("matmul capture");
    flush_tensor(&result);

    let function = recorded_entry_function(&backend);
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::DotGeneral(_))),
        "matmul should emit a DotGeneral operation"
    );
}

#[test]
fn layer_norm_captures_reduction_chain() {
    let backend = Arc::new(RecordingBackend::default());
    let x = make_tensor(&backend, &[2, 3, 4]);
    let gamma = make_tensor(&backend, &[4]);
    let beta = make_tensor(&backend, &[4]);

    let result = functional::layer_norm(backend.as_ref(), &x, &gamma, &beta, 1e-5)
        .expect("layer norm capture");
    // Capture the full program on the first materialization.
    flush_tensor(&result.output);
    let function = recorded_entry_function(&backend);

    // Materialize remaining outputs to keep their tensors ready for downstream tests.
    flush_tensor(&result.normalized);
    flush_tensor(&result.mean);
    flush_tensor(&result.inv_std);
    let reduce_count = function
        .body
        .iter()
        .filter(|inst| matches!(inst.op, Operation::Reduce(_)))
        .count();
    assert!(
        reduce_count >= 2,
        "layer norm should emit multiple reductions for mean/variance (got {reduce_count})"
    );
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::BroadcastTo(_))),
        "layer norm should broadcast mean/std back to the input shape"
    );
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::ElementwiseBinary(_))),
        "layer norm mixes inputs via binary ops (sub/div/mul/add)"
    );
}

#[test]
fn attention_records_context_and_cache_outputs() {
    let backend = Arc::new(RecordingBackend::default());
    let config = AttentionConfig::with_equal_heads(8, 2);
    let qkv = make_tensor(&backend, &[2, config.total_projection_dim()]);

    let attention = functional::attention(backend.as_ref(), &config, &qkv, None)
        .expect("attention capture succeeds");
    flush_tensor(&attention.output);
    flush_tensor(attention.present.keys());
    flush_tensor(attention.present.values());

    let function = recorded_entry_function(&backend);
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::DotGeneral(_))),
        "attention should express dot products through DotGeneral ops"
    );
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::Reduce(_))),
        "attention softmax path should rely on reductions"
    );
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::Select)),
        "attention masking should leverage select operations"
    );
}

#[test]
fn attention_cache_outputs_use_seq_len_only() {
    let backend = Arc::new(RecordingBackend::default());
    let config = AttentionConfig::with_kv(8, 4, 2);
    let existing_len = 3usize;
    let seq_len = 2usize;

    let qkv = make_tensor(&backend, &[seq_len, config.total_projection_dim()]);
    let cache_keys = make_tensor_for_dtype(
        &backend,
        &[config.num_key_value_heads, existing_len, config.kv_head_dim],
        DType::F32,
    );
    let cache_values = make_tensor_for_dtype(
        &backend,
        &[config.num_key_value_heads, existing_len, config.kv_head_dim],
        DType::F32,
    );

    let cache =
        functional::AttentionCache::new(cache_keys, cache_values).expect("cache tensors validated");

    let attention = functional::attention(backend.as_ref(), &config, &qkv, Some(&cache))
        .expect("attention capture succeeds");

    assert_eq!(
        attention.output.shape().dims(),
        &[seq_len, config.query_projection_dim()],
        "output should cover only new tokens"
    );
    assert_eq!(
        attention.present.keys().shape().dims(),
        &[config.num_key_value_heads, seq_len, config.kv_head_dim],
        "present cache should span the latest chunk"
    );
    assert_eq!(
        attention.cache.keys().shape().dims(),
        &[
            config.num_key_value_heads,
            existing_len + seq_len,
            config.kv_head_dim
        ],
        "combined cache should append new tokens without padding"
    );
}

#[test]
fn attention_decode_cache_emits_dynamic_slice_ops() {
    let backend = Arc::new(RecordingBackend::default());
    let config = AttentionConfig::with_equal_heads(8, 2);
    let seq_len = 1usize;
    let capacity = 8usize;

    let qkv = make_tensor(&backend, &[seq_len, config.total_projection_dim()]);
    let cache_keys = make_tensor_for_dtype(
        &backend,
        &[config.num_key_value_heads, capacity, config.kv_head_dim],
        DType::F32,
    );
    let cache_values = make_tensor_for_dtype(
        &backend,
        &[config.num_key_value_heads, capacity, config.kv_head_dim],
        DType::F32,
    );
    let cache =
        functional::DecodeKvCache::new(cache_keys, cache_values, 0).expect("cache validated");

    let update_starts = make_tensor_for_dtype(&backend, &[3], DType::I32);
    let query_start = make_tensor_for_dtype(&backend, &[1], DType::I32);

    let attention = functional::attention_decode_cache(
        backend.as_ref(),
        &config,
        &qkv,
        &cache,
        &update_starts,
        &query_start,
    )
    .expect("decode attention capture succeeds");

    assert_eq!(
        attention.output.shape().dims(),
        &[seq_len, config.query_projection_dim()],
        "decode attention output stays shape-stable"
    );
    assert_eq!(
        attention.cache.keys().shape().dims(),
        &[config.num_key_value_heads, capacity, config.kv_head_dim],
        "decode cache keeps fixed capacity"
    );
    assert_eq!(attention.cache.len(), 1, "decode cache length increments");

    flush_tensor(&attention.output);

    let function = recorded_entry_function(&backend);
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::DynamicUpdateSlice(_))),
        "decode attention should write KV via DynamicUpdateSlice"
    );
    assert!(
        function
            .body
            .iter()
            .any(|inst| matches!(inst.op, Operation::DynamicSlice(_))),
        "decode attention should derive the causal mask via DynamicSlice"
    );
}

#[test]
fn attention_allows_wider_query_projection_than_embed_dim() {
    let backend = Arc::new(RecordingBackend::default());
    let config = AttentionConfig::with_projection_dims(12, 4, 2, 8, 8);
    let seq_len = 3usize;
    let qkv = make_tensor(&backend, &[seq_len, config.total_projection_dim()]);

    let attention = functional::attention(backend.as_ref(), &config, &qkv, None)
        .expect("attention capture succeeds for asymmetric projection dims");
    assert_eq!(
        attention.output.shape().dims(),
        &[seq_len, config.query_projection_dim()],
        "attention output width must match query projection dim"
    );
    flush_tensor(&attention.output);
}

#[test]
fn attention_decode_cache_allows_wider_query_projection_than_embed_dim() {
    let backend = Arc::new(RecordingBackend::default());
    let config = AttentionConfig::with_projection_dims(12, 4, 2, 8, 8);
    let seq_len = 1usize;
    let capacity = 8usize;
    let qkv = make_tensor(&backend, &[seq_len, config.total_projection_dim()]);
    let cache_keys = make_tensor_for_dtype(
        &backend,
        &[config.num_key_value_heads, capacity, config.kv_head_dim],
        DType::F32,
    );
    let cache_values = make_tensor_for_dtype(
        &backend,
        &[config.num_key_value_heads, capacity, config.kv_head_dim],
        DType::F32,
    );
    let cache =
        functional::DecodeKvCache::new(cache_keys, cache_values, 0).expect("cache validated");
    let update_starts = make_tensor_for_dtype(&backend, &[3], DType::I32);
    let query_start = make_tensor_for_dtype(&backend, &[1], DType::I32);

    let attention = functional::attention_decode_cache(
        backend.as_ref(),
        &config,
        &qkv,
        &cache,
        &update_starts,
        &query_start,
    )
    .expect("decode attention capture succeeds for asymmetric projection dims");
    assert_eq!(
        attention.output.shape().dims(),
        &[seq_len, config.query_projection_dim()],
        "decode attention output width must match query projection dim"
    );
    assert_eq!(attention.cache.len(), 1, "decode cache length increments");
    flush_tensor(&attention.output);
}

#[test]
fn embedding_rejects_rank2_indices() {
    let backend = Arc::new(RecordingBackend::default());
    let weight = make_tensor(&backend, &[4, 8]);
    let indices = make_tensor_for_dtype(&backend, &[2, 1], DType::I32);

    let err = functional::embedding_lookup(backend.as_ref(), &weight, &indices).unwrap_err();
    assert!(
        err.to_string()
            .contains("embedding indices must have rank 1"),
        "unexpected error: {err:?}"
    );
}

fn make_tensor(backend: &Arc<RecordingBackend>, dims: &[usize]) -> DeviceTensor<RecordingBackend> {
    DeviceTensor::from_handle(
        Arc::clone(backend),
        DeviceShape::new(dims.to_vec()),
        DType::F32,
        (),
    )
}

fn make_tensor_for_dtype(
    backend: &Arc<RecordingBackend>,
    dims: &[usize],
    dtype: DType,
) -> DeviceTensor<RecordingBackend> {
    DeviceTensor::from_handle(
        Arc::clone(backend),
        DeviceShape::new(dims.to_vec()),
        dtype,
        (),
    )
}

fn flush_tensor<B: PortableBackend + 'static>(tensor: &DeviceTensor<B>) {
    tensor
        .clone()
        .materialize()
        .expect("lazy tensor materialization should succeed");
}

fn recorded_entry_function(backend: &RecordingBackend) -> Function {
    let recorded = backend
        .recorded_program()
        .expect("backend should record emitted program");
    recorded
        .functions
        .iter()
        .find(|func| func.name == recorded.entry)
        .expect("captured function present")
        .clone()
}
