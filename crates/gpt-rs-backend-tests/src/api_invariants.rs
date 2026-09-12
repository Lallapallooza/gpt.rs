use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::nn::layers::{AttentionConfig, CausalSelfAttention, RopeConfig, RopeScaling};
use gpt_rs::tensor::{Shape, Tensor};

use crate::load_layer;

fn named(name: &str, dims: &[usize]) -> (String, Tensor) {
    (name.to_string(), Tensor::ones(Shape::new(dims.to_vec())))
}

pub fn linear_loads_named_parameters<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let tensors = [named("proj.weight", &[6, 4]), named("proj.bias", &[6])];
    let layer = load_layer(backend, tensors.clone(), |p| p.linear("proj", 4, 6, true)).unwrap();
    assert_eq!(layer.weight.shape().dims(), &[6, 4]);
    assert_eq!(layer.bias.as_ref().unwrap().shape().dims(), &[6]);
    assert!(Arc::ptr_eq(&layer.weight.backend(), backend));

    let without_bias = load_layer(backend, tensors.clone(), |p| p.linear("proj", 4, 6, false));
    assert!(without_bias.unwrap().bias.is_none());

    let err = load_layer(backend, tensors, |p| p.linear("proj", 6, 4, true))
        .err()
        .expect("a transposed weight must be rejected");
    assert!(err.to_string().contains("proj.weight"), "{err}");
}

pub fn layer_norm_loads_named_parameters<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let tensors = [named("norm.weight", &[8]), named("norm.bias", &[8])];
    let layer = load_layer(backend, tensors.clone(), |p| p.layer_norm("norm", 8, 1e-5)).unwrap();
    assert_eq!(layer.weight.shape().dims(), &[8]);
    assert_eq!(layer.bias.shape().dims(), &[8]);
    assert_eq!(layer.eps, 1e-5);

    let err = load_layer(backend, tensors, |p| p.layer_norm("norm", 4, 1e-5))
        .err()
        .expect("a mismatched width must be rejected");
    assert!(err.to_string().contains("norm.weight"), "{err}");
}

pub fn embedding_loads_named_weight<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let tensors = [named("embed.weight", &[32, 16])];
    let layer = load_layer(backend, tensors.clone(), |p| p.embedding("embed", 32, 16)).unwrap();
    assert_eq!(layer.weight.shape().dims(), &[32, 16]);

    let err = load_layer(backend, tensors, |p| p.embedding("tokens", 32, 16))
        .err()
        .expect("a missing weight must be rejected");
    assert!(err.to_string().contains("tokens.weight"), "{err}");
}

pub fn causal_self_attention_validates_projection_shapes<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    assert!(AttentionConfig::with_equal_heads(8, 3).is_err());
    assert!(AttentionConfig::with_kv(8, 4, 3).is_err());
    let config = AttentionConfig::with_kv(8, 4, 2).unwrap();
    let projections = |q_rows: usize, k_rows: usize| {
        [
            named("attn.q_proj.weight", &[q_rows, 8]),
            named("attn.k_proj.weight", &[k_rows, 8]),
            named("attn.v_proj.weight", &[4, 8]),
            named("attn.o_proj.weight", &[8, 8]),
        ]
    };
    let load = |config: &AttentionConfig, q_rows: usize, k_rows: usize| {
        load_layer(backend, projections(q_rows, k_rows), |p| {
            CausalSelfAttention::load(p, "attn", config.clone())
        })
    };

    load(&config, 8, 4).unwrap();
    let err = load(&config, 8, 8)
        .err()
        .expect("mismatched k_proj must be rejected");
    assert!(err.to_string().contains("k_proj"), "{err}");

    let gated = config.clone().with_output_gate();
    let err = load(&gated, 8, 4)
        .err()
        .expect("an output gate must require twice the q_proj rows");
    assert!(err.to_string().contains("q_proj"), "{err}");
    load(&gated, 16, 4).unwrap();

    let rope = |rotary_dim| RopeConfig {
        rotary_dim,
        theta: 10_000.0,
        scaling: RopeScaling::None,
    };
    assert!(
        config.clone().with_rope(rope(3)).is_err(),
        "an odd rotary_dim must be rejected"
    );
    assert!(
        config.clone().with_rope(rope(4)).is_err(),
        "rotary_dim > head_dim must be rejected"
    );
    assert_eq!(config.with_rope(rope(2)).unwrap().rope, Some(rope(2)));
}
