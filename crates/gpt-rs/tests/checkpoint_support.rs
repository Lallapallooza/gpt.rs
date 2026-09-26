//! Small Hugging Face model directories with random weights, for tests that load a model but do
//! not check its outputs.

// Each test crate that includes this module uses some of the models.
#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use gpt_rs::inference::decoder::PREFILL_CHUNK;
use gpt_rs::io::safetensors;
use gpt_rs::tensor::{Shape, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};

/// Context length of the models, enough for a prompt that spans two prefill chunks.
pub const CONTEXT: usize = PREFILL_CHUNK + 64;

const VOCAB: usize = 64;

/// A fresh directory under the system temp dir, removed on drop.
pub struct TempDir {
    pub path: PathBuf,
}

impl TempDir {
    pub fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}_{nanos}"));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// A two-layer `GPT2LMHeadModel` whose `lm_head` is tied to `wte`, so the checkpoint has no
/// `lm_head.weight`.
pub fn gpt2() -> Result<TempDir> {
    let d = 16;
    let mut tensors = vec![
        ("transformer.wte.weight".to_string(), vec![VOCAB, d]),
        ("transformer.wpe.weight".to_string(), vec![CONTEXT, d]),
        ("transformer.ln_f.weight".to_string(), vec![d]),
        ("transformer.ln_f.bias".to_string(), vec![d]),
    ];
    for layer in 0..2 {
        for (name, dims) in [
            ("ln_1.weight", vec![d]),
            ("ln_1.bias", vec![d]),
            ("attn.c_attn.weight", vec![d, 3 * d]),
            ("attn.c_attn.bias", vec![3 * d]),
            ("attn.c_proj.weight", vec![d, d]),
            ("attn.c_proj.bias", vec![d]),
            ("ln_2.weight", vec![d]),
            ("ln_2.bias", vec![d]),
            ("mlp.c_fc.weight", vec![d, 4 * d]),
            ("mlp.c_fc.bias", vec![4 * d]),
            ("mlp.c_proj.weight", vec![4 * d, d]),
            ("mlp.c_proj.bias", vec![d]),
        ] {
            tensors.push((format!("transformer.h.{layer}.{name}"), dims));
        }
    }
    let config = json!({
        "model_type": "gpt2",
        "n_positions": CONTEXT,
        "n_embd": d,
        "n_layer": 2,
        "n_head": 2,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "gelu_new",
    });
    write("gpt_rs_tiny_gpt2", config, &tensors)
}

/// A `Qwen3_5ForCausalLM` whose first layer is linear attention and second is full attention.
pub fn qwen3_5() -> Result<TempDir> {
    let (h, m, head_dim) = (16, 32, 8);
    // Full attention: 2 query heads and 1 key-value head. Linear attention: 1 key head and 2
    // value heads.
    let (key_dim, value_heads) = (head_dim, 2);
    let value_dim = value_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let mut tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![VOCAB, h]),
        ("model.norm.weight".to_string(), vec![h]),
        ("lm_head.weight".to_string(), vec![VOCAB, h]),
    ];
    let mixers = [
        vec![
            ("linear_attn.in_proj_qkv.weight", vec![conv_dim, h]),
            ("linear_attn.in_proj_z.weight", vec![value_dim, h]),
            ("linear_attn.in_proj_b.weight", vec![value_heads, h]),
            ("linear_attn.in_proj_a.weight", vec![value_heads, h]),
            ("linear_attn.conv1d.weight", vec![conv_dim, 1, 4]),
            ("linear_attn.dt_bias", vec![value_heads]),
            ("linear_attn.A_log", vec![value_heads]),
            ("linear_attn.norm.weight", vec![head_dim]),
            ("linear_attn.out_proj.weight", vec![h, value_dim]),
        ],
        vec![
            // The query projection also produces the output gate.
            ("self_attn.q_proj.weight", vec![2 * 2 * head_dim, h]),
            ("self_attn.k_proj.weight", vec![head_dim, h]),
            ("self_attn.v_proj.weight", vec![head_dim, h]),
            ("self_attn.o_proj.weight", vec![h, 2 * head_dim]),
            ("self_attn.q_norm.weight", vec![head_dim]),
            ("self_attn.k_norm.weight", vec![head_dim]),
        ],
    ];
    for (layer, mixer) in mixers.into_iter().enumerate() {
        let common = [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("mlp.gate_proj.weight", vec![m, h]),
            ("mlp.up_proj.weight", vec![m, h]),
            ("mlp.down_proj.weight", vec![h, m]),
        ];
        for (name, dims) in common.into_iter().chain(mixer) {
            tensors.push((format!("model.layers.{layer}.{name}"), dims));
        }
    }
    let config = json!({
        "model_type": "qwen3_5_text",
        "max_position_embeddings": CONTEXT,
        "hidden_size": h,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "rms_norm_eps": 1e-6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": head_dim,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
        "linear_num_key_heads": 1,
        "linear_num_value_heads": value_heads,
        "linear_key_head_dim": head_dim,
        "linear_value_head_dim": head_dim,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": false,
    });
    write("gpt_rs_tiny_qwen3_5", config, &tensors)
}

/// A model directory holding `config` and random tensors with the names and dims in `tensors`.
fn write(prefix: &str, config: Value, tensors: &[(String, Vec<usize>)]) -> Result<TempDir> {
    let dir = TempDir::new(prefix);
    fs::write(dir.path.join("config.json"), serde_json::to_vec(&config)?)?;
    let mut rng = StdRng::seed_from_u64(0);
    let tensors = tensors
        .iter()
        .map(|(name, dims)| {
            let len = dims.iter().product();
            let data = (0..len).map(|_| rng.gen_range(-0.3..0.3)).collect();
            Ok((
                name.as_str(),
                Tensor::from_vec(Shape::new(dims.clone()), data)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::save(
        dir.path.join("model.safetensors"),
        tensors.iter().map(|(name, tensor)| (*name, tensor)),
    )?;
    Ok(dir)
}
