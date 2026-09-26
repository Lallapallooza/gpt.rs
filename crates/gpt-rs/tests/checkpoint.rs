mod checkpoint_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use gpt_rs::checkpoint;
use gpt_rs::io::safetensors;
use gpt_rs::runtime::{load_model_with_options, LoadOptions, LoadedModel};
use gpt_rs::tensor::{DType, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use serde_json::json;

use checkpoint_support::TempDir;

/// Writes a safetensors file of `(name, dtype, dims, payload)` tensors, for dtypes that host
/// tensors cannot hold.
fn write_raw_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, dims, payload) in tensors {
        let start = data.len();
        data.extend_from_slice(payload);
        header.insert(
            name.to_string(),
            json!({"dtype": dtype, "shape": dims, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    Ok(fs::write(path, file)?)
}

#[test]
fn sharded_directory_reads_the_tensors_of_every_shard() -> Result<()> {
    let single = checkpoint_support::gpt2()?;
    let tensors: BTreeMap<String, Tensor> =
        safetensors::load(single.path.join("model.safetensors"))?
            .into_iter()
            .collect();
    let sharded = TempDir::new("gpt_rs_hf_sharded");
    fs::copy(
        single.path.join("config.json"),
        sharded.path.join("config.json"),
    )?;
    let mut weight_map = serde_json::Map::new();
    let names: Vec<&String> = tensors.keys().collect();
    let (first, second) = names.split_at(names.len() / 2);
    for (shard, names) in [
        ("model-00001-of-00002.safetensors", first),
        ("model-00002-of-00002.safetensors", second),
    ] {
        safetensors::save(
            sharded.path.join(shard),
            names.iter().map(|name| (name.as_str(), &tensors[*name])),
        )?;
        for name in names {
            weight_map.insert(name.to_string(), shard.into());
        }
    }
    fs::write(
        sharded.path.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata": {}, "weight_map": weight_map}))?,
    )?;

    let checkpoint = checkpoint::open(&sharded.path)?;
    let read = checkpoint
        .tensors()
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            Ok((
                entry.name.clone(),
                checkpoint.bytes(index)?.as_slice().to_vec(),
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let written: BTreeMap<String, Vec<u8>> = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.to_le_bytes().to_vec()))
        .collect();
    assert_eq!(read, written);
    Ok(())
}

#[test]
fn config_overrides_replace_checkpoint_config_fields() -> Result<()> {
    let checkpoint = checkpoint_support::gpt2()?;
    let options = LoadOptions {
        config_overrides: json!({"n_layer": 1})
            .as_object()
            .cloned()
            .unwrap_or_default(),
        ..LoadOptions::default()
    };
    let backend = Arc::new(CpuPortableBackend::new());
    let loaded = load_model_with_options(backend, &checkpoint.path, options)?;
    let causal = loaded.as_causal_lm().expect("gpt2 is a causal LM");
    assert_eq!(causal.num_layers(), 1);
    Ok(())
}

#[test]
fn multimodal_fp8_checkpoint_opens_as_its_text_model() -> Result<()> {
    let dir = TempDir::new("gpt_rs_hf_multimodal");
    fs::write(
        dir.path.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "mistral3",
            "tie_word_embeddings": true,
            "text_config": {"model_type": "ministral3", "eos_token_id": 3},
        }))?,
    )?;
    fs::write(
        dir.path.join("generation_config.json"),
        serde_json::to_vec(&json!({"eos_token_id": [2, 3]}))?,
    )?;
    let bf16 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect()
    };
    // 0x38 is 1.0 and 0x40 is 2.0 in float8_e4m3fn.
    write_raw_safetensors(
        &dir.path.join("model.safetensors"),
        &[
            (
                "model.language_model.embed_tokens.weight",
                "BF16",
                vec![2, 2],
                bf16(&[1.0, 2.0, 3.0, 4.0]),
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                "F8_E4M3",
                vec![2, 2],
                vec![0x38, 0x40, 0x38, 0x40],
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight_scale_inv",
                "BF16",
                vec![],
                bf16(&[0.5]),
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.activation_scale",
                "BF16",
                vec![],
                bf16(&[1.0]),
            ),
            (
                "model.language_model.norm.weight",
                "BF16",
                vec![2],
                bf16(&[1.5, 2.5]),
            ),
            (
                "model.vision_tower.patch_conv.weight",
                "BF16",
                vec![2],
                bf16(&[0.0, 0.0]),
            ),
            (
                "lm_head.weight",
                "BF16",
                vec![2, 2],
                bf16(&[1.0, 2.0, 3.0, 4.0]),
            ),
        ],
    )?;

    let checkpoint = checkpoint::open(&dir.path)?;
    let config = checkpoint.config();
    assert_eq!(config.kind, "ministral3");
    assert_eq!(config.eos_token_ids, vec![2, 3]);
    assert_eq!(config.config["tie_word_embeddings"], json!(true));

    let read = |name: &str| -> Result<Tensor> {
        let index = checkpoint
            .tensors()
            .iter()
            .position(|entry| entry.name == name)
            .ok_or_else(|| anyhow::anyhow!("no tensor {name}"))?;
        let entry = &checkpoint.tensors()[index];
        Tensor::from_le_bytes(
            Shape::new(entry.dims.clone()),
            entry.dtype,
            checkpoint.bytes(index)?.as_slice(),
        )
    };
    let mut names: Vec<&str> = checkpoint
        .tensors()
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.norm.weight",
        ]
    );
    let up_proj = read("model.layers.0.mlp.up_proj.weight")?;
    assert_eq!(up_proj.dtype(), DType::BF16);
    assert_eq!(
        up_proj.to_le_bytes(),
        bf16(&[0.5, 1.0, 0.5, 1.0]).as_slice()
    );
    let norm = read("model.norm.weight")?;
    assert_eq!(norm.dtype(), DType::F32);
    assert_eq!(norm.data(), [1.5, 2.5]);
    Ok(())
}

#[test]
fn truncated_or_missing_weights_are_rejected() -> Result<()> {
    let checkpoint = checkpoint_support::gpt2()?;
    let weights = checkpoint.path.join("model.safetensors");
    let bytes = fs::read(&weights)?;
    fs::write(&weights, &bytes[..bytes.len() - 1])?;
    let err = checkpoint::open(&checkpoint.path)
        .err()
        .expect("a truncated payload must be rejected");
    assert!(
        format!("{err:#}").contains("invalid safetensors file"),
        "{err:#}"
    );

    fs::remove_file(&weights)?;
    let err = checkpoint::open(&checkpoint.path)
        .err()
        .expect("a directory without weights must be rejected");
    assert!(format!("{err:#}").contains("model.safetensors"), "{err:#}");
    Ok(())
}
