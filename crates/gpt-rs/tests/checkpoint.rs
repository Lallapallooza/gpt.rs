mod checkpoint_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use gpt_rs::checkpoint;
use gpt_rs::gpt_rs_ir::ir::TensorType;
use gpt_rs::io::safetensors;
use gpt_rs::runtime::{load_model_with_options, LoadOptions};
use gpt_rs::tensor::{DType, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use serde_json::json;

use self::checkpoint_support::TempDir;

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
    let tensors: BTreeMap<String, HostTensor> =
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
        config_overrides: json!({"n_positions": 8})
            .as_object()
            .cloned()
            .unwrap_or_default(),
        ..LoadOptions::default()
    };
    let backend = Arc::new(CpuPortableBackend::new());
    let loaded = load_model_with_options(backend, &checkpoint.path, options)?;
    assert_eq!(loaded.causal_lm()?.context_length(), 8);
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

    let read = |name: &str| -> Result<HostTensor> {
        let index = checkpoint
            .tensors()
            .iter()
            .position(|entry| entry.name == name)
            .ok_or_else(|| anyhow::anyhow!("no tensor {name}"))?;
        let entry = &checkpoint.tensors()[index];
        HostTensor::from_le_bytes(
            &TensorType::new(entry.dtype, entry.dims.iter().copied()),
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
            "model.layers.0.mlp.up_proj.weight_scale_inv",
            "model.norm.weight",
        ]
    );
    // The FP8 weight and its scale keep their stored dtypes and bytes.
    let up_proj = read("model.layers.0.mlp.up_proj.weight")?;
    assert_eq!(up_proj.dtype(), DType::F8E4M3);
    assert_eq!(up_proj.to_le_bytes(), [0x38, 0x40, 0x38, 0x40]);
    let scale = read("model.layers.0.mlp.up_proj.weight_scale_inv")?;
    assert_eq!(scale.dtype(), DType::Bf16);
    assert_eq!(scale.to_le_bytes(), bf16(&[0.5]).as_slice());
    let norm = read("model.norm.weight")?;
    assert_eq!(norm.dtype(), DType::F32);
    assert_eq!(norm.as_f32().unwrap(), [1.5, 2.5]);
    Ok(())
}

#[test]
fn stored_dtypes_load_as_stored() -> Result<()> {
    let dir = TempDir::new("gpt_rs_hf_dtypes");
    fs::write(
        dir.path.join("config.json"),
        serde_json::to_vec(&json!({"model_type": "ministral3"}))?,
    )?;
    // The bytes and scales of an MXFP4 block as gpt-oss stores them, its codes as f4e2m1, a
    // BatchNorm counter and a mask.
    let tensors = [
        ("blocks", "U8", vec![2, 2], vec![0x21, 0x43, 0x65, 0x87]),
        ("scales", "F8_E8M0", vec![2], vec![127, 128]),
        ("codes", "F4", vec![4], vec![0x21, 0x43]),
        (
            "num_batches_tracked",
            "I64",
            vec![],
            7i64.to_le_bytes().to_vec(),
        ),
        ("mask", "BOOL", vec![3], vec![1, 0, 1]),
    ];
    write_raw_safetensors(&dir.path.join("model.safetensors"), &tensors)?;
    let checkpoint = checkpoint::open(&dir.path)?;
    for (name, _, dims, bytes) in &tensors {
        let index = checkpoint
            .tensors()
            .iter()
            .position(|entry| entry.name == *name)
            .ok_or_else(|| anyhow::anyhow!("no tensor {name}"))?;
        assert_eq!(checkpoint.tensors()[index].dims, *dims, "{name}");
        assert_eq!(
            checkpoint.bytes(index)?.as_slice(),
            bytes.as_slice(),
            "{name}"
        );
    }
    let dtypes: Vec<DType> = checkpoint.tensors().iter().map(|e| e.dtype).collect();
    assert_eq!(
        dtypes,
        [
            DType::Ui8,
            DType::E8M0,
            DType::F4E2M1,
            DType::Si64,
            DType::I1
        ]
    );
    Ok(())
}

#[test]
fn fp8_tensors_that_would_load_unscaled_are_rejected() -> Result<()> {
    let open_with = |weight_dims: Vec<usize>, scale_dtype: &str, scale_dims: Vec<usize>| {
        let dir = TempDir::new("gpt_rs_hf_fp8");
        fs::write(
            dir.path.join("config.json"),
            serde_json::to_vec(&json!({"model_type": "ministral3"}))?,
        )?;
        let weight = vec![0x38; weight_dims.iter().product()];
        let scale = vec![0; scale_dims.iter().product::<usize>() * 2];
        write_raw_safetensors(
            &dir.path.join("model.safetensors"),
            &[
                ("proj.weight", "F8_E4M3", weight_dims, weight),
                ("proj.weight_scale_inv", scale_dtype, scale_dims, scale),
            ],
        )?;
        let err = checkpoint::open(&dir.path)
            .err()
            .expect("the checkpoint must be rejected");
        anyhow::Ok(format!("{err:#}"))
    };
    assert_eq!(
        open_with(vec![2, 2, 2], "BF16", vec![])?,
        "proj.weight: only 2-D FP8 tensors are supported, got shape [2, 2, 2]"
    );
    assert_eq!(
        open_with(vec![2, 2], "BF16", vec![1])?,
        "proj.weight_scale_inv: only per-tensor FP8 scales of rank 0 are supported, got shape [1]"
    );
    assert_eq!(
        open_with(vec![2, 2], "I16", vec![])?,
        "proj.weight_scale_inv: FP8 scale has unsupported dtype I16"
    );
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
