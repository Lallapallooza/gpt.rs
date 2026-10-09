//! Hugging Face model directories: `config.json`, an optional `generation_config.json`, and the
//! weights in `model.safetensors` or in the shards that `model.safetensors.index.json` lists.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde_json::{Map, Value};

use super::{load_dtype, widen_to_f32, Checkpoint, TensorEntry, FP8_SCALE};
use crate::io::safetensors::{runtime_dtype, SafetensorsFile, StoredTensor};
use crate::model::ModelConfig;
use crate::tensor::DType;
use gpt_rs_ir::host::ExternalBytes;

/// Prefixes of the text decoder inside multimodal checkpoints. `language_model.model.` is the
/// layout older `transformers` versions saved.
const TEXT_PREFIXES: [&str; 2] = ["model.language_model.", "language_model.model."];
const TEXT_LM_HEADS: [&str; 1] = ["language_model.lm_head.weight"];

/// Input scales of FP8 matmuls, which matmuls on f32 or bf16 inputs do not use.
const FP8_ACTIVATION_SCALE: &str = ".activation_scale";

/// Where the payload of a tensor comes from: the file it is stored in, and whether it is stored in
/// another dtype than it loads as.
struct Source {
    file: usize,
    tensor: StoredTensor,
    converted: bool,
}

/// A Hugging Face model directory. A multimodal checkpoint loads as its text model: the
/// `text_config`, and the text decoder's tensors under the text model's names (`model.`,
/// `lm_head.weight`).
pub struct HfCheckpoint {
    config: ModelConfig,
    files: Vec<SafetensorsFile>,
    entries: Vec<TensorEntry>,
    sources: Vec<Source>,
}

impl HfCheckpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let raw = read_json(&dir.join("config.json"))?;
        let generation = dir.join("generation_config.json");
        let generation = generation
            .is_file()
            .then(|| read_json(&generation))
            .transpose()?;
        let text = raw.get("text_config").and_then(Value::as_object);
        let config = model_config(&raw, text, generation.as_ref())?;

        let files = weight_files(dir)?
            .iter()
            .map(|file| SafetensorsFile::open(dir.join(file)))
            .collect::<Result<Vec<_>>>()?;

        let mut entries = Vec::new();
        let mut sources = Vec::new();
        for (file, f) in files.iter().enumerate() {
            for tensor in f.tensors() {
                let Some(name) = loaded_name(&tensor.name, text.is_some()) else {
                    continue;
                };
                if tensor.name.ends_with(FP8_ACTIVATION_SCALE) {
                    continue;
                }
                let is_scale = tensor.name.ends_with(FP8_SCALE);
                // Skip tensors that the runtime has no dtype for: complex ones. Without its float
                // scale, an FP8 weight would load unscaled.
                let runtime = runtime_dtype(tensor.dtype);
                ensure!(
                    !is_scale || runtime.is_some_and(DType::is_float),
                    "{}: FP8 scale has unsupported dtype {}",
                    tensor.name,
                    tensor.dtype
                );
                let Some(stored_dtype) = runtime else {
                    continue;
                };
                ensure!(
                    !is_scale || tensor.dims.is_empty(),
                    "{}: only per-tensor FP8 scales of rank 0 are supported, got shape {:?}",
                    tensor.name,
                    tensor.dims
                );
                // Only `Linear` reads an FP8 scale, so an FP8 tensor of another rank would compute
                // unscaled.
                ensure!(
                    stored_dtype != DType::F8E4M3 || tensor.dims.len() == 2,
                    "{}: only 2-D FP8 tensors are supported, got shape {:?}",
                    tensor.name,
                    tensor.dims
                );
                let dtype = load_dtype(&tensor.name, &tensor.dims, stored_dtype);
                entries.push(TensorEntry {
                    name,
                    dims: tensor.dims.clone(),
                    dtype,
                });
                sources.push(Source {
                    file,
                    tensor: tensor.clone(),
                    converted: dtype != stored_dtype,
                });
            }
        }
        Ok(Self {
            config,
            files,
            entries,
            sources,
        })
    }
}

impl Checkpoint for HfCheckpoint {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn tensors(&self) -> &[TensorEntry] {
        &self.entries
    }

    fn bytes(&self, index: usize) -> Result<ExternalBytes> {
        let entry = &self.entries[index];
        let source = &self.sources[index];
        let raw = self.files[source.file].bytes(&source.tensor)?;
        if !source.converted {
            return Ok(raw);
        }
        let bytes = Arc::new(widen_to_f32(raw.as_slice(), source.tensor.dtype)?);
        let len = bytes.len();
        ExternalBytes::new(bytes, 0, len).map_err(|err| anyhow!("tensor {}: {err}", entry.name))
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("invalid JSON in {}", path.display()))
}

/// The weight files of `dir`: the shards of `model.safetensors.index.json`, or
/// `model.safetensors`.
fn weight_files(dir: &Path) -> Result<Vec<String>> {
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let index = read_json(&index)?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("{} has no weight_map", dir.display()))?;
        let files: BTreeSet<&str> = weight_map.values().filter_map(Value::as_str).collect();
        return Ok(files.into_iter().map(str::to_string).collect());
    }
    if dir.join("model.safetensors").is_file() {
        return Ok(vec!["model.safetensors".to_string()]);
    }
    bail!(
        "{} has neither model.safetensors nor model.safetensors.index.json",
        dir.display()
    )
}

/// The name a tensor loads under, or `None` when it is not part of the loaded model. A
/// multimodal checkpoint keeps only its text decoder and language-model head.
fn loaded_name(name: &str, multimodal: bool) -> Option<String> {
    if !multimodal {
        return Some(name.to_string());
    }
    if let Some(rest) = TEXT_PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))
    {
        return Some(format!("model.{rest}"));
    }
    if name == "lm_head.weight" || TEXT_LM_HEADS.contains(&name) {
        return Some("lm_head.weight".to_string());
    }
    None
}

/// The model config: the text config of a multimodal checkpoint (with the top-level
/// `tie_word_embeddings` when the text config has none), otherwise the whole config.
fn model_config(
    raw: &Value,
    text: Option<&Map<String, Value>>,
    generation: Option<&Value>,
) -> Result<ModelConfig> {
    let mut config = match text {
        Some(text) => {
            let mut text = text.clone();
            if let Some(tied) = raw.get("tie_word_embeddings") {
                text.entry("tie_word_embeddings").or_insert(tied.clone());
            }
            Value::Object(text)
        }
        None => raw.clone(),
    };
    let kind = config
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("config has no model_type"))?
        .to_string();
    // The stop tokens are every `eos_token_id` of the generation config, the text config, and the
    // top-level config.
    let mut eos_token_ids = Vec::new();
    for source in [generation, Some(&config), Some(raw)].into_iter().flatten() {
        let ids = match source.get("eos_token_id") {
            Some(Value::Array(ids)) => ids.clone(),
            Some(id) => vec![id.clone()],
            None => Vec::new(),
        };
        for id in ids.iter().filter_map(Value::as_u64) {
            let id = id as usize;
            if !eos_token_ids.contains(&id) {
                eos_token_ids.push(id);
            }
        }
    }
    if let Some(fields) = config.as_object_mut() {
        // Where the config was loaded from, not part of the model.
        fields.remove("_name_or_path");
    }
    Ok(ModelConfig {
        kind,
        config,
        runtime: Default::default(),
        eos_token_ids,
    })
}
