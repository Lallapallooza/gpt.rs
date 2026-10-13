use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use gpt_rs_ir::ir::TensorType;

use super::handle::ModelHandle;
use super::namespace::next_namespace;
use crate::backend::PortableBackend;
use crate::checkpoint::{self, Checkpoint};
use crate::model::config::ModelRuntimeConfig;
use crate::model::registry as model_registry;
use crate::params::{base_param_id, param_key, ModelNamespaceId};
use crate::tensor::{DeviceTensor, HostTensor, Shape};

/// Hands a checkpoint tensor to the backend as a view into the memory-mapped file, so a backend
/// on host memory leaves the weights in the OS page cache. An entry that the file does not align
/// for its dtype is copied, and its size is added to `copied`.
fn upload_entry<B: PortableBackend + 'static>(
    backend: &B,
    checkpoint: &dyn Checkpoint,
    index: usize,
    copied: &mut usize,
) -> Result<B::TensorHandle> {
    let entry = &checkpoint.tensors()[index];
    let bytes = checkpoint
        .bytes(index)
        .with_context(|| format!("failed to read tensor '{}'", entry.name))?;
    let ty = TensorType::new(entry.dtype, entry.dims.iter().copied());
    let align = entry.dtype.size_in_bytes().unwrap_or(1);
    let host = if (bytes.as_slice().as_ptr() as usize).is_multiple_of(align) {
        HostTensor::view(&ty, bytes)
    } else {
        *copied += bytes.len();
        HostTensor::from_le_bytes(&ty, bytes.as_slice())
    }
    .with_context(|| format!("checkpoint tensor '{}'", entry.name))?;
    backend.upload(host)
}

/// Caller-side options for [`load_model_with_options`].
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Parameter namespace. `None` allocates a fresh one.
    pub namespace: Option<ModelNamespaceId>,
    /// How the runtime executes the model (see [`ModelRuntimeConfig`]). A field that is set
    /// replaces what the checkpoint declares.
    pub runtime: ModelRuntimeConfig,
    /// Keys that replace those of the model config (the text config of a multimodal checkpoint).
    /// For example, `num_hidden_layers` loads only the first layers of a Ministral or Qwen3.5
    /// checkpoint.
    pub config_overrides: serde_json::Map<String, serde_json::Value>,
}

/// Loads the Hugging Face model directory `path` onto `backend`.
pub fn load_model<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
) -> Result<ModelHandle> {
    load_model_with_options(backend, path, LoadOptions::default())
}

/// Loads the Hugging Face model directory `path` onto `backend`, as `options` say.
pub fn load_model_with_options<B: PortableBackend + 'static>(
    backend: Arc<B>,
    path: impl AsRef<Path>,
    options: LoadOptions,
) -> Result<ModelHandle> {
    let namespace = options.namespace.unwrap_or_else(next_namespace);
    let checkpoint = checkpoint::open(&path)
        .with_context(|| format!("failed to open checkpoint {}", path.as_ref().display()))?;
    let mut config = checkpoint.config().clone();
    if let Some(dtype) = options.runtime.matmul_input_dtype {
        config.runtime.matmul_input_dtype = Some(dtype);
    }
    if !options.config_overrides.is_empty() {
        let fields = config
            .config
            .as_object_mut()
            .ok_or_else(|| anyhow!("model config is not a JSON object"))?;
        fields.extend(options.config_overrides);
    }

    let indices: HashMap<&str, usize> = checkpoint
        .tensors()
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.name.as_str(), index))
        .collect();
    let mut copied = 0usize;
    let mut get = |name: &str| -> Result<Option<DeviceTensor<B>>> {
        let Some(&index) = indices.get(name) else {
            return Ok(None);
        };
        let entry = &checkpoint.tensors()[index];
        let key = param_key(namespace, base_param_id(name)?);
        let handle = upload_entry(backend.as_ref(), checkpoint.as_ref(), index, &mut copied)?;
        DeviceTensor::from_handle(
            Arc::clone(&backend),
            Shape::new(entry.dims.clone()),
            entry.dtype,
            handle,
        )
        .as_param(key.0)
        .map(Some)
    };

    let kind = config.kind.as_str();
    let factory = model_registry::model_factory::<B>(kind)
        .ok_or_else(|| anyhow!("unsupported model kind '{kind}'"))?;
    let model = (factory)(Arc::clone(&backend), &config, &mut get)?;
    if copied > 0 {
        eprintln!(
            "gpt-rs: copied {copied} bytes of {} that the file does not align for their dtypes",
            path.as_ref().display()
        );
    }
    Ok(ModelHandle::new(
        model,
        config.eos_token_ids,
        config.runtime,
    ))
}
