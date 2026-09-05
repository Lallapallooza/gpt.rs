use crate::backend::spec::{ExternalBytes, PortableBackend};
use crate::io::tensor_index::{decode_index, read_header, ByteReader};
use crate::model::{Gpt, GptConfig, ModelConfig};
use crate::nn::LayerLoader;
use crate::params::{base_param_id, BaseParamId};
use crate::tensor::{DType, DeviceTensor, Shape, Tensor};
use anyhow::{anyhow, bail, ensure, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub(crate) const MAGIC: &[u8; 8] = b"GPTRSCHK";

#[derive(Clone, Debug)]
pub struct CheckpointTensorEntry {
    pub name: String,
    pub base_id: BaseParamId,
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub offset: u64,
    pub len: u64,
}

/// Random-access checkpoint reader backed by a read-only memory map.
///
/// With [`ExternalBytes`] views of the map, host-memory backends execute directly on the mapped
/// file, and the OS page cache manages the resident weight memory.
pub struct CheckpointReader {
    map: Arc<memmap2::Mmap>,
    config: ModelConfig,
    entries: Vec<CheckpointTensorEntry>,
    by_name: HashMap<String, usize>,
    by_base_id: HashMap<BaseParamId, usize>,
}

impl CheckpointReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: checkpoint files must not change in place while they are mapped. The writers in
        // this repository write a new file and rename it over the old one.
        let map = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        let mut reader = read_header(&map, MAGIC)?;
        let config_len = reader.u32()? as usize;
        let config: ModelConfig = serde_json::from_slice(reader.take(config_len)?)?;
        let index_len = reader.u32()? as usize;
        let mut index = ByteReader::new(reader.take(index_len)?);

        let mut entries = Vec::new();
        for entry in decode_index(&mut index, true, map.len() as u64)? {
            let base_id = base_param_id(&entry.name)?;
            ensure!(
                entry.base_id == 0 || entry.base_id == base_id.0,
                "tensor {} base_id mismatch: expected {:?}, got {:?}",
                entry.name,
                base_id,
                BaseParamId(entry.base_id)
            );
            entries.push(CheckpointTensorEntry {
                name: entry.name,
                base_id,
                dims: entry.dims,
                dtype: entry.dtype,
                offset: entry.offset,
                len: entry.len,
            });
        }
        let by_name = entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (entry.name.clone(), i))
            .collect();
        let by_base_id = entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (entry.base_id, i))
            .collect();
        Ok(Self {
            map,
            config,
            entries,
            by_name,
            by_base_id,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn entries(&self) -> &[CheckpointTensorEntry] {
        &self.entries
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        let idx = *self
            .by_name
            .get(name)
            .ok_or_else(|| anyhow!("tensor '{}' not found in checkpoint", name))?;
        self.get_entry(&self.entries[idx])
    }

    pub fn get_by_base_id(&self, id: BaseParamId) -> Result<Tensor> {
        self.get_entry(self.entry_by_base_id(id)?)
    }

    /// Copies the entry payload into an owned host tensor.
    pub fn get_entry(&self, entry: &CheckpointTensorEntry) -> Result<Tensor> {
        let payload = &self.map[entry.offset as usize..(entry.offset + entry.len) as usize];
        Tensor::from_le_bytes(Shape::new(entry.dims.clone()), entry.dtype, payload)
            .map_err(|err| anyhow!("tensor {}: {err}", entry.name))
    }

    pub fn entry_by_base_id(&self, id: BaseParamId) -> Result<&CheckpointTensorEntry> {
        let idx = *self
            .by_base_id
            .get(&id)
            .ok_or_else(|| anyhow!("tensor id {:?} not found in checkpoint", id))?;
        Ok(&self.entries[idx])
    }

    /// Returns a zero-copy view of the entry payload inside the memory-mapped checkpoint.
    pub fn entry_external_bytes(&self, entry: &CheckpointTensorEntry) -> Result<ExternalBytes> {
        ExternalBytes::new(self.map.clone(), entry.offset as usize, entry.len as usize)
            .map_err(|err| anyhow!("tensor {}: {err}", entry.name))
    }
}

pub struct LoadedCheckpoint {
    pub config: ModelConfig,
    pub tensors: HashMap<String, Tensor>,
}

impl LoadedCheckpoint {
    pub fn into_model<B: PortableBackend>(self, backend: Arc<B>) -> Result<Gpt<B>> {
        if self.config.kind != "gpt" {
            bail!(
                "LoadedCheckpoint::into_model only supports kind='gpt', got '{}'",
                self.config.kind
            );
        }
        let config: GptConfig = serde_json::from_value(self.config.config)
            .map_err(|err| anyhow!("invalid gpt config: {err}"))?;
        let mut tensors = self.tensors;
        let mut get = |name: &str| -> Result<DeviceTensor<B>> {
            let tensor = tensors
                .remove(name)
                .ok_or_else(|| anyhow!("missing tensor '{name}' in checkpoint"))?;
            DeviceTensor::from_host(Arc::clone(&backend), tensor)
        };
        let mut params = LayerLoader::new(Arc::clone(&backend), &mut get)
            .with_linear_input_dtype(self.config.runtime.matmul_input_dtype);
        Gpt::build(config, &mut params)
    }
}

pub struct CheckpointLoader;

impl CheckpointLoader {
    pub fn load(path: impl AsRef<Path>) -> Result<LoadedCheckpoint> {
        let reader = CheckpointReader::open(path)?;
        let tensors = reader
            .entries()
            .iter()
            .map(|entry| Ok((entry.name.clone(), reader.get_entry(entry)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(LoadedCheckpoint {
            config: reader.config.clone(),
            tensors,
        })
    }
}
