use crate::backend::spec::PortableBackend;
use crate::model::{Gpt, GptConfig, ModelConfig};
use crate::params::{base_param_id, BaseParamId};
use crate::tensor::{DType, Shape, Tensor};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"GPTRSCHK";
const VERSION_V2: u32 = 2;

#[derive(Clone, Debug)]
pub struct CheckpointTensorEntry {
    pub name: String,
    pub base_id: BaseParamId,
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub offset: u64,
    pub len: u64,
}

pub struct CheckpointReader {
    file: File,
    config: ModelConfig,
    entries: Vec<CheckpointTensorEntry>,
    by_name: HashMap<String, usize>,
    by_base_id: HashMap<BaseParamId, usize>,
}

impl CheckpointReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("invalid checkpoint magic header");
        }

        let version = read_u32(&mut file)?;
        match version {
            VERSION_V2 => Self::open_v2(file),
            other => bail!("unsupported checkpoint version {}", other),
        }
    }

    fn open_v2(mut file: File) -> Result<Self> {
        let config_len = read_u32(&mut file)? as usize;
        let mut config_bytes = vec![0u8; config_len];
        file.read_exact(&mut config_bytes)?;
        let config: ModelConfig = serde_json::from_slice(&config_bytes)?;

        let index_len = read_u32(&mut file)? as usize;
        let mut index_bytes = vec![0u8; index_len];
        file.read_exact(&mut index_bytes)?;
        let mut index = Cursor::new(index_bytes);
        let tensor_count = read_u32(&mut index)? as usize;

        let mut entries = Vec::with_capacity(tensor_count);
        let mut by_name = HashMap::with_capacity(tensor_count);
        let mut by_base_id = HashMap::with_capacity(tensor_count);

        for i in 0..tensor_count {
            let name_len = read_u32(&mut index)? as usize;
            let mut name_bytes = vec![0u8; name_len];
            index.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)?;

            let stored_base_id = BaseParamId(read_u128(&mut index)?);

            let rank = read_u32(&mut index)? as usize;
            let mut dims_u64 = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims_u64.push(read_u64(&mut index)?);
            }

            let dtype_tag = read_u32(&mut index)?;
            let dtype = DType::from_tag(dtype_tag)
                .ok_or_else(|| anyhow!("unknown dtype tag {} in checkpoint", dtype_tag))?;
            let _reserved = read_bool(&mut index)?;

            let offset = read_u64(&mut index)?;
            let len = read_u64(&mut index)?;

            let dims = dims_u64
                .into_iter()
                .map(|d| usize::try_from(d).map_err(|_| anyhow!("tensor {} dim overflow", name)))
                .collect::<Result<Vec<_>>>()?;

            let computed_base_id = base_param_id(&name)?;
            if stored_base_id.0 != 0 && stored_base_id != computed_base_id {
                bail!(
                    "tensor {} base_id mismatch: expected {:?}, got {:?}",
                    name,
                    computed_base_id,
                    stored_base_id
                );
            }
            let base_id = computed_base_id;

            let entry = CheckpointTensorEntry {
                name: name.clone(),
                base_id,
                dims,
                dtype,
                offset,
                len,
            };
            by_name.insert(name, i);
            by_base_id.insert(base_id, i);
            entries.push(entry);
        }

        Ok(Self {
            file,
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

    pub fn get(&mut self, name: &str) -> Result<Tensor> {
        let idx = *self
            .by_name
            .get(name)
            .ok_or_else(|| anyhow!("tensor '{}' not found in checkpoint", name))?;
        let entry = self
            .entries
            .get(idx)
            .cloned()
            .ok_or_else(|| anyhow!("tensor '{}' index out of range", name))?;
        self.read_entry(&entry)
    }

    pub fn get_by_base_id(&mut self, id: BaseParamId) -> Result<Tensor> {
        let idx = *self
            .by_base_id
            .get(&id)
            .ok_or_else(|| anyhow!("tensor id {:?} not found in checkpoint", id))?;
        let entry = self
            .entries
            .get(idx)
            .cloned()
            .ok_or_else(|| anyhow!("tensor id {:?} index out of range", id))?;
        self.read_entry(&entry)
    }

    pub fn get_entry(&mut self, entry: &CheckpointTensorEntry) -> Result<Tensor> {
        self.read_entry(entry)
    }

    fn read_entry(&mut self, entry: &CheckpointTensorEntry) -> Result<Tensor> {
        self.file.seek(SeekFrom::Start(entry.offset))?;
        let byte_len = entry.len as usize;
        let mut raw = vec![0u8; byte_len];
        self.file.read_exact(&mut raw)?;

        let tensor = match entry.dtype {
            DType::F32 => {
                if !byte_len.is_multiple_of(4) {
                    bail!("tensor {} data size misaligned", entry.name);
                }
                let elem_count = byte_len / 4;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in raw.chunks_exact(4) {
                    data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Tensor::from_vec(Shape::new(entry.dims.clone()), data)?
            }
            DType::I32 => {
                if !byte_len.is_multiple_of(4) {
                    bail!("tensor {} data size misaligned", entry.name);
                }
                let elem_count = byte_len / 4;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in raw.chunks_exact(4) {
                    data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Tensor::from_i32(Shape::new(entry.dims.clone()), data)?
            }
            DType::F16 | DType::BF16 => {
                bail!("checkpoint dtype {:?} is not supported yet", entry.dtype);
            }
        };

        Ok(tensor)
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
        Gpt::from_named_tensors(config, backend, self.tensors)
    }
}

pub struct CheckpointLoader;

impl CheckpointLoader {
    pub fn load(path: impl AsRef<Path>) -> Result<LoadedCheckpoint> {
        let mut reader = CheckpointReader::open(path)?;
        let config = reader.config.clone();
        let entries = reader.entries().to_vec();
        let mut tensors = HashMap::with_capacity(entries.len());
        for entry in entries {
            let tensor = reader.get_entry(&entry)?;
            tensors.insert(entry.name.clone(), tensor);
        }
        Ok(LoadedCheckpoint { config, tensors })
    }
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u128(reader: &mut impl Read) -> Result<u128> {
    let mut buf = [0u8; 16];
    reader.read_exact(&mut buf)?;
    Ok(u128::from_le_bytes(buf))
}

fn read_bool(reader: &mut impl Read) -> Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}
