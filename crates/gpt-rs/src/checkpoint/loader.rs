use crate::backend::spec::PortableBackend;
use crate::model::{Gpt, ModelConfig};
use crate::tensor::{DType, Shape, Tensor};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"GPTRSCHK";
const VERSION: u32 = 1;

pub struct LoadedCheckpoint {
    pub config: ModelConfig,
    pub tensors: HashMap<String, Tensor>,
}

impl LoadedCheckpoint {
    pub fn into_model<B: PortableBackend>(self, backend: Arc<B>) -> Result<Gpt<B>> {
        Gpt::from_named_tensors(self.config, backend, self.tensors)
    }
}

pub struct CheckpointLoader;

impl CheckpointLoader {
    pub fn load(path: impl AsRef<Path>) -> Result<LoadedCheckpoint> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("invalid checkpoint magic header");
        }
        let version = read_u32(&mut reader)?;
        if version != VERSION {
            bail!("unsupported checkpoint version {}", version);
        }

        let config_len = read_u32(&mut reader)? as usize;
        let mut config_bytes = vec![0u8; config_len];
        reader.read_exact(&mut config_bytes)?;
        let config: ModelConfig = serde_json::from_slice(&config_bytes)?;

        let tensor_count = read_u32(&mut reader)? as usize;
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name_len = read_u32(&mut reader)? as usize;
            let mut name_bytes = vec![0u8; name_len];
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)?;

            let rank = read_u32(&mut reader)? as usize;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(read_u64(&mut reader)? as usize);
            }

            let dtype_tag = read_u32(&mut reader)?;
            let dtype = DType::from_tag(dtype_tag)
                .ok_or_else(|| anyhow!("unknown dtype tag {} in checkpoint", dtype_tag))?;
            if dtype != DType::F32 {
                bail!("only f32 tensors are currently supported");
            }
            let requires_grad = read_bool(&mut reader)?;

            let byte_len = read_u64(&mut reader)? as usize;
            let elem_size = dtype.size_in_bytes();
            if byte_len / elem_size * elem_size != byte_len {
                bail!("tensor {} data size misaligned", name);
            }
            let elem_count = byte_len / elem_size;
            let mut raw = vec![0u8; byte_len];
            reader.read_exact(&mut raw)?;
            let mut data = Vec::with_capacity(elem_count);
            for chunk in raw.chunks_exact(dtype.size_in_bytes()) {
                data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let tensor = Tensor::from_vec(Shape::new(dims), data)?.requires_grad(requires_grad);
            tensors.insert(name, tensor);
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

fn read_bool(reader: &mut impl Read) -> Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}
