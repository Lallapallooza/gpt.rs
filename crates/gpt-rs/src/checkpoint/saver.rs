use crate::backend::spec::PortableBackend;
use crate::model::Gpt;
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GPTRSCHK";
const VERSION: u32 = 1;

use crate::Tensor;

pub struct CheckpointSaver;

impl CheckpointSaver {
    pub fn save<B: PortableBackend>(path: impl AsRef<Path>, model: &Gpt<B>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;

        let config_bytes = serde_json::to_vec(&model.config)?;
        writer.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&config_bytes)?;

        let mut params: Vec<(String, Tensor)> = Vec::new();
        model.for_each_parameter(|name, tensor| {
            let host = tensor.to_host()?;
            params.push((name.to_string(), host));
            Ok(())
        })?;
        writer.write_all(&(params.len() as u32).to_le_bytes())?;
        for (name, tensor) in params {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            let dims = tensor.shape().dims();
            writer.write_all(&(dims.len() as u32).to_le_bytes())?;
            for &dim in dims {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }

            let dtype = tensor.dtype();
            writer.write_all(&dtype.tag().to_le_bytes())?;
            writer.write_all(&[tensor.requires_grad_flag() as u8])?;

            let data = tensor.data();
            writer.write_all(&((data.len() * dtype.size_in_bytes()) as u64).to_le_bytes())?;
            for &value in data {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }
}
