use crate::backend::spec::PortableBackend;
use crate::model::Gpt;
use crate::params::base_param_id;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GPTRSCHK";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;

use crate::Tensor;

pub struct CheckpointSaver;

impl CheckpointSaver {
    pub fn save<B: PortableBackend>(path: impl AsRef<Path>, model: &Gpt<B>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION_V1.to_le_bytes())?;

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

    pub fn save_v2<B: PortableBackend>(path: impl AsRef<Path>, model: &Gpt<B>) -> Result<()> {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        struct CheckpointIndexV2 {
            entries: Vec<CheckpointIndexEntryV2>,
        }

        #[derive(Clone, Debug, Serialize, Deserialize)]
        struct CheckpointIndexEntryV2 {
            name: String,
            base_id: u128,
            dims: Vec<u64>,
            dtype_tag: u32,
            requires_grad: bool,
            offset: u64,
            len: u64,
        }

        let config_bytes = serde_json::to_vec(&model.config)?;

        let mut params: Vec<(String, Tensor)> = Vec::new();
        model.for_each_parameter(|name, tensor| {
            let host = tensor.to_host()?;
            params.push((name.to_string(), host));
            Ok(())
        })?;
        params.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));

        let mut running_offset: u64 = 0;
        let mut entries = Vec::with_capacity(params.len());
        for (name, tensor) in &params {
            let base_id = base_param_id(name)?.0;
            let dims = tensor.shape().dims().iter().map(|&d| d as u64).collect();
            let dtype = tensor.dtype();
            let len = match dtype {
                crate::tensor::DType::F32 => (tensor.data().len() * dtype.size_in_bytes()) as u64,
                crate::tensor::DType::I32 => {
                    (tensor.data_i32().len() * dtype.size_in_bytes()) as u64
                }
                crate::tensor::DType::F16 | crate::tensor::DType::BF16 => {
                    anyhow::bail!("checkpoint dtype {:?} is not supported yet", dtype);
                }
            };

            entries.push(CheckpointIndexEntryV2 {
                name: name.clone(),
                base_id,
                dims,
                dtype_tag: dtype.tag(),
                requires_grad: tensor.requires_grad_flag(),
                offset: running_offset,
                len,
            });

            running_offset = running_offset
                .checked_add(len)
                .ok_or_else(|| anyhow::anyhow!("checkpoint data offset overflow"))?;
        }

        let mut index = CheckpointIndexV2 { entries };
        let index_bytes_rel = bincode::serialize(&index)?;
        let index_len = index_bytes_rel.len();
        anyhow::ensure!(index_len <= u32::MAX as usize, "checkpoint index too large");

        let data_start = (MAGIC.len() + 4 + 4 + config_bytes.len() + 4 + index_len) as u64;
        for entry in index.entries.iter_mut() {
            entry.offset = entry
                .offset
                .checked_add(data_start)
                .ok_or_else(|| anyhow::anyhow!("checkpoint offset overflow"))?;
        }
        let index_bytes_abs = bincode::serialize(&index)?;
        anyhow::ensure!(
            index_bytes_abs.len() == index_len,
            "checkpoint index length mismatch after offset fixup"
        );

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION_V2.to_le_bytes())?;
        writer.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&config_bytes)?;
        writer.write_all(&(index_len as u32).to_le_bytes())?;
        writer.write_all(&index_bytes_abs)?;

        for (_name, tensor) in params {
            match tensor.dtype() {
                crate::tensor::DType::F32 => {
                    for &value in tensor.data() {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                crate::tensor::DType::I32 => {
                    for &value in tensor.data_i32() {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                crate::tensor::DType::F16 | crate::tensor::DType::BF16 => {
                    anyhow::bail!("checkpoint dtype {:?} is not supported yet", tensor.dtype());
                }
            }
        }

        writer.flush()?;
        Ok(())
    }
}
