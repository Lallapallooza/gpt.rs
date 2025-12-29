use crate::backend::spec::PortableBackend;
use crate::model::Gpt;
use crate::model::ModelConfig;
use crate::params::base_param_id;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GPTRSCHK";
const VERSION_V2: u32 = 2;

use crate::Tensor;

pub struct CheckpointSaver;

struct IndexEntry {
    name: String,
    base_id: u128,
    dims: Vec<u64>,
    dtype_tag: u32,
    offset_rel: u64,
    len: u64,
}

fn push_u32(dst: &mut Vec<u8>, value: u32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(dst: &mut Vec<u8>, value: u64) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn push_u128(dst: &mut Vec<u8>, value: u128) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn build_index_bytes(entries: &[IndexEntry], data_start: u64) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    push_u32(&mut out, entries.len() as u32);
    for entry in entries {
        let name_bytes = entry.name.as_bytes();
        push_u32(&mut out, name_bytes.len() as u32);
        out.extend_from_slice(name_bytes);

        push_u128(&mut out, entry.base_id);

        push_u32(&mut out, entry.dims.len() as u32);
        for dim in &entry.dims {
            push_u64(&mut out, *dim);
        }

        push_u32(&mut out, entry.dtype_tag);
        out.push(0u8);
        let offset = data_start
            .checked_add(entry.offset_rel)
            .ok_or_else(|| anyhow::anyhow!("checkpoint offset overflow"))?;
        push_u64(&mut out, offset);
        push_u64(&mut out, entry.len);
    }
    Ok(out)
}

impl CheckpointSaver {
    pub fn save<B: PortableBackend>(path: impl AsRef<Path>, model: &Gpt<B>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION_V2.to_le_bytes())?;

        let config = ModelConfig::new("gpt", serde_json::to_value(&model.config)?);
        let config_bytes = serde_json::to_vec(&config)?;
        writer.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&config_bytes)?;

        let mut params: Vec<(String, Tensor)> = Vec::new();
        model.for_each_parameter(|name, tensor| {
            let host = tensor
                .to_host()
                .with_context(|| format!("failed to export checkpoint tensor '{name}'"))?;
            params.push((name.to_string(), host));
            Ok(())
        })?;
        params.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));

        let mut entries: Vec<IndexEntry> = Vec::with_capacity(params.len());
        let mut running_offset: u64 = 0;
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

            entries.push(IndexEntry {
                name: name.clone(),
                base_id,
                dims,
                dtype_tag: dtype.tag(),
                offset_rel: running_offset,
                len,
            });
            running_offset = running_offset
                .checked_add(len)
                .ok_or_else(|| anyhow::anyhow!("checkpoint data offset overflow"))?;
        }

        let index_bytes_rel = build_index_bytes(&entries, 0)?;
        anyhow::ensure!(
            index_bytes_rel.len() <= u32::MAX as usize,
            "checkpoint index too large"
        );
        let index_len = index_bytes_rel.len() as u32;

        let data_start = (MAGIC.len() + 4 + 4 + config_bytes.len() + 4 + index_len as usize) as u64;
        let index_bytes_abs = build_index_bytes(&entries, data_start)?;
        anyhow::ensure!(
            index_bytes_abs.len() == index_bytes_rel.len(),
            "checkpoint index length mismatch after offset fixup"
        );

        writer.write_all(&index_len.to_le_bytes())?;
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
