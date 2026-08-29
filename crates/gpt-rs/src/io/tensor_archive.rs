use crate::io::tensor_index::{decode_index, read_header, write_tensor_file, ByteReader};
use crate::tensor::{Shape, Tensor};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

const MAGIC: &[u8; 8] = b"GPTRSTEN";

pub struct TensorArchive;

impl TensorArchive {
    pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
        let bytes = std::fs::read(path.as_ref())?;
        let mut reader = read_header(&bytes, MAGIC)?;
        let index_len = reader.u32()? as usize;
        let mut index = ByteReader::new(reader.take(index_len)?);
        decode_index(&mut index, false, bytes.len() as u64)?
            .into_iter()
            .map(|entry| {
                let payload = &bytes[entry.offset as usize..(entry.offset + entry.len) as usize];
                let tensor = Tensor::from_le_bytes(Shape::new(entry.dims), entry.dtype, payload)
                    .map_err(|err| anyhow!("tensor {}: {err}", entry.name))?;
                Ok((entry.name, tensor))
            })
            .collect()
    }

    pub fn save(path: impl AsRef<Path>, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let mut entries: Vec<_> = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), 0, tensor))
            .collect();
        entries.sort_by_key(|entry| entry.0);
        write_tensor_file(path.as_ref(), MAGIC, &[], &entries, false)
    }
}
