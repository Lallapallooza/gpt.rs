use crate::tensor::{DType, Shape, Tensor};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GPTRSTEN";
const VERSION: u32 = 1;

pub struct TensorArchive;

impl TensorArchive {
    pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("invalid tensor archive magic header");
        }
        let version = read_u32(&mut reader)?;
        if version != VERSION {
            bail!("unsupported tensor archive version {}", version);
        }

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
                .ok_or_else(|| anyhow!("unknown dtype tag {} in tensor archive", dtype_tag))?;
            let requires_grad = read_bool(&mut reader)?;

            let byte_len = read_u64(&mut reader)? as usize;
            let elem_size = dtype.size_in_bytes();
            if byte_len / elem_size * elem_size != byte_len {
                bail!("tensor {} data size misaligned", name);
            }
            let elem_count = byte_len / elem_size;
            let mut raw = vec![0u8; byte_len];
            reader.read_exact(&mut raw)?;

            let tensor = match dtype {
                DType::F32 => {
                    let mut data = Vec::with_capacity(elem_count);
                    for chunk in raw.chunks_exact(elem_size) {
                        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    Tensor::from_vec(Shape::new(dims), data)?.requires_grad(requires_grad)
                }
                DType::I32 => {
                    let mut data = Vec::with_capacity(elem_count);
                    for chunk in raw.chunks_exact(elem_size) {
                        data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    Tensor::from_i32(Shape::new(dims), data)?.requires_grad(requires_grad)
                }
                DType::F16 | DType::BF16 => {
                    bail!("tensor archive dtype {:?} is not supported yet", dtype);
                }
            };

            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }

    pub fn save(path: impl AsRef<Path>, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(tensors.len() as u32).to_le_bytes())?;

        let mut entries: Vec<(&String, &Tensor)> = tensors.iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));

        for (name, tensor) in entries {
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

            match dtype {
                DType::F32 => {
                    let data = tensor.data();
                    writer
                        .write_all(&((data.len() * dtype.size_in_bytes()) as u64).to_le_bytes())?;
                    for &value in data {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                DType::I32 => {
                    let data = tensor.data_i32();
                    writer
                        .write_all(&((data.len() * dtype.size_in_bytes()) as u64).to_le_bytes())?;
                    for &value in data {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                DType::F16 | DType::BF16 => {
                    bail!("tensor archive dtype {:?} is not supported yet", dtype);
                }
            }
        }

        writer.flush()?;
        Ok(())
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
