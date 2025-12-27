use crate::tensor::{DType, Shape, Tensor};
use anyhow::{anyhow, bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GPTRSTEN";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;

#[derive(Clone, Debug)]
pub struct TensorArchiveEntry {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub requires_grad: bool,
    pub offset: u64,
    pub len: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TensorArchiveIndexV2 {
    entries: Vec<TensorArchiveIndexEntryV2>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TensorArchiveIndexEntryV2 {
    name: String,
    dims: Vec<u64>,
    dtype_tag: u32,
    requires_grad: bool,
    offset: u64,
    len: u64,
}

pub struct TensorArchiveReader {
    file: File,
    entries: Vec<TensorArchiveEntry>,
    by_name: HashMap<String, usize>,
}

impl TensorArchiveReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("invalid tensor archive magic header");
        }

        let version = read_u32(&mut file)?;
        match version {
            VERSION_V1 => Self::open_v1(file),
            VERSION_V2 => Self::open_v2(file),
            other => bail!("unsupported tensor archive version {}", other),
        }
    }

    fn open_v1(mut file: File) -> Result<Self> {
        let tensor_count = read_u32(&mut file)? as usize;
        let mut entries = Vec::with_capacity(tensor_count);
        let mut by_name = HashMap::with_capacity(tensor_count);

        for i in 0..tensor_count {
            let name_len = read_u32(&mut file)? as usize;
            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)?;

            let rank = read_u32(&mut file)? as usize;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(read_u64(&mut file)?);
            }

            let dtype_tag = read_u32(&mut file)?;
            let dtype = DType::from_tag(dtype_tag)
                .ok_or_else(|| anyhow!("unknown dtype tag {} in tensor archive", dtype_tag))?;
            let requires_grad = read_bool(&mut file)?;

            let byte_len = read_u64(&mut file)?;
            let offset = file.stream_position()?;

            let skip = i64::try_from(byte_len)
                .map_err(|_| anyhow!("tensor {} data length {} out of range", name, byte_len))?;
            file.seek(SeekFrom::Current(skip))?;

            let dims_usize = dims
                .into_iter()
                .map(|d| usize::try_from(d).map_err(|_| anyhow!("tensor {} dim overflow", name)))
                .collect::<Result<Vec<_>>>()?;

            let entry = TensorArchiveEntry {
                name: name.clone(),
                dims: dims_usize,
                dtype,
                requires_grad,
                offset,
                len: byte_len,
            };
            by_name.insert(name, i);
            entries.push(entry);
        }

        Ok(Self {
            file,
            entries,
            by_name,
        })
    }

    fn open_v2(mut file: File) -> Result<Self> {
        let index_len = read_u32(&mut file)? as usize;
        let mut index_bytes = vec![0u8; index_len];
        file.read_exact(&mut index_bytes)?;
        let index: TensorArchiveIndexV2 = bincode::deserialize(&index_bytes)?;

        let mut entries = Vec::with_capacity(index.entries.len());
        let mut by_name = HashMap::with_capacity(index.entries.len());

        for (i, e) in index.entries.into_iter().enumerate() {
            let dtype = DType::from_tag(e.dtype_tag)
                .ok_or_else(|| anyhow!("unknown dtype tag {} in tensor archive", e.dtype_tag))?;
            let dims = e
                .dims
                .into_iter()
                .map(|d| usize::try_from(d).map_err(|_| anyhow!("tensor {} dim overflow", e.name)))
                .collect::<Result<Vec<_>>>()?;

            let entry = TensorArchiveEntry {
                name: e.name.clone(),
                dims,
                dtype,
                requires_grad: e.requires_grad,
                offset: e.offset,
                len: e.len,
            };
            by_name.insert(e.name, i);
            entries.push(entry);
        }

        Ok(Self {
            file,
            entries,
            by_name,
        })
    }

    pub fn entries(&self) -> &[TensorArchiveEntry] {
        &self.entries
    }

    pub fn get(&mut self, name: &str) -> Result<Tensor> {
        let idx = *self
            .by_name
            .get(name)
            .ok_or_else(|| anyhow!("tensor '{}' not found in archive", name))?;
        let entry = self
            .entries
            .get(idx)
            .cloned()
            .ok_or_else(|| anyhow!("tensor '{}' index out of range", name))?;
        self.read_entry(&entry)
    }

    pub fn get_entry(&mut self, entry: &TensorArchiveEntry) -> Result<Tensor> {
        self.read_entry(entry)
    }

    fn read_entry(&mut self, entry: &TensorArchiveEntry) -> Result<Tensor> {
        self.file.seek(SeekFrom::Start(entry.offset))?;
        let byte_len = entry.len as usize;
        let mut raw = vec![0u8; byte_len];
        self.file.read_exact(&mut raw)?;

        let tensor = match entry.dtype {
            DType::F32 => {
                ensure!(
                    byte_len.is_multiple_of(4),
                    "tensor {} data size misaligned",
                    entry.name
                );
                let elem_count = byte_len / 4;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in raw.chunks_exact(4) {
                    data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Tensor::from_vec(Shape::new(entry.dims.clone()), data)?
            }
            DType::I32 => {
                ensure!(
                    byte_len.is_multiple_of(4),
                    "tensor {} data size misaligned",
                    entry.name
                );
                let elem_count = byte_len / 4;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in raw.chunks_exact(4) {
                    data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Tensor::from_i32(Shape::new(entry.dims.clone()), data)?
            }
            DType::F16 | DType::BF16 => {
                bail!(
                    "tensor archive dtype {:?} is not supported yet",
                    entry.dtype
                );
            }
        };

        Ok(tensor.requires_grad(entry.requires_grad))
    }
}

pub struct TensorArchive;

impl TensorArchive {
    pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
        let mut reader = TensorArchiveReader::open(path)?;
        let entries = reader.entries().to_vec();
        let mut tensors = HashMap::with_capacity(entries.len());
        for entry in entries {
            let tensor = reader.get_entry(&entry)?;
            tensors.insert(entry.name.clone(), tensor);
        }
        Ok(tensors)
    }

    pub fn save(path: impl AsRef<Path>, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION_V1.to_le_bytes())?;
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

    pub fn save_v2(path: impl AsRef<Path>, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let mut entries: Vec<(&String, &Tensor)> = tensors.iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));

        let mut index_entries: Vec<TensorArchiveIndexEntryV2> = Vec::with_capacity(entries.len());
        let mut running_offset: u64 = 0;
        for (name, tensor) in &entries {
            let dtype = tensor.dtype();
            let dims = tensor
                .shape()
                .dims()
                .iter()
                .map(|&d| d as u64)
                .collect::<Vec<_>>();
            let len = match dtype {
                DType::F32 => (tensor.data().len() * dtype.size_in_bytes()) as u64,
                DType::I32 => (tensor.data_i32().len() * dtype.size_in_bytes()) as u64,
                DType::F16 | DType::BF16 => {
                    bail!("tensor archive dtype {:?} is not supported yet", dtype);
                }
            };
            index_entries.push(TensorArchiveIndexEntryV2 {
                name: (*name).clone(),
                dims,
                dtype_tag: dtype.tag(),
                requires_grad: tensor.requires_grad_flag(),
                offset: running_offset,
                len,
            });
            running_offset = running_offset
                .checked_add(len)
                .ok_or_else(|| anyhow!("tensor archive data offset overflow"))?;
        }

        let mut index = TensorArchiveIndexV2 {
            entries: index_entries,
        };
        let index_bytes_rel = bincode::serialize(&index)?;
        let index_len = index_bytes_rel.len();
        ensure!(
            index_len <= u32::MAX as usize,
            "tensor archive index too large"
        );

        let data_start = (MAGIC.len() + 4 + 4 + index_len) as u64;
        for entry in index.entries.iter_mut() {
            entry.offset = entry
                .offset
                .checked_add(data_start)
                .ok_or_else(|| anyhow!("tensor archive offset overflow"))?;
        }
        let index_bytes_abs = bincode::serialize(&index)?;
        ensure!(
            index_bytes_abs.len() == index_len,
            "tensor archive index length mismatch after offset fixup"
        );

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION_V2.to_le_bytes())?;
        writer.write_all(&(index_len as u32).to_le_bytes())?;
        writer.write_all(&index_bytes_abs)?;

        for (_name, tensor) in entries {
            match tensor.dtype() {
                DType::F32 => {
                    for &value in tensor.data() {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                DType::I32 => {
                    for &value in tensor.data_i32() {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                DType::F16 | DType::BF16 => {
                    bail!(
                        "tensor archive dtype {:?} is not supported yet",
                        tensor.dtype()
                    );
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub fn reader(path: impl AsRef<Path>) -> Result<TensorArchiveReader> {
        TensorArchiveReader::open(path)
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
