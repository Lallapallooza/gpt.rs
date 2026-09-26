//! Safetensors files: reading through a memory map, and writing.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors, View};

use crate::backend::spec::ExternalBytes;
use crate::tensor::{DType, Shape, Tensor};

pub use safetensors::tensor::Dtype as StoredDtype;

/// One tensor of a [`SafetensorsFile`].
#[derive(Clone, Debug)]
pub struct StoredTensor {
    pub name: String,
    pub dtype: Dtype,
    pub dims: Vec<usize>,
    offset: usize,
    len: usize,
}

/// A safetensors file, read through a read-only memory map.
pub struct SafetensorsFile {
    map: Arc<memmap2::Mmap>,
    tensors: Vec<StoredTensor>,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // SAFETY: the file must not change while it is mapped.
        let map = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        let (header_len, metadata) = SafeTensors::read_metadata(&map)
            .map_err(|err| anyhow!("invalid safetensors file {}: {err}", path.display()))?;
        let data_start = 8 + header_len;
        let mut tensors: Vec<StoredTensor> = metadata
            .tensors()
            .into_iter()
            .map(|(name, info)| {
                let (start, end) = info.data_offsets;
                StoredTensor {
                    name,
                    dtype: info.dtype,
                    dims: info.shape.clone(),
                    offset: data_start + start,
                    len: end - start,
                }
            })
            .collect();
        tensors.sort_by_key(|tensor| tensor.offset);
        Ok(Self { map, tensors })
    }

    pub fn tensors(&self) -> &[StoredTensor] {
        &self.tensors
    }

    /// A view of the little-endian payload of `tensor` inside the memory map.
    pub fn bytes(&self, tensor: &StoredTensor) -> Result<ExternalBytes> {
        ExternalBytes::new(self.map.clone(), tensor.offset, tensor.len)
            .map_err(|err| anyhow!("tensor {}: {err}", tensor.name))
    }
}

/// The dtype of a stored tensor, when the runtime has it.
pub fn runtime_dtype(dtype: Dtype) -> Option<DType> {
    match dtype {
        Dtype::F32 => Some(DType::F32),
        Dtype::F16 => Some(DType::F16),
        Dtype::BF16 => Some(DType::BF16),
        Dtype::I32 => Some(DType::I32),
        _ => None,
    }
}

fn stored_dtype(dtype: DType) -> Dtype {
    match dtype {
        DType::F32 => Dtype::F32,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::I32 => Dtype::I32,
    }
}

/// Reads every tensor of the file at `path`.
pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let file = SafetensorsFile::open(path)?;
    file.tensors()
        .iter()
        .map(|stored| {
            let dtype = runtime_dtype(stored.dtype).ok_or_else(|| {
                anyhow!(
                    "tensor {} has unsupported dtype {}",
                    stored.name,
                    stored.dtype
                )
            })?;
            let tensor = Tensor::from_le_bytes(
                Shape::new(stored.dims.clone()),
                dtype,
                file.bytes(stored)?.as_slice(),
            )
            .map_err(|err| anyhow!("tensor {}: {err}", stored.name))?;
            Ok((stored.name.clone(), tensor))
        })
        .collect()
}

struct TensorData<'a>(&'a Tensor);

impl View for TensorData<'_> {
    fn dtype(&self) -> Dtype {
        stored_dtype(self.0.dtype())
    }

    fn shape(&self) -> &[usize] {
        self.0.shape().dims()
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.0.to_le_bytes())
    }

    fn data_len(&self) -> usize {
        self.0.to_le_bytes().len()
    }
}

/// Writes `tensors` to a safetensors file at `path`.
pub fn save<'a>(
    path: impl AsRef<Path>,
    tensors: impl IntoIterator<Item = (&'a str, &'a Tensor)>,
) -> Result<()> {
    let path = path.as_ref();
    let views = tensors
        .into_iter()
        .map(|(name, tensor)| (name, TensorData(tensor)));
    safetensors::serialize_to_file(views, None, path)
        .map_err(|err| anyhow!("failed to write {}: {err}", path.display()))
}
