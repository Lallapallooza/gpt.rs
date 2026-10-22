//! Safetensors files: reading through a memory map, and writing.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use gpt_rs_ir::host::ExternalBytes;
use gpt_rs_ir::ir::TensorType;
pub use safetensors::tensor::Dtype as StoredDtype;
use safetensors::tensor::{Dtype, SafeTensors, View};

use crate::tensor::{DType, HostTensor};

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
        let file_path = path.as_ref();
        let file = File::open(file_path)
            .with_context(|| format!("failed to open {}", file_path.display()))?;
        // SAFETY: the file must not change while it is mapped.
        let map = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        let (header_len, metadata) = SafeTensors::read_metadata(&map)
            .map_err(|err| anyhow!("invalid safetensors file {}: {err}", file_path.display()))?;
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

/// The safetensors dtypes and the runtime dtypes they hold.
const DTYPES: [(Dtype, DType); 21] = [
    (Dtype::BOOL, DType::I1),
    (Dtype::U8, DType::Ui8),
    (Dtype::I8, DType::Si8),
    (Dtype::U16, DType::Ui16),
    (Dtype::I16, DType::Si16),
    (Dtype::U32, DType::Ui32),
    (Dtype::I32, DType::Si32),
    (Dtype::U64, DType::Ui64),
    (Dtype::I64, DType::Si64),
    (Dtype::F4, DType::F4E2M1),
    (Dtype::F6_E2M3, DType::F6E2M3),
    (Dtype::F6_E3M2, DType::F6E3M2),
    (Dtype::F8_E4M3, DType::F8E4M3),
    (Dtype::F8_E5M2, DType::F8E5M2),
    (Dtype::F8_E4M3FNUZ, DType::F8E4M3Fnuz),
    (Dtype::F8_E5M2FNUZ, DType::F8E5M2Fnuz),
    (Dtype::F8_E8M0, DType::E8M0),
    (Dtype::BF16, DType::Bf16),
    (Dtype::F16, DType::F16),
    (Dtype::F32, DType::F32),
    (Dtype::F64, DType::F64),
];

/// The dtype of a stored tensor, when the runtime has it: every safetensors dtype but complex.
pub fn runtime_dtype(dtype: Dtype) -> Option<DType> {
    DTYPES
        .iter()
        .find(|(stored, _)| *stored == dtype)
        .map(|&(_, runtime)| runtime)
}

/// The safetensors dtype of `dtype`.
fn stored_dtype(dtype: DType) -> Option<Dtype> {
    DTYPES
        .iter()
        .find(|(_, runtime)| *runtime == dtype)
        .map(|&(stored, _)| stored)
}

/// Reads every tensor of the file at `path`.
pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, HostTensor>> {
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
            let ty = TensorType::new(dtype, stored.dims.iter().copied());
            let tensor = HostTensor::from_le_bytes(&ty, file.bytes(stored)?.as_slice())
                .map_err(|err| anyhow!("tensor {}: {err}", stored.name))?;
            Ok((stored.name.clone(), tensor))
        })
        .collect()
}

struct TensorData<'a> {
    tensor: &'a HostTensor,
    dtype: Dtype,
}

impl View for TensorData<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.tensor.dims
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.tensor.to_le_bytes())
    }

    fn data_len(&self) -> usize {
        self.tensor.to_le_bytes().len()
    }
}

/// Writes `tensors` to a safetensors file at `path`.
pub fn save<'a>(
    path: impl AsRef<Path>,
    tensors: impl IntoIterator<Item = (&'a str, &'a HostTensor)>,
) -> Result<()> {
    let file_path = path.as_ref();
    let views = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let dtype = stored_dtype(tensor.dtype()).ok_or_else(|| {
                anyhow!("tensor {name}: safetensors has no {} dtype", tensor.dtype())
            })?;
            Ok((name, TensorData { tensor, dtype }))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize_to_file(views, None, file_path)
        .map_err(|err| anyhow!("failed to write {}: {err}", file_path.display()))
}
