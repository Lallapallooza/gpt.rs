//! Stored models: a model configuration plus named parameter tensors. [`open`] reads a Hugging Face
//! model directory. A [`Checkpoint`] gives the model's `model_type` as the kind, its config, and
//! its tensors under their Hugging Face names.

mod hf;

use std::path::Path;

use anyhow::{bail, ensure, Result};
use half::{bf16, f16};

use crate::backend::spec::ExternalBytes;
use crate::io::safetensors::StoredDtype;
use crate::model::ModelConfig;
use crate::tensor::DType;

pub use hf::HfCheckpoint;

/// One parameter tensor of a [`Checkpoint`], in the dtype the runtime loads it as.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl TensorEntry {
    pub fn byte_len(&self) -> usize {
        self.dims.iter().product::<usize>() * self.dtype.size_in_bytes()
    }
}

/// A stored model in some file format.
pub trait Checkpoint: Send + Sync {
    fn config(&self) -> &ModelConfig;

    fn tensors(&self) -> &[TensorEntry];

    /// Little-endian payload of `tensors()[index]` in its `dtype`: a view into the mapped file
    /// when the tensor is stored that way, a converted copy otherwise.
    fn bytes(&self, index: usize) -> Result<ExternalBytes>;
}

/// Opens the checkpoint at `path`: a Hugging Face model directory (`config.json` plus
/// safetensors).
pub fn open(path: impl AsRef<Path>) -> Result<Box<dyn Checkpoint>> {
    let path = path.as_ref();
    ensure!(
        path.join("config.json").is_file(),
        "{} is not a Hugging Face model directory (no config.json)",
        path.display()
    );
    Ok(Box::new(HfCheckpoint::open(path)?))
}

/// The dtype the runtime loads a stored tensor as. 2-D tensors (projection and embedding matrices)
/// and integer tensors keep their stored dtype, which the kernels read directly. Other float
/// tensors widen to f32, the dtype of the activations they combine with.
fn load_dtype(dims: &[usize], stored: DType) -> DType {
    if dims.len() == 2 || stored == DType::I32 {
        stored
    } else {
        DType::F32
    }
}

/// Decodes little-endian `bytes` of `dtype` to f32.
fn decode_f32(bytes: &[u8], dtype: StoredDtype) -> Result<Vec<f32>> {
    Ok(match dtype {
        StoredDtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        StoredDtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        StoredDtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        StoredDtype::F8_E4M3 => bytes.iter().map(|&b| f8_e4m3_to_f32(b)).collect(),
        other => bail!("cannot convert {other} tensors"),
    })
}

/// Encodes `values` as little-endian `dtype`.
fn encode(values: &[f32], dtype: DType) -> Result<Vec<u8>> {
    Ok(match dtype {
        DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::F16 => values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect(),
        DType::I32 => bail!("cannot convert float tensors to i32"),
    })
}

/// `float8_e4m3fn`: 4 exponent bits with bias 7, 3 mantissa bits, no infinities, and NaN at
/// `S.1111.111`.
fn f8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 3) & 0x0f);
    let mantissa = f32::from(bits & 0x07);
    if exponent == 0x0f && bits & 0x07 == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    sign * magnitude
}
