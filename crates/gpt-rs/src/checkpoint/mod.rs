//! Stored models: a model configuration plus named parameter tensors. [`open`] reads a Hugging Face
//! model directory. A [`Checkpoint`] gives the model's `model_type` as the kind, its config, and
//! its tensors under their Hugging Face names.

mod hf;

use std::path::Path;

use anyhow::{bail, ensure, Result};
use gpt_rs_ir::host::ExternalBytes;
use half::{bf16, f16};

pub use self::hf::HfCheckpoint;
use crate::io::safetensors::StoredDtype;
use crate::model::ModelConfig;
use crate::tensor::DType;

/// One parameter tensor of a [`Checkpoint`], in the dtype the runtime loads it as.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
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
    let dir = path.as_ref();
    ensure!(
        dir.join("config.json").is_file(),
        "{} is not a Hugging Face model directory (no config.json)",
        dir.display()
    );
    Ok(Box::new(HfCheckpoint::open(dir)?))
}

/// Suffix of the per-tensor scale of an FP8 weight.
const FP8_SCALE: &str = "_scale_inv";

/// The dtype the runtime loads a stored tensor as. 2-D tensors (projection and embedding matrices)
/// keep their stored dtype, which the kernels read directly, as do tensors of every dtype but f16
/// and bf16. The scale of an FP8 weight also keeps its dtype, because the weight dequantizes to
/// that dtype. Other f16 and bf16 tensors widen to f32, the dtype of the activations they combine
/// with.
fn load_dtype(name: &str, dims: &[usize], stored: DType) -> DType {
    let widened =
        matches!(stored, DType::F16 | DType::Bf16) && dims.len() != 2 && !name.ends_with(FP8_SCALE);
    if widened {
        DType::F32
    } else {
        stored
    }
}

/// Widens little-endian f16 or bf16 `bytes` to little-endian f32.
fn widen_to_f32(bytes: &[u8], dtype: StoredDtype) -> Result<Vec<u8>> {
    let widen: fn([u8; 2]) -> f32 = match dtype {
        StoredDtype::F16 => |element| f16::from_le_bytes(element).to_f32(),
        StoredDtype::BF16 => |element| bf16::from_le_bytes(element).to_f32(),
        other => bail!("cannot widen {other} tensors to f32"),
    };
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|&element| widen(element).to_le_bytes())
        .collect())
}
