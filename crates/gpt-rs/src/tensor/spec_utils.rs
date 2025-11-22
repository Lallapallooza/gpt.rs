//! Shared helpers for converting tensor metadata to backend specifications.

use anyhow::{bail, Result};

use crate::backend::ptir_utils;
use crate::backend::spec::{DType as BackendDType, Dimension, Shape as BackendShape, TensorSpec};
use crate::tensor::{DType as TensorDType, Shape as TensorShape};

/// Converts a frontend [`TensorShape`] into the backend shape representation.
pub(crate) fn backend_shape_from_shape(shape: &TensorShape) -> BackendShape {
    ptir_utils::shape_static(shape.dims())
}

/// Maps a frontend tensor dtype into the backend equivalent.
pub(crate) fn backend_dtype(dtype: TensorDType) -> BackendDType {
    match dtype {
        TensorDType::F32 => BackendDType::F32,
        TensorDType::F16 => BackendDType::F16,
        TensorDType::BF16 => BackendDType::Bf16,
        TensorDType::I32 => BackendDType::Si32,
    }
}

/// Maps a backend dtype into the frontend tensor variant, returning an error on unsupported types.
pub(crate) fn frontend_dtype(dtype: BackendDType) -> Result<TensorDType> {
    match dtype {
        BackendDType::F32 => Ok(TensorDType::F32),
        BackendDType::F16 => Ok(TensorDType::F16),
        BackendDType::Bf16 => Ok(TensorDType::BF16),
        BackendDType::Si32 => Ok(TensorDType::I32),
        other => bail!(
            "backend dtype {:?} is not supported in portable frontend",
            other
        ),
    }
}

/// Extracts a frontend [`TensorShape`] from a backend tensor specification.
/// Dynamic dimensions are rejected so callers catch unsupported scenarios early.
pub(crate) fn shape_from_spec(spec: &TensorSpec) -> Result<TensorShape> {
    let dims = spec
        .shape
        .dims()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(value) => Ok(*value),
            Dimension::Dynamic(sym) => {
                bail!(
                    "dynamic dimension {:?} not supported in portable frontend",
                    sym
                )
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TensorShape::new(dims))
}
