//! Tensor placeholder helpers used by the PTIR DSL.

use std::marker::PhantomData;

use crate::backend::ptir_utils;
use crate::backend::spec::{DType, TensorSpec};
use crate::tensor::Shape;

/// Trait implemented for scalar types that map cleanly to PTIR dtypes.
pub trait TensorDType {
    const DTYPE: DType;
}

impl TensorDType for f32 {
    const DTYPE: DType = DType::F32;
}

impl TensorDType for f64 {
    const DTYPE: DType = DType::F64;
}

impl TensorDType for i32 {
    const DTYPE: DType = DType::Si32;
}

impl TensorDType for i64 {
    const DTYPE: DType = DType::Si64;
}

impl TensorDType for u32 {
    const DTYPE: DType = DType::Ui32;
}

impl TensorDType for u64 {
    const DTYPE: DType = DType::Ui64;
}

/// Captures type information for tensors imported into the PTIR DSL.
#[derive(Debug, Clone)]
pub struct TensorPlaceholder<T> {
    spec: TensorSpec,
    _marker: PhantomData<T>,
}

impl<T> TensorPlaceholder<T> {
    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    pub fn into_spec(self) -> TensorSpec {
        self.spec
    }
}

/// Convenience helper that produces a tensor placeholder from a shape and scalar type.
pub fn tensor<T>(shape: Shape) -> TensorPlaceholder<T>
where
    T: TensorDType,
{
    TensorPlaceholder {
        spec: ptir_utils::tensor_spec_static(T::DTYPE, shape.dims()),
        _marker: PhantomData,
    }
}
