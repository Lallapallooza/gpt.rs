use std::sync::Arc;

use crate::backend::spec::{
    DType, DimSymbol, Dimension, Shape, TensorLiteral, TensorSpec, ValueType,
};

/// Builds a static backend shape descriptor from explicit dimensions.
pub fn shape_static(dims: &[usize]) -> Shape {
    Shape::new(
        dims.iter()
            .copied()
            .map(Dimension::Static)
            .collect::<Vec<_>>(),
    )
}

/// Builds a mixed static/dynamic backend shape descriptor.
pub fn shape_mixed(dims: &[Option<usize>]) -> Shape {
    Shape::new(
        dims.iter()
            .enumerate()
            .map(|(idx, dim)| match dim {
                Some(value) => Dimension::Static(*value),
                None => Dimension::Dynamic(DimSymbol::new(format!("d{idx}"))),
            })
            .collect::<Vec<_>>(),
    )
}

/// Builds a tensor spec with fully static dimensions.
pub fn tensor_spec_static(dtype: DType, dims: &[usize]) -> TensorSpec {
    TensorSpec::new(dtype, shape_static(dims))
}

/// Builds a tensor spec with mixed static/dynamic dimensions.
pub fn tensor_spec_mixed(dtype: DType, dims: &[Option<usize>]) -> TensorSpec {
    TensorSpec::new(dtype, shape_mixed(dims))
}

/// Creates a zeroed tensor literal for a fully static spec.
pub fn tensor_literal_zeros(spec: TensorSpec) -> TensorLiteral {
    let byte_len = spec
        .byte_len()
        .expect("tensor_literal_zeros requires a static tensor spec");
    TensorLiteral::new(spec, Arc::<[u8]>::from(vec![0u8; byte_len]))
}

/// Creates a zeroed f32 tensor literal for the provided dimensions.
pub fn tensor_literal_f32_zeros(dims: &[usize]) -> TensorLiteral {
    tensor_literal_zeros(tensor_spec_static(DType::F32, dims))
}

/// Wraps a tensor spec as a value type.
pub fn value_type_tensor(spec: TensorSpec) -> ValueType {
    ValueType::Tensor(spec)
}
